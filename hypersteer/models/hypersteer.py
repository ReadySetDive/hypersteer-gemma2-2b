import gc
import os
import random
from contextlib import nullcontext
from typing import Any

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyvene import IntervenableConfig, IntervenableModel
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from hypersteer.data.utils import get_batch_locs, make_data_module
from hypersteer.training import ResumableRandomSampler, TrainerMixin
from hypersteer.utils.debug_utils import debug_print
from hypersteer.utils.helpers import (
    configure_tokenizer_model,
    get_logger,
    set_default_device,
)
from hypersteer.utils.model_utils import calculate_perplexity
from hypersteer.utils.visualization import Visualizer

from .hypernet.configuration_hypernet import HypernetConfig
from .hypernet.modeling_hypernet import HypernetModel
from .model import Model
from .modules.interventions import HyperAdditiveIntervention
from .modules.registry import register_model

logger = get_logger(__name__)


class RegressionWrapper(nn.Module):
    def __init__(self, base_model, hidden_size, output_dim):
        super().__init__()
        self.base_model = base_model
        self.regression_head = nn.Linear(hidden_size, output_dim)

    def forward(
        self,
        input_ids,
        attention_mask,
        output_attentions=False,
        normalize=False,
    ):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=True,
        )
        if isinstance(outputs, tuple):
            last_hiddens = outputs[0].hidden_states[-1]
        else:
            last_hiddens = outputs.hidden_states[-1]
        last_token_representations = last_hiddens[:, -1]
        preds = self.regression_head(last_token_representations)
        if normalize:
            preds = F.normalize(preds, p=2, dim=-1)
        if output_attentions:
            return preds, outputs[1:][1]
        return preds


@register_model("HyperSteer")
class HyperSteer(Model, TrainerMixin):
    """Base HyperSteer model implementation. Supports various hypernet types."""

    def __str__(self):
        return "HyperSteer"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visualizer = Visualizer()

    def make_model(self, **kwargs):
        self.ax = HyperAdditiveIntervention(
            embed_dim=self.model.config.hidden_size,
            low_rank_dimension=self.model_config.low_rank_dimension,
            use_selection_head=self.model_config.use_selection_head,
            use_ln=self.model_config.use_selection_ln,
            selection_head_start_temperature=self.model_config.selection_head_start_temperature,
            selection_head_end_temperature=self.model_config.selection_head_end_temperature,
            selection_head_learnable_temperature=self.model_config.selection_head_learnable_temperature,
            selection_head_anneal_temperature=self.model_config.selection_head_anneal_temperature,
            selection_head_add_gumbel_noise=self.model_config.selection_head_add_gumbel_noise,
            selection_head_threshold=self.model_config.selection_head_threshold,
            selection_head_straight_through=self.model_config.selection_head_straight_through,
        ).to(self.device)

        self.ax.train()

        layers = self.steering_layers if self.steering_layers else [self.layer]
        if layers == "all":
            layers = list(range(self.model.config.num_hidden_layers))
        self.num_of_layers = len(layers)
        ax_config = IntervenableConfig(
            representations=[
                {
                    "layer": layer,
                    "component": f"model.layers[{layer}].output",
                    "low_rank_dimension": self.model_config.low_rank_dimension,
                    "intervention": self.ax,
                }
                for layer in layers
            ]
        )
        ax_model = IntervenableModel(ax_config, self.model)
        ax_model.set_device(self.device)
        self.ax_model = ax_model

        self.base_model_tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.base_model_name, model_max_length=512
        )
        self.base_model_tokenizer.padding_side = "left"
        base_model_config = AutoConfig.from_pretrained(
            self.model_config.base_model_name
        )
        if self.model_config.hypernet_type == "regression":
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_config.base_model_name, torch_dtype=torch.bfloat16
            )
            configure_tokenizer_model(base_model, self.base_model_tokenizer)
            with set_default_device(self.device):
                self.concept_embedding = RegressionWrapper(
                    base_model=base_model,
                    hidden_size=base_model.config.hidden_size,
                    output_dim=self.model.config.hidden_size,
                )
        elif self.model_config.hypernet_type == "attn":
            hypernet_config = HypernetConfig(
                num_hidden_layers=self.model_config.cross_attn_hidden_layers,
                target_model_name_or_path=self.model_config.base_model_name,
                hidden_size=base_model_config.hidden_size,
                torch_dtype=torch.bfloat16,
            )
            with set_default_device(self.device):
                self.concept_embedding = HypernetModel(config=hypernet_config)

        self.concept_embedding = self.concept_embedding.to(torch.bfloat16)
        # Initialize empty concept mapping - will be populated from dataset
        self.concept_id_to_text = {}

        self.eval_steps = kwargs.get("eval_steps", 20)
        self.log_per_step = kwargs.get("log_per_step", 10)
        self.include_sentence_in_embedding = kwargs.get(
            "include_sentence_in_embedding", False
        )

    def setup_model(self):
        """
        Setup the model for training.
        # TODO: distributed training
        """
        # Load the concept embedding from the checkpoint
        self.concept_embedding.train()
        self.ax.train()

    def setup_optimizer(self):
        """Setup and return the optimizer."""
        _param_groups = [
            {
                "params": self.concept_embedding.parameters(),
                "lr": self.training_args.lr,
            },
        ]

        if self.model_config.use_selection_head:
            _param_groups.append(
                {
                    "params": [
                        p for n, p in self.ax.named_parameters() if "temperature" in n
                    ],
                    "lr": self.model_config.temperature_lr,
                }
            )
            _param_groups.append(
                {
                    "params": [
                        p
                        for n, p in self.ax.named_parameters()
                        if "temperature" not in n
                    ],
                    "lr": self.training_args.lr,
                }
            )

        optimizer = torch.optim.AdamW(
            _param_groups, weight_decay=self.training_args.weight_decay
        )
        return optimizer

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Return parameters that should be included in gradient clipping."""
        return list(self.ax.parameters()) + list(self.concept_embedding.parameters())

    def get_watchable_modules(self) -> list[nn.Module]:
        """Return modules that should be watched by wandb."""
        return [self.ax, self.concept_embedding]

    def post_backward(
        self,
        step_outputs,
        lr_scheduler,
        optimizer,
        global_step,
    ) -> dict[str, Any]:
        # Gradient clipping
        ax_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.ax.parameters(),
            self.training_args.max_grad_norm,
        )
        concept_embedding_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.concept_embedding.parameters(),
            self.training_args.max_grad_norm,
        )
        _all_params = list(self.ax.parameters()) + list(
            self.concept_embedding.parameters()
        )
        total_grad_norm = torch.nn.utils.clip_grad_norm_(
            _all_params,
            self.training_args.max_grad_norm,
        )
        step_outputs[("grad_norms", "ax")] = ax_grad_norm
        step_outputs[("grad_norms", "concept_embedding")] = concept_embedding_grad_norm
        step_outputs[("grad_norms", "total")] = total_grad_norm
        step_outputs[("lr", "main")] = lr_scheduler.get_last_lr()[0]
        if self.model_config.selection_head_learnable_temperature:
            step_outputs[("lr", "temperature")] = lr_scheduler.get_last_lr()[-1]

        if (
            self.model_config.selection_head_anneal_temperature
            and not self.model_config.selection_head_learnable_temperature
        ):
            self.ax.selection_head.step_temperature(global_step)

        return step_outputs

    def train_step(self, batch, global_step):
        """Perform a single training step."""
        inputs = {k: v.to(self.device) for k, v in batch.items()}

        unit_locations = {
            "sources->base": (
                None,
                inputs["intervention_locations"]
                .permute(1, 0, 2)
                .expand(self.num_of_layers, -1, -1)
                .tolist(),
            )
        }
        subspaces = [{"k": self.model_config.topk} for _ in range(self.num_of_layers)]

        # Compute main loss
        loss = self.compute_main_loss_and_outputs(inputs, unit_locations, subspaces)

        step_outputs = {}

        # Add selection sparsity loss if enabled
        if self.model_config.use_selection_head:
            mask = self.ax_model.full_intervention_outputs[0].payload["mask"]
            with (
                nullcontext()
                if self.model_config.compute_sparsity_loss
                else torch.no_grad()
            ):
                selection_sparsity_loss = self.compute_selection_sparsity_loss(
                    gathered_sparse_mask=mask,
                    attention_mask=inputs["attention_mask"],
                    locs=inputs["intervention_locations"],
                )
            if self.model_config.compute_sparsity_loss:
                loss += (
                    selection_sparsity_loss * self.model_config.selection_l1_loss_coeff
                )
                step_outputs[("loss", "mask_l1")] = selection_sparsity_loss

            with torch.no_grad():
                step_outputs[("metrics", "mask_sparsity")] = 1 - selection_sparsity_loss

            # Visualize sparse mask at the configured frequency
            self._visualize_training_mask(mask, inputs, global_step)

        step_outputs[("loss", "main")] = loss

        return step_outputs

    @torch.no_grad()
    def val_step(self, batch, global_step):
        """Perform a single validation step."""
        inputs = {k: v.to(self.device) for k, v in batch.items()}

        unit_locations = {
            "sources->base": (
                None,
                inputs["intervention_locations"]
                .permute(1, 0, 2)
                .expand(self.num_of_layers, -1, -1)
                .tolist(),
            )
        }
        subspaces = [{"k": self.model_config.topk} for _ in range(self.num_of_layers)]
        if self.model_config.use_selection_head:
            for subspace in subspaces:
                subspace.update({"locs": inputs["intervention_locations"]})

        # Compute concept embedding
        if self.model_config.hypernet_type == "regression":
            v = self.concept_embedding(
                inputs["concept_input_ids"],
                inputs["concept_attention_mask"],
            )
        elif self.model_config.hypernet_type == "attn":
            concept_inputs_embeds = self.model.model.embed_tokens(
                inputs["concept_input_ids"]
            )
            base_intervention_mask = inputs["labels"] == -100
            base_intervention_mask = base_intervention_mask & inputs["attention_mask"]
            base_hidden_state = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            ).hidden_states[self.layer]
            v = self.concept_embedding(
                input_ids=None,
                inputs_embeds=concept_inputs_embeds,
                attention_mask=inputs["concept_attention_mask"],
                base_encoder_hidden_states=base_hidden_state,
                base_encoder_attention_mask=base_intervention_mask,
                output_hidden_states=False,
            ).last_hidden_state

        self.ax._update_v(v)
        base_out, cf_out = self.ax_model(
            base={
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            },
            unit_locations=unit_locations,
            labels=inputs["labels"],
            subspaces=subspaces,
            use_cache=False,
            output_original_output=True,
        )

        steering_loss = cf_out.loss
        step_outputs = {("loss", "main"): steering_loss}

        # Handle selection head validation
        if self.model_config.use_selection_head:
            gathered_sparse_mask = self.ax_model.full_intervention_outputs[0].payload[
                "mask"
            ]
            selection_sparsity_loss = self.compute_selection_sparsity_loss(
                gathered_sparse_mask=gathered_sparse_mask,
                attention_mask=inputs["attention_mask"],
                locs=inputs["intervention_locations"],
            )
            step_outputs[("loss", "mask_l1")] = selection_sparsity_loss

            # Visualize validation mask
            self._visualize_validation_mask(gathered_sparse_mask, inputs, global_step)

        # Logit diff visualization
        if (
            self.model_config.logit_diff_visualization.log_heatmap
            and global_step
            % self.model_config.logit_diff_visualization.log_heatmap_freq
            == 0
        ):
            self._logit_diff_visualization(
                base_out,
                cf_out,
                inputs,
                mode="val/logit_diff",
                step=global_step,
                normalize=False,
                dump_dir=self.dump_dir,
            )

        self.ax._reset_v()
        return step_outputs

    def on_validation_start(self):
        """Called at the start of validation."""
        self.concept_embedding.eval()
        self.ax.eval()

    def on_validation_end(self):
        """Called at the end of validation."""
        self.concept_embedding.train()
        self.ax.train()

    # Helper methods for visualization
    def _visualize_training_mask(self, mask, inputs, global_step):
        """Visualize training mask."""
        batch_tokens = [
            self.tokenizer.convert_ids_to_tokens(input_id)
            for input_id in inputs["input_ids"]
        ]
        concept_strings = []
        for concept_id in inputs["concept_ids"]:
            concept_id_val = concept_id.item()
            concept_str = self.concept_id_to_text.get(
                concept_id_val, f"concept_{concept_id_val}"
            )
            concept_strings.append(concept_str)

        self._visualize_token_heatmap(
            mask.squeeze(-1),
            step=global_step,
            dump_dir=self.dump_dir or "assets/cache/sparse_masks",
            batch_tokens=batch_tokens,
            concept_ids=inputs["concept_ids"].tolist(),
            concept_strings=concept_strings,
            attention_mask=inputs["attention_mask"],
            viz_mode="train/mask",
            title_prefix="Training",
        )

    def _visualize_validation_mask(self, mask, inputs, global_step):
        """Visualize validation mask."""
        batch_tokens = [
            self.tokenizer.convert_ids_to_tokens(input_id)
            for input_id in inputs["input_ids"]
        ]
        concept_strings = []
        for concept_id in inputs["concept_ids"]:
            concept_id_val = concept_id.item()
            concept_str = self.concept_id_to_text.get(
                concept_id_val, f"concept_{concept_id_val}"
            )
            concept_strings.append(concept_str)

        self._visualize_token_heatmap(
            mask.squeeze(),
            step=global_step,
            dump_dir=self.dump_dir or "assets/cache/sparse_masks",
            batch_tokens=batch_tokens,
            concept_ids=inputs["concept_ids"].tolist(),
            concept_strings=concept_strings,
            attention_mask=inputs["attention_mask"],
            viz_mode="val/mask",
            title_prefix="Validation",
        )

    def compute_main_loss_and_outputs(
        self,
        inputs,
        unit_locations,
        subspaces,
    ):
        if self.model_config.hypernet_type == "regression":
            v = self.concept_embedding(
                inputs["concept_input_ids"],
                inputs["concept_attention_mask"],
            )
        elif self.model_config.hypernet_type == "attn":
            concept_inputs_embeds = self.model.model.embed_tokens(
                inputs["concept_input_ids"]
            )
            base_intervention_mask = inputs["labels"] == -100
            base_intervention_mask = base_intervention_mask & inputs["attention_mask"]
            base_hidden_state = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            ).hidden_states[self.layer]
            v = self.concept_embedding(
                input_ids=None,
                inputs_embeds=concept_inputs_embeds,
                attention_mask=inputs["concept_attention_mask"],
                base_encoder_hidden_states=base_hidden_state,
                base_encoder_attention_mask=base_intervention_mask,
                output_hidden_states=False,
            ).last_hidden_state

        if self.model_config.debug_print:
            debug_print(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["labels"],
                inputs["intervention_locations"],
                inputs["concept_input_ids"],
                self.tokenizer,
            )

        self.ax._update_v(v)
        base_output, cf_outputs = self.ax_model(
            base={
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            },
            unit_locations=unit_locations,
            labels=inputs["labels"],
            subspaces=subspaces,
            use_cache=False,
            output_original_output=True,
        )
        self.ax._reset_v()

        steering_loss = cf_outputs.loss

        loss = steering_loss
        return loss

    def compute_selection_sparsity_loss(
        self,
        gathered_sparse_mask=None,
        attention_mask=None,
        locs=None,
    ):
        # Scatter sparse mask to right locs in padded window
        if locs is not None:
            sparse_mask = torch.zeros_like(
                attention_mask,
                dtype=gathered_sparse_mask.dtype,
                device=gathered_sparse_mask.device,
            )

            sparse_mask.scatter_(
                -1, locs.squeeze(1).long(), gathered_sparse_mask.squeeze(-1)
            )

        total_counts = attention_mask.sum(-1)
        # L1 loss
        aux_loss = (sparse_mask * attention_mask).abs().sum(
            -1
        ) / total_counts.clamp_min(1)

        return aux_loss.mean()

    def _visualize_token_heatmap(
        self,
        mask,
        step=None,
        dump_dir=None,
        batch_tokens=None,
        concept_ids=None,
        concept_strings=None,
        attention_mask=None,
        viz_mode="sparse_mask",
        title_prefix=None,
    ):
        """
        Helper to visualize sparse mask using the configured visualization options.
        """
        if not self.model_config.mask_visualization.log_heatmap:
            return
        freq = self.model_config.mask_visualization.log_heatmap_freq
        # If step is a tuple (step, freq), use step[0] for step and step[1] for freq
        if isinstance(step, tuple):
            step, freq = step
        if step is not None and freq is not None and step % freq != 0:
            return

        self.visualizer.log_visualization(
            mask,
            step=step,
            dump_dir=dump_dir or self.dump_dir or "assets/cache/sparse_masks",
            batch_tokens=batch_tokens,
            concept_ids=concept_ids,
            concept_strings=concept_strings,
            attention_mask=attention_mask,
            viz_mode=viz_mode,
            pdf_visualization=self.model_config.mask_visualization.pdf_visualization,
            png_visualization=self.model_config.mask_visualization.png_visualization,
            title_prefix=title_prefix or "Sparse Mask",
            log_all_examples=self.model_config.mask_visualization.log_all_examples,
            log_to_console=False,
        )

    def _extract_concept_metadata_from_dataset(self, examples):
        """Extract concept ID to concept text mapping from the dataset."""
        concept_id_to_text = {}

        # Extract from output_concept column if available
        if "output_concept" in examples.columns:
            for _, row in examples.iterrows():
                concept_id = row.get("concept_id")
                concept_text = row.get("output_concept")
                if concept_id is not None and concept_text is not None:
                    concept_id_to_text[concept_id] = concept_text

        # Also check for input_concept column as fallback
        elif "input_concept" in examples.columns:
            for _, row in examples.iterrows():
                concept_id = row.get("concept_id")
                concept_text = row.get("input_concept")
                if concept_id is not None and concept_text is not None:
                    concept_id_to_text[concept_id] = concept_text

        return concept_id_to_text

    def make_dataloader(
        self, examples, rank=0, world_size=1, shuffle=True, distributed=False, **kwargs
    ):
        # Extract concept metadata from dataset before creating dataloader
        extracted_concept_mapping = self._extract_concept_metadata_from_dataset(
            examples
        )
        # Update the concept_id_to_text mapping with extracted data
        self.concept_id_to_text.update(extracted_concept_mapping)

        if distributed:
            sampler = DistributedSampler(
                examples, num_replicas=world_size, rank=rank, shuffle=shuffle
            )
            data_module = make_data_module(self.tokenizer, examples, **kwargs)
            g = torch.Generator()
            g.manual_seed(self.seed)
            train_dataloader = DataLoader(
                data_module["train_dataset"],
                batch_size=self.training_args.batch_size,
                collate_fn=data_module["data_collator"],
                sampler=sampler,
                generator=g,
                drop_last=False,
            )
        else:
            data_module = make_data_module(self.tokenizer, examples, **kwargs)
            # Seeded per-epoch permutation that can start mid-epoch, so training can
            # resume from a checkpoint on the exact next batch (Trainer.set_epoch/set_start)
            sampler = (
                ResumableRandomSampler(len(data_module["train_dataset"]), seed=self.seed)
                if shuffle
                else None
            )
            train_dataloader = DataLoader(
                data_module["train_dataset"],
                batch_size=self.training_args.batch_size,
                collate_fn=data_module["data_collator"],
                sampler=sampler,
                shuffle=False,
                # Own generator: iter() draws a base seed from it instead of the global
                # RNG, which would shift dropout masks after a resume
                generator=torch.Generator().manual_seed(self.seed),
                drop_last=False,
            )
        return train_dataloader, sampler

    def save(self, dump_dir, **kwargs):
        model_name = kwargs.get("model_name", self.__str__())
        weight_file = os.path.join(dump_dir, f"{model_name}_weight.safetensors")
        self.concept_embedding.cpu()
        save_file(self.concept_embedding.state_dict(), weight_file)

        # Save token selection (sparse_selection) if enabled
        if hasattr(self.ax, "selection_head") and self.model_config.use_selection_head:
            path = os.path.join(dump_dir, f"{model_name}_selection_head.safetensors")
            save_file(self.ax.selection_head.state_dict(), path)
            logger.debug(f"Saved selection head to {path}")

    def load(self, dump_dir=None, **kwargs):
        model_name = kwargs.get("model_name", self.__str__())
        weight_file = os.path.join(dump_dir, f"{model_name}_weight.safetensors")
        self.make_model(**kwargs)

        self.concept_embedding.load_state_dict(
            load_file(weight_file, device=str(self.device))
        )
        self.concept_embedding.to(self.device)

        # Load token selection (sparse_selection) if enabled and file exists
        if self.model_config.use_selection_head and hasattr(self.ax, "selection_head"):
            path = os.path.join(dump_dir, f"{model_name}_selection_head.safetensors")
            if os.path.exists(path):
                self.ax.selection_head.load_state_dict(
                    load_file(path, device=str(self.device))
                )
                logger.debug(f"Loaded selection head from {path}")

    def get_logits(self, concept_id, k=10):
        top_logits, neg_logits = [None], [None]

        W_U = self.model.lm_head.weight.T
        W_U = (
            W_U
            * (
                self.model.model.norm.weight
                + torch.ones_like(self.model.model.norm.weight)
            )[:, None]
        )
        W_U -= einops.reduce(W_U, "d_model d_vocab -> 1 d_vocab", "mean")

        concept_text = self.concept_id_to_text.get(concept_id)
        if concept_text is None:
            raise ValueError(f"Concept ID {concept_id} not found in concept mapping.")

        concept_input = self.base_model_tokenizer(
            concept_text,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
            truncation=True,
        ).to(self.device)

        concept_subspace = self.concept_embedding(
            concept_input["input_ids"],
            concept_input["attention_mask"],
        )

        vocab_logits = concept_subspace @ W_U
        top_values, top_indices = vocab_logits.topk(k=k, sorted=True)
        top_tokens = self.tokenizer.batch_decode(top_indices)

        top_logits = [list(zip(top_tokens, top_values.tolist()))]

        neg_values, neg_indices = vocab_logits.topk(k=k, largest=False, sorted=True)
        neg_tokens = self.tokenizer.batch_decode(neg_indices)
        neg_logits = [list(zip(neg_tokens, neg_values.tolist()))]

        return top_logits, neg_logits

    def predict_step(self, batch_examples, batch_idx, **kwargs):
        """HyperSteer-specific prediction step with concept embeddings and visualizations."""
        self.dump_dir = kwargs.get("dump_dir", None)
        eval_output_length = kwargs.get("eval_output_length", 128)
        temperature = kwargs.get("temperature", 1.0)
        # lean: only the steered generation (no unsteered copy, rescoring pass or
        # perplexity) - ~3x less compute per call, for interactive demos
        lean = kwargs.get("lean", False)

        infer_dump_dir = os.path.join(
            kwargs.get("dump_dir") or "assets/cache/sparse_masks", "inference_steer"
        )
        os.makedirs(infer_dump_dir, exist_ok=True)

        cross_attn_dump_dir = os.path.join(
            kwargs.get("dump_dir") or "assets/cache/sparse_masks", "cross_attn_heatmaps"
        )
        os.makedirs(cross_attn_dump_dir, exist_ok=True)

        input_strings = batch_examples["input"].tolist()

        mag = torch.tensor(batch_examples["factor"].tolist()).to(self.device)
        idx = torch.tensor(batch_examples["concept_id"].tolist()).to(self.device)

        # tokenize input_strings
        inputs = self.tokenizer(
            input_strings, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        if self.model_config.include_sentence_in_embedding:
            input_concept = []
            concepts = batch_examples["input_concept"].tolist()
            inputs_list = batch_examples["input"].tolist()
            assert len(concepts) == len(inputs_list)
            for concept, input_string in zip(concepts, inputs_list):
                input_concept.append(concept + " [Input] " + input_string)
        else:
            input_concept = batch_examples["input_concept"].tolist()

        concept_inputs = self.base_model_tokenizer(
            input_concept,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
            truncation=True,
        ).to(self.device)

        # --- Concept embedding (v) ---
        if self.model_config.hypernet_type == "regression":
            v = self.concept_embedding(
                concept_inputs["input_ids"],
                concept_inputs["attention_mask"],
            )
        elif self.model_config.hypernet_type == "attn":
            concept_inputs_embeds = self.model.model.embed_tokens(
                concept_inputs["input_ids"]
            )
            base_intervention_mask = inputs["attention_mask"]
            base_hidden_state = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            ).hidden_states[self.layer]

            if self.model_config.cross_attn_heatmap_visualization.log_heatmap:
                outputs = self.concept_embedding(
                    input_ids=None,
                    inputs_embeds=concept_inputs_embeds,
                    attention_mask=concept_inputs["attention_mask"],
                    base_encoder_hidden_states=base_hidden_state,
                    base_encoder_attention_mask=base_intervention_mask,
                    output_hidden_states=False,
                    output_attentions=True,
                    return_dict=True,
                )
                v = outputs.last_hidden_state
                cross_attn_weights = outputs.cross_attentions
                freq = (
                    self.model_config.cross_attn_heatmap_visualization.log_heatmap_freq
                )
                if batch_idx % freq == 0:
                    self._save_cross_attn_heatmaps(
                        cross_attn_weights,
                        inputs["input_ids"],
                        concept_inputs["input_ids"],
                        inputs["attention_mask"],
                        concept_inputs["attention_mask"],
                        cross_attn_dump_dir,
                        batch_idx,
                        tokenizer=self.tokenizer,
                        prefix="cross_attn",
                    )
            else:
                v = self.concept_embedding(
                    input_ids=None,
                    inputs_embeds=concept_inputs_embeds,
                    attention_mask=concept_inputs["attention_mask"],
                    base_encoder_hidden_states=base_hidden_state,
                    base_encoder_attention_mask=base_intervention_mask,
                    output_hidden_states=False,
                ).last_hidden_state

        # Store steering vectors for each example in the batch (move to cpu)
        v_np = v.detach().float().cpu().numpy()
        steering_vectors = [row.copy() for row in v_np]

        self.ax._update_v(v)

        locs = get_batch_locs(
            prefix_length=kwargs["prefix_length"],
            inputs=batch_examples["input"],
            attention_mask=inputs["attention_mask"],
            tokenizer=self.tokenizer,
        )

        # Always define subspaces as a list of dicts (one per layer)
        subspaces = [
            {
                "idx": idx,
                "mag": mag,
                "prefix_length": kwargs["prefix_length"],
                "locs": locs.to(self.device),
            }
            for _ in range(self.num_of_layers)
        ]

        # Generate with intervention (steered)
        base_out, steered_out = self.ax_model.generate(
            inputs,
            unit_locations=None,
            intervene_on_prompt=True,
            subspaces=subspaces,
            max_new_tokens=eval_output_length,
            # temperature 0 = greedy (deterministic); HF rejects sampling at temperature 0
            **({"do_sample": True, "temperature": temperature} if temperature > 0
               else {"do_sample": False}),
            output_original_output=not lean,
            return_dict_in_generate=True,
            output_scores=not lean,
        )

        # Get generated sequences
        if isinstance(steered_out, dict):
            generations = steered_out.get("sequences", None)
        else:
            generations = getattr(steered_out, "sequences", None)

        if lean:
            input_lengths = [len(input_ids) for input_ids in inputs.input_ids]
            generated_texts = [
                self.tokenizer.decode(g[n:], skip_special_tokens=True)
                for g, n in zip(generations, input_lengths)
            ]
            self.ax._reset_v()
            del base_out, steered_out, v, v_np
            return {"generations": generated_texts, "perplexities": None,
                    "steering_vectors": steering_vectors}

        # Forward pass through ax_model to get full logits for generated sequences
        gen_attention_mask = (generations != self.tokenizer.pad_token_id).long()
        with torch.no_grad():
            base_out_gen, steered_out_gen = self.ax_model(
                base={
                    "input_ids": generations,
                    "attention_mask": gen_attention_mask,
                },
                unit_locations=None,
                labels=None,
                subspaces=subspaces,
                use_cache=False,
                output_original_output=True,
            )

        # Logit diff visualization every N batches (on generated text)
        if (
            self.model_config.logit_diff_visualization.log_heatmap
            and batch_idx % self.model_config.logit_diff_visualization.log_heatmap_freq
            == 0
        ):
            self._logit_diff_visualization(
                base_out_gen,
                steered_out_gen,
                {
                    "input_ids": generations,
                    "attention_mask": gen_attention_mask,
                },
                mode="steer/logit_diff",
                step=batch_idx,
                normalize=False,
                dump_dir=self.dump_dir,
            )

        # Decode and print only the generated text without prompt tokens
        input_lengths = [len(input_ids) for input_ids in inputs.input_ids]
        generated_texts = [
            self.tokenizer.decode(generation[input_length:], skip_special_tokens=True)
            for generation, input_length in zip(generations, input_lengths)
        ]

        # Calculate perplexity for each sequence
        unpruned_generated_texts = [
            self.tokenizer.decode(generation, skip_special_tokens=True)
            for generation in generations
        ]

        perplexities = calculate_perplexity(
            self.model, self.tokenizer, unpruned_generated_texts, self.device
        )

        # Clear the steering vector generated for this batch
        self.ax._reset_v()

        # Cleanup
        del base_out, steered_out, base_out_gen, steered_out_gen, v, v_np
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "generations": generated_texts,
            "perplexities": perplexities,
            "steering_vectors": steering_vectors,
        }

    def _logit_diff_visualization(
        self,
        base_output,
        cf_outputs,
        inputs,
        normalize=False,
        mode="train",
        step=None,
        dump_dir=None,
    ):
        if not self.model_config.logit_diff_visualization.log_heatmap:
            return

        full_base_logits = base_output.logits
        full_cf_logits = cf_outputs.logits

        # Compute logprobs and logit diff
        cf_logps = F.log_softmax(full_cf_logits, dim=-1)
        base_logps = F.log_softmax(full_base_logits, dim=-1)
        logit_diff = cf_logps - base_logps
        logit_diff = (logit_diff * inputs["attention_mask"].unsqueeze(-1)).sum(dim=-1)

        if normalize:
            logit_diff = logit_diff / inputs["attention_mask"].sum(dim=-1)

        # Clean mode string for directory name
        mode_dir = str(mode).replace("/", "_").replace(" ", "_")
        dump_dir_final = os.path.join(dump_dir, mode_dir)

        viz_mode_str = f"{mode}/logit_diff" if mode else "train/logit_diff"
        self._visualize_token_heatmap(
            logit_diff,
            step=step,
            dump_dir=dump_dir_final,
            batch_tokens=[
                self.tokenizer.convert_ids_to_tokens(input_id)
                for input_id in inputs["input_ids"]
            ],
            attention_mask=inputs["attention_mask"],
            concept_ids=inputs["concept_ids"] if "concept_ids" in inputs else None,
            concept_strings=[
                self.concept_id_to_text.get(concept_id, f"concept_{concept_id}")
                for concept_id in inputs["concept_ids"].tolist()
            ]
            if "concept_ids" in inputs and hasattr(inputs["concept_ids"], "tolist")
            else None,
            viz_mode=viz_mode_str,
            title_prefix="Logit Diff",
        )

    def _save_cross_attn_heatmaps(
        self,
        attn_weights,
        input_ids,
        concept_ids,
        input_attention_mask,
        concept_attention_mask,
        dump_dir,
        batch_idx,
        tokenizer=None,
        prefix="cross_attn",
        max_samples=5,  # Only visualize up to 5 random samples per batch
    ):
        """
        Save cross-attention heatmap grids for a random subset of samples in a batch.
        Each grid: rows=heads, columns=layers, each cell is a heatmap.
        attn_weights: list/tuple of (num_layers,) each [batch, num_heads, q_len, kv_len]
        input_ids: [batch, seq_len] (input tokens)
        concept_ids: [batch, seq_len] (concept tokens)
        input_attention_mask: [batch, seq_len] (mask for input tokens)
        concept_attention_mask: [batch, seq_len] (mask for concept tokens)
        dump_dir: directory to save PNGs
        batch_idx: int, batch number
        tokenizer: tokenizer to decode tokens
        prefix: filename prefix
        max_samples: maximum number of samples to visualize per batch
        """

        os.makedirs(dump_dir, exist_ok=True)
        if tokenizer is None:
            tokenizer = self.tokenizer
        clean = self.visualizer.clean_text_for_display

        num_layers = len(attn_weights)
        if num_layers == 0:
            return

        num_heads = attn_weights[0].shape[1]
        batch_size = attn_weights[0].shape[0]
        # Pick up to max_samples random indices
        if batch_size > max_samples:
            sample_indices = random.sample(range(batch_size), max_samples)
        else:
            sample_indices = list(range(batch_size))
        for sample_idx in sample_indices:
            # Create a subdirectory for this sample
            sample_dir = os.path.join(dump_dir, f"sample_{sample_idx}")
            os.makedirs(sample_dir, exist_ok=True)
            input_mask = input_attention_mask[sample_idx].detach().cpu().bool().numpy()
            input_tokens = [
                t
                for t, m in zip(
                    tokenizer.convert_ids_to_tokens(
                        input_ids[sample_idx].detach().cpu()
                    ),
                    input_mask,
                )
                if m
            ]
            input_tokens = clean(input_tokens)
            # Decode concept string for this sample (without special tokens)
            concept_str = tokenizer.decode(
                concept_ids[sample_idx].detach().cpu(),
                skip_special_tokens=True,
            )
            # Truncate or wrap concept string for title
            max_title_len = 80
            if len(concept_str) > max_title_len:
                concept_str_disp = concept_str[:max_title_len] + "..."
            else:
                concept_str_disp = concept_str
            for l in range(num_layers):
                attn_layer = attn_weights[l][sample_idx]  # [num_heads, q_len, kv_len]
                # Get valid (unpadded) tokens for axes using indices
                input_mask = (
                    input_attention_mask[sample_idx].detach().cpu().bool().numpy()
                )
                input_indices = np.where(input_mask)[0]
                input_tokens = [
                    t
                    for i, t in enumerate(
                        tokenizer.convert_ids_to_tokens(
                            input_ids[sample_idx].detach().cpu()
                        )
                    )
                    if input_mask[i]
                ]

                # Replace each whitespace token in input_tokens with '[SPACE]'
                input_tokens = [
                    "[SPACE]" if t == "" or t == "\n" or t == " " else t
                    for t in input_tokens
                ]

                concept_mask = (
                    concept_attention_mask[sample_idx].detach().cpu().bool().numpy()
                )
                concept_indices = np.where(concept_mask)[0]
                concept_tokens = [
                    t
                    for i, t in enumerate(
                        tokenizer.convert_ids_to_tokens(
                            concept_ids[sample_idx].detach().cpu()
                        )
                    )
                    if concept_mask[i]
                ]
                concept_tokens = clean(concept_tokens)
                num_heads, q_len, kv_len = attn_layer.shape
                fig, axes = plt.subplots(
                    nrows=num_heads,
                    ncols=1,
                    figsize=(max(6, len(input_tokens) // 2), max(3, num_heads * 2)),
                    sharex=True,
                )
                if num_heads == 1:
                    axes = [axes]
                im = None
                for h in range(num_heads):
                    ax = axes[h]
                    # Use np.ix_ to select the correct submatrix
                    attn_2d = (
                        attn_layer[h][np.ix_(concept_indices, input_indices)]
                        .cpu()
                        .float()
                        .numpy()
                    )
                    im = ax.imshow(
                        attn_2d, aspect="auto", cmap="viridis", vmin=0, vmax=1
                    )
                    ax.set_ylabel(f"Head {h}")
                    ax.set_yticks(np.arange(len(concept_tokens)))
                    ax.set_yticklabels(concept_tokens, fontsize=6)
                    if h == num_heads - 1:
                        ax.set_xticks(np.arange(len(input_tokens)))
                        ax.set_xticklabels(input_tokens, rotation=90, fontsize=6)
                        ax.set_xlabel("Input Tokens")
                    else:
                        ax.set_xticks([])
                fig.suptitle(
                    f"{prefix} Layer {l} | Concept: {concept_str_disp}", fontsize=10
                )
                fig.tight_layout(rect=[0, 0, 1, 0.97])
                if im is not None:
                    fig.colorbar(im, ax=axes, fraction=0.02)
                fname = os.path.join(sample_dir, f"layer_{l}.png")
                plt.savefig(fname, dpi=100)
                plt.close(fig)
