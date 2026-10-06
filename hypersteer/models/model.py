import os

import einops
import pandas as pd
import torch
from pyvene import (
    IntervenableModel,
)
from torch.utils.data import DataLoader

from hypersteer.data.utils import *  # noqa: F403
from hypersteer.utils.configs import ModelConfig, TrainingArgs, WandbConfig
from hypersteer.utils.helpers import get_logger

from .base import BaseModel

# Initialize the logger
logger = get_logger(__name__)


class Model(BaseModel):
    def __init__(
        self,
        model,
        tokenizer,
        training_args: TrainingArgs | None = None,
        model_config: ModelConfig | None = None,
        wandb_config: WandbConfig | None = None,
        **kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        # abstracting layer
        self.layer = getattr(model_config, "layer", None)
        self.training_args: TrainingArgs = training_args
        self.model_config: ModelConfig = model_config
        self.wandb_config: WandbConfig = wandb_config
        self.max_activations = {}
        # Set default device to GPU if available, otherwise CPU
        self.device = kwargs.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.seed = kwargs.get("seed", 42)
        self.steering_layers = getattr(model_config, "steering_layers", None)
        self.num_of_layers = len(self.steering_layers) if self.steering_layers else 1
        self.dump_dir = kwargs.get("dump_dir", None)

    def make_model(self, **kwargs):
        pass

    def train(self):
        """
        Set the model and any torch modules in this class to train mode.
        """
        if hasattr(self.model, "train"):
            self.model.train()
        # If there are other torch.nn.Module attributes, set them to train mode as well
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.nn.Module) and attr is not self.model:
                attr.train()

    def eval(self):
        """
        Set the model and any torch modules in this class to eval mode.
        """
        if hasattr(self.model, "eval"):
            self.model.eval()
        # If there are other torch.nn.Module attributes, set them to eval mode as well
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.nn.Module) and attr is not self.model:
                attr.eval()

    def make_dataloader(self, examples, **kwargs):
        data_module = make_data_module(self.tokenizer, examples, **kwargs)  # noqa: F405
        g = torch.Generator()
        g.manual_seed(self.seed)
        train_dataloader = DataLoader(
            data_module["train_dataset"],
            shuffle=True,  # we shuffle for examples.
            batch_size=self.training_args.batch_size,
            collate_fn=data_module["data_collator"],
            generator=g,
        )
        return train_dataloader

    def save(self, dump_dir, **kwargs):
        model_name = kwargs.get("model_name", self.__str__())
        weight_file = dump_dir / f"{model_name}_weight.pt"
        weight = self.ax.proj.weight.data.cpu()
        if weight_file.exists():
            weight = torch.cat([torch.load(weight_file), weight], dim=0)
        torch.save(weight, weight_file)

        bias_file = dump_dir / f"{model_name}_bias.pt"
        if self.ax.proj.bias is not None and bias_file.exists():
            bias = self.ax.proj.bias.data.cpu()
            bias = torch.cat([torch.load(bias_file), bias], dim=0)
            torch.save(bias, bias_file)

    def load(self, dump_dir=None, **kwargs):
        model_name = kwargs.get("model_name", self.__str__())
        weight = torch.load(f"{dump_dir}/{model_name}_weight.pt")
        if os.path.exists(f"{dump_dir}/{model_name}_bias.pt"):
            bias = torch.load(f"{dump_dir}/{model_name}_bias.pt")
        else:
            bias = None
        self.make_model(low_rank_dimension=weight.shape[0], **kwargs)
        self.ax.proj.weight.data = weight.to(self.device)
        if bias is not None and self.ax.proj.bias is not None:
            self.ax.proj.bias.data = bias.to(self.device)

    @torch.no_grad()
    def predict_steer(self, examples, **kwargs):
        """Use the generic inference function."""
        self.ax.eval()

        # Import the generic inference function
        try:
            from inference import run_steering_inference
        except ImportError:
            # Fallback for when running from different directory
            import importlib.util
            import os

            # Get the path to inference.py in the root directory
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            inference_path = os.path.join(root_dir, "inference.py")

            # Load the module dynamically
            spec = importlib.util.spec_from_file_location("inference", inference_path)
            inference_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(inference_module)

            run_steering_inference = inference_module.run_steering_inference

        return run_steering_inference(self, examples, **kwargs)

    def predict_step(self, batch_examples, batch_idx, **kwargs):
        """Model-specific prediction step for basic steering models."""
        # depending on the model, we use different concept id columns
        concept_id_col = (
            "sae_id"
            if "sae" in self.__str__().lower()
            and not kwargs.get("disable_neuronpedia_max_act", False)
            else "concept_id"
        )
        use_synergy = kwargs.get("use_synergy", False)
        eval_output_length = kwargs.get("eval_output_length", 128)
        temperature = kwargs.get("temperature", 1.0)

        if use_synergy:
            input_strings = batch_examples["steered_input"].tolist()
        else:
            input_strings = batch_examples["input"].tolist()

        mag = torch.tensor(batch_examples["factor"].tolist()).to(self.device)
        idx = torch.tensor(batch_examples["concept_id"].tolist()).to(self.device)
        max_acts = torch.tensor(
            [
                self.max_activations.get(id, 1.0)
                for id in batch_examples[concept_id_col].tolist()
            ]
        ).to(self.device)

        # tokenize input_strings
        inputs = self.tokenizer(
            input_strings, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        _, generations = self.ax_model.generate(
            inputs,
            unit_locations=None,
            intervene_on_prompt=True,
            subspaces=[
                {
                    "idx": idx,
                    "mag": mag,
                    "max_act": max_acts,
                    "prefix_length": kwargs["prefix_length"],
                }
            ]
            * self.num_of_layers,
            max_new_tokens=eval_output_length,
            do_sample=True,
            temperature=temperature,
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

        # Import here to avoid circular imports
        from hypersteer.training.trainer import calculate_perplexity

        perplexities = calculate_perplexity(
            self.model, self.tokenizer, unpruned_generated_texts, self.device
        )

        strengths = (mag * max_acts).cpu().float().tolist()

        return {
            "generations": generated_texts,
            "perplexities": perplexities,
            "strengths": strengths,
            "steering_vectors": [],  # Basic model doesn't have steering vectors
        }

    def get_logits(self, concept_id, k=10):
        top_logits, neg_logits = [None], [None]
        if concept_id is not None:
            W_U = self.model.lm_head.weight.T
            W_U = (
                W_U
                * (
                    self.model.model.norm.weight
                    + torch.ones_like(self.model.model.norm.weight)
                )[:, None]
            )
            W_U -= einops.reduce(W_U, "d_model d_vocab -> 1 d_vocab", "mean")

            vocab_logits = self.ax.proj.weight.data[concept_id] @ W_U.to(
                self.ax.proj.weight.data.dtype
            )
            top_values, top_indices = vocab_logits.topk(k=k, sorted=True)
            top_tokens = self.tokenizer.batch_decode(top_indices.unsqueeze(dim=-1))
            top_logits = [list(zip(top_tokens, top_values.tolist()))]

            neg_values, neg_indices = vocab_logits.topk(k=k, largest=False, sorted=True)
            neg_tokens = self.tokenizer.batch_decode(neg_indices.unsqueeze(dim=-1))
            neg_logits = [list(zip(neg_tokens, neg_values.tolist()))]
        return top_logits, neg_logits

    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        max_activations = {}  # sae_id to max_activation
        # Loop over saved latent files in dump_dir.
        for file in os.listdir(dump_dir):
            if file.startswith("latent_") and file.endswith(".parquet"):
                latent_path = os.path.join(dump_dir, file)
                latent = pd.read_parquet(latent_path)
                # loop through unique sorted concept_id
                for concept_id in sorted(latent["concept_id"].unique()):
                    concept_latent = latent[latent["concept_id"] == concept_id]
                    max_act = concept_latent[f"{self.__str__()}_max_act"].max()
                    max_activations[concept_id] = max_act if max_act > 0 else 50
        self.max_activations = max_activations
        return max_activations

    def to(self, device):
        """Move model to specified device"""
        self.device = device
        if hasattr(self, "ax"):
            self.ax = self.ax.to(device)
            if hasattr(self, "ax_model"):
                if isinstance(self.ax_model, IntervenableModel):
                    self.ax_model.set_device(device)
                else:
                    self.ax_model = self.ax_model.to(device)
        return self

    def backward_step(self, step_outputs, optimizer, lr_scheduler, training_args):
        # Default: just step optimizer and lr_scheduler, optionally grad norm if implemented
        if (
            hasattr(self, "get_trainable_parameters")
            and training_args.max_grad_norm > 0
        ):
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.get_trainable_parameters(),
                training_args.max_grad_norm,
            )
            step_outputs["grad_norm"] = grad_norm
        optimizer.step()
        lr_scheduler.step()
        return step_outputs
