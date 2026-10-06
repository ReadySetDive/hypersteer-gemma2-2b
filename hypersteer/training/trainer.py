import gc
import os
import random
import shutil
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from transformers import get_scheduler

import wandb
from hypersteer.utils.configs import ModelConfig, TrainingArgs, WandbConfig
from hypersteer.utils.helpers import get_logger, get_rank, get_world_size

logger = get_logger(__name__)

TRAINER_STATE = "trainer_state.pt"


class ResumableRandomSampler(Sampler):
    """Random permutation seeded by (seed, epoch) that can skip its first `start`
    indices, so a resumed run continues on the exact batch it stopped before."""

    def __init__(self, n: int, seed: int = 42):
        self.n, self.seed, self.epoch, self.start = n, seed, 0, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_start(self, start: int) -> None:
        self.start = start

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.n, generator=g)[self.start :].tolist())

    def __len__(self) -> int:
        return self.n - self.start


class TrainerMixin(ABC):
    """Abstract mixin that defines the interface for trainable models."""

    @abstractmethod
    def setup_model(self, **kwargs) -> None:
        """Setup the model for training."""
        pass

    @abstractmethod
    def setup_optimizer(self, **kwargs) -> torch.optim.Optimizer:
        """Setup and return the optimizer."""
        pass

    @abstractmethod
    def train_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Perform a single training step.

        Args:
            batch: The input batch

        Returns:
            Dictionary containing loss and any other metrics
        """
        pass

    @abstractmethod
    def val_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Perform a single validation step.

        Args:
            batch: The input batch

        Returns:
            Dictionary containing loss and any other metrics
        """
        pass

    def log_metrics(self, metrics, mode="train"):
        """Log metrics to wandb and logger."""
        # Prepare log dictionary
        log_dict = {}

        for key_tuple, value in metrics.items():
            section, keyname = key_tuple
            # Determine the log key format based on section
            # Special case for 'counters' section (e.g., step, lr)
            if section == "counters":
                log_key = f"{section}/{keyname}"
            # General case: {section}_{mode}/keyname
            else:
                log_key = f"{section}_{mode}/{keyname}"

            # Handle value extraction (for tensors, etc.)
            if isinstance(value, torch.Tensor):
                processed_value = value.detach().item()
            else:
                processed_value = float(value)  # Ensure it's a standard float

            log_dict[log_key] = processed_value

        # Log to wandb
        if wandb.run and (not dist.is_initialized() or dist.get_rank() == 0):
            wandb.log(log_dict)

        logger.info(log_dict)

    @abstractmethod
    def get_trainable_parameters(self):
        """Return the trainable parameters for gradient clipping."""
        pass

    def on_train_epoch_start(self, epoch: int) -> None:
        """Called at the start of each training epoch."""
        pass

    def on_train_epoch_end(self, epoch: int) -> None:
        """Called at the end of each training epoch."""
        pass

    def on_validation_start(self, global_step: int) -> None:
        """Called at the start of validation."""
        pass

    def on_validation_end(self, global_step: int) -> None:
        """Called at the end of validation."""
        pass


class Trainer:
    """Generic trainer that works with any model implementing TrainerMixin."""

    def __init__(
        self: "Trainer",
        model: TrainerMixin,
        training_args: TrainingArgs,
        model_config: ModelConfig,
        wandb_config: WandbConfig | None = None,
        device=None,
        seed=42,
    ) -> None:
        self.model = model
        self.training_args = training_args
        self.model_config = model_config
        self.wandb_config = wandb_config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.seed = seed

        # Training state
        self.optimizer = None
        self.lr_scheduler = None
        self.global_step = 0
        self.global_val_step = 0
        self.epoch = 0
        self.step_in_epoch = 0

        # Distributed training
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.is_distributed = self.world_size > 1

    def setup_training(
        self: "Trainer",
        train_dataloader: DataLoader,
        dev_dataloader: DataLoader | None = None,
    ) -> tuple[int, int]:
        """Setup training components."""
        # Setup model
        self.model.setup_model()

        # Setup optimizer
        self.optimizer = self.model.setup_optimizer()

        # Determine training configuration
        use_step_limit = self.training_args.n_steps > 0
        if use_step_limit:
            num_training_steps = self.training_args.n_steps
            effective_epochs = float("inf")
            logger.info(f"Training with step limit: {num_training_steps} steps")
        else:
            num_training_steps = self.training_args.n_epochs * (
                len(train_dataloader) // self.training_args.gradient_accumulation_steps
            )
            effective_epochs = self.training_args.n_epochs
            logger.info(f"Training with epoch limit: {effective_epochs} epochs")

        # Setup learning rate scheduler
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=self.training_args.warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Log training configuration
        steps_per_epoch = (
            len(train_dataloader) // self.training_args.gradient_accumulation_steps
        )
        logger.info("Training configuration:")
        logger.info(f"  - Batch size: {self.training_args.batch_size}")
        logger.info(
            f"  - Gradient accumulation steps: {self.training_args.gradient_accumulation_steps}"
        )
        logger.info(f"  - Steps per epoch: {steps_per_epoch}")
        logger.info(f"  - Total training steps: {num_training_steps}")
        if use_step_limit:
            estimated_epochs = num_training_steps / steps_per_epoch
            logger.info(f"  - Estimated epochs to complete: {estimated_epochs:.2f}")
        else:
            logger.info(f"  - Training epochs: {effective_epochs}")

        # Setup wandb watching
        if wandb.run and self.wandb_config and self.wandb_config.watch_grads:
            # Let the model decide what to watch
            if hasattr(self.model, "get_watchable_modules"):
                modules = self.model.get_watchable_modules()
                wandb.watch(modules, log_freq=self.wandb_config.watch_grads_freq)

        return num_training_steps, effective_epochs

    def train(
        self,
        train_dataloader: DataLoader,
        dev_dataloader: DataLoader | None = None,
        train_sampler: DistributedSampler | None = None,
    ):
        """Main training loop."""
        num_training_steps, effective_epochs = self.setup_training(
            train_dataloader, dev_dataloader
        )
        use_step_limit = self.training_args.n_steps > 0
        steps_per_epoch = len(train_dataloader)

        # Training state
        accum_counter = 0
        epoch, start_step = 0, 0
        if self.training_args.resume_from:
            epoch, start_step = self.load_resume_state(
                self.training_args.resume_from, steps_per_epoch
            )
            if start_step >= steps_per_epoch:
                epoch, start_step = epoch + 1, 0

        # Training loop
        while epoch < effective_epochs:
            self.model.on_train_epoch_start(epoch)

            step = start_step
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
                if hasattr(train_sampler, "set_start"):
                    train_sampler.set_start(step * self.training_args.batch_size)
            elif step:
                raise RuntimeError("Mid-epoch resume needs a ResumableRandomSampler")
            start_step = 0
            train_iter = iter(train_dataloader)

            while step < steps_per_epoch:
                # Check if we've reached the step limit
                if use_step_limit and self.global_step >= num_training_steps:
                    logger.info(
                        f"Reached step limit of {num_training_steps} steps. Terminating training."
                    )
                    return

                # Get next batch
                batch = next(train_iter)

                # Training step
                if accum_counter == 0:
                    self.optimizer.zero_grad()

                # Forward pass
                step_outputs = self.model.train_step(batch, self.global_step)
                loss = step_outputs[("loss", "main")]

                # Backward pass
                scaled_loss = loss / self.training_args.gradient_accumulation_steps
                scaled_loss.backward()
                accum_counter += 1

                # Optimizer step
                if accum_counter == self.training_args.gradient_accumulation_steps:
                    step_outputs = self.model.post_backward(
                        step_outputs,
                        self.lr_scheduler,
                        self.optimizer,
                        self.global_step,
                    )
                    accum_counter = 0

                    self.optimizer.step()
                    self.lr_scheduler.step()

                    # Log metrics
                    step_outputs[("counters", "global_step")] = self.global_step
                    self.model.log_metrics(step_outputs, mode="train")

                step += 1
                self.global_step += 1
                self.epoch, self.step_in_epoch = epoch, step

                # Validation
                if (
                    dev_dataloader is not None
                    and hasattr(self.training_args, "val_interval")
                    and self.training_args.val_interval > 0
                    and self.global_step % self.training_args.val_interval == 0
                ):
                    self.validate(dev_dataloader)

                # Periodic checkpoint
                ckpt_every = self.training_args.checkpoint_per_step
                if ckpt_every and self.global_step % ckpt_every == 0:
                    self.save_checkpoint()

                # Cleanup
                del batch, step_outputs, loss, scaled_loss
                torch.cuda.empty_cache()

            self.model.on_train_epoch_end(epoch)
            epoch += 1

    def save_checkpoint(self):
        """Save hypernet weights to <dump_dir>/checkpoints/step_N without leaving the
        GPU, keep the 2 newest locally, and upload to $HF_CKPT_REPO in the background."""
        from safetensors.torch import save_file

        dump_dir = getattr(self.model, "dump_dir", None)
        if dump_dir is None:
            return
        ckpt_root = Path(dump_dir) / "checkpoints"
        path = ckpt_root / f"step_{self.global_step}"
        path.mkdir(parents=True, exist_ok=True)
        state = {
            k: v.detach().cpu().contiguous()
            for k, v in self.model.concept_embedding.state_dict().items()
        }
        save_file(state, str(path / f"{self.model_config.model_name}_weight.safetensors"))
        del state
        self.save_resume_state(path)
        logger.info(f"Saved checkpoint to {path}")

        for old in sorted(
            ckpt_root.glob("step_*"), key=lambda p: int(p.name.split("_")[1])
        )[:-2]:
            shutil.rmtree(old, ignore_errors=True)

        repo_id = os.environ.get("HF_CKPT_REPO")
        if repo_id:
            # path_in_repo: <run>/checkpoints/step_N
            path_in_repo = str(path.relative_to(Path(dump_dir).parent.parent)).replace(os.sep, "/")

            def _upload():
                try:
                    from huggingface_hub import HfApi

                    api = HfApi()
                    # Weights only: trainer_state.pt (optimizer, ~2x weights) stays local
                    api.upload_folder(
                        repo_id=repo_id, folder_path=str(path), path_in_repo=path_in_repo,
                        ignore_patterns=[TRAINER_STATE],
                    )
                    # The run's config.yaml (needed to load any checkpoint) otherwise only
                    # reaches HF with the end-of-run upload
                    config = Path(dump_dir).parent / "config.yaml"
                    if config.exists():
                        api.upload_file(
                            repo_id=repo_id, path_or_fileobj=str(config),
                            path_in_repo=f"{path_in_repo.split('/')[0]}/config.yaml",
                        )
                    logger.info(f"Uploaded checkpoint to {repo_id}/{path_in_repo}")
                except Exception as e:
                    logger.warning(f"Checkpoint upload failed (training continues): {e}")

            # Non-daemon: interpreter exit waits for an in-flight upload instead of
            # killing it mid-request ("FATAL: exception not rethrown" abort)
            threading.Thread(target=_upload, daemon=False).start()

    def save_resume_state(self, path: Path) -> None:
        """Everything besides the weights needed to continue this run exactly."""
        state = {
            "global_step": self.global_step,
            "global_val_step": self.global_val_step,
            "epoch": self.epoch,
            "step_in_epoch": self.step_in_epoch,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        tmp = path / (TRAINER_STATE + ".tmp")
        torch.save(state, tmp)
        os.replace(tmp, path / TRAINER_STATE)  # never leave a half-written state

    def load_resume_state(self, path, steps_per_epoch: int) -> tuple[int, int]:
        """Load a checkpoint dir written by save_checkpoint. Returns (epoch, step in
        epoch) to continue from. Weights-only dirs (e.g. downloaded from HF) restore the
        weights and step count but start a fresh optimizer."""
        from safetensors.torch import load_file

        path = Path(path)
        weights = path / f"{self.model_config.model_name}_weight.safetensors"
        emb = self.model.concept_embedding
        emb.load_state_dict(load_file(str(weights), device=str(next(emb.parameters()).device)))
        state_file = path / TRAINER_STATE
        if not state_file.exists():
            step = int(path.name.split("_")[-1])
            logger.warning(
                f"{state_file} missing: resuming weights from {path} at step {step} "
                "with a FRESH optimizer state (Adam moments reset)"
            )
            self.global_step = step
            for _ in range(step):  # fast-forward the LR schedule
                self.lr_scheduler.step()
            return divmod(step, steps_per_epoch)
        state = torch.load(state_file, map_location="cpu", weights_only=False)
        self.optimizer.load_state_dict(state["optimizer"])
        self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        self.global_step = state["global_step"]
        self.global_val_step = state["global_val_step"]
        rng = state["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        logger.info(
            f"Resumed from {path}: global_step={self.global_step}, "
            f"epoch={state['epoch']}, step_in_epoch={state['step_in_epoch']}"
        )
        return state["epoch"], state["step_in_epoch"]

    @torch.no_grad()
    def validate(self, dev_dataloader: DataLoader):
        """Run validation."""
        logger.info(f"Running validation at step {self.global_step}")

        self.model.on_validation_start()

        all_metrics = []

        for batch in dev_dataloader:
            step_outputs = self.model.val_step(batch, self.global_step)
            all_metrics.append(step_outputs)

        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(all_metrics)
        aggregated_metrics[("counters", "val_step")] = self.global_step
        aggregated_metrics[("counters", "global_val_step")] = self.global_val_step

        # Log validation metrics
        self.model.log_metrics(aggregated_metrics, mode="val")

        self.global_val_step += 1

        # Set back to train mode
        if hasattr(self.model, "train"):
            self.model.train()

        self.model.on_validation_end()

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

    def _aggregate_metrics(self, metrics_list):
        """Aggregate metrics from multiple validation steps."""
        if not metrics_list:
            return {}

        aggregated = {}
        for _, key in metrics_list[0].keys():
            if key in ["loss", "grad_norm"] or key.endswith("_loss"):
                # Average numerical metrics
                values = [
                    m[key] for m in metrics_list if key in m and m[key] is not None
                ]
                if values:
                    aggregated[key] = sum(values) / len(values)
            elif key.endswith("_count"):
                # Sum count metrics
                values = [
                    m[key] for m in metrics_list if key in m and m[key] is not None
                ]
                if values:
                    aggregated[key] = sum(values)

        return aggregated
