import hashlib
import json
import logging
import os
import pickle
from contextlib import contextmanager
from os import PathLike
from pathlib import Path

import torch
import torch.distributed as dist

from hypersteer.utils.constants import EVAL_STATE_FILE


def load_state(dump_dir, mode, eval_run="evaluate"):
    """
    Load the state from a file if it exists.

    Args:
        dump_dir (str): The directory to load the state file from.

    Returns:
        dict: The loaded state dictionary, or None if no state file exists.
    """
    assert mode in ["latent", "steering"], "Invalid mode"
    state_path = os.path.join(dump_dir, eval_run, f"{mode}_{EVAL_STATE_FILE}")
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            return pickle.load(f)
    return None


def combine_scores_per_concept(concept_data):
    """Combine scores from concept and following evaluators for each method."""
    return concept_data["results"]["LMJudgeEvaluator"]


def process_jsonl_file(jsonl_lines):
    for data in jsonl_lines:
        data["results"]["LMJudgeEvaluator"] = combine_scores_per_concept(data)
    return jsonl_lines


@contextmanager
def set_default_device(device):
    """Context manager to temporarily set the default device for tensor creation."""
    original_device = torch.get_default_device()
    torch.set_default_device(device)
    try:
        yield
    finally:
        torch.set_default_device(original_device)


class DistributedAwareLogger(logging.Logger):
    """Enhanced distributed-aware logger that works with Hydra's colorlog."""

    def __init__(self, name):
        super().__init__(name)
        self.log_on_all_ranks = int(os.environ.get("LOG_ON_ALL_RANKS", 0)) == 1

    def _should_log(self):
        """Determine if this rank should log."""
        if not dist.is_initialized():
            return True

        rank = dist.get_rank()
        return rank == 0 or self.log_on_all_ranks

    def _add_rank_info(self, msg):
        """Add rank information to log message if distributed training is active."""
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()

            if world_size > 1:
                rank_info = f"[Rank {rank}/{world_size}] "
            else:
                rank_info = f"[Rank {rank}] "

            return f"{rank_info}{msg}"
        return msg

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False):
        """Override _log to implement distributed-aware logging."""
        if self._should_log():
            # Add rank information to the message
            msg = self._add_rank_info(msg)
            super()._log(level, msg, args, exc_info, extra, stack_info)

    def debug_all_ranks(self, msg, *args, **kwargs):
        """Force debug logging on all ranks regardless of settings."""
        if dist.is_initialized():
            original_setting = self.log_on_all_ranks
            self.log_on_all_ranks = True
            self.debug(msg, *args, **kwargs)
            self.log_on_all_ranks = original_setting
        else:
            self.debug(msg, *args, **kwargs)

    def info_all_ranks(self, msg, *args, **kwargs):
        """Force info logging on all ranks regardless of settings."""
        if dist.is_initialized():
            original_setting = self.log_on_all_ranks
            self.log_on_all_ranks = True
            self.info(msg, *args, **kwargs)
            self.log_on_all_ranks = original_setting
        else:
            self.info(msg, *args, **kwargs)

    def warning_all_ranks(self, msg, *args, **kwargs):
        """Force warning logging on all ranks regardless of settings."""
        if dist.is_initialized():
            original_setting = self.log_on_all_ranks
            self.log_on_all_ranks = True
            self.warning(msg, *args, **kwargs)
            self.log_on_all_ranks = original_setting
        else:
            self.warning(msg, *args, **kwargs)

    def error_all_ranks(self, msg, *args, **kwargs):
        """Force error logging on all ranks regardless of settings."""
        if dist.is_initialized():
            original_setting = self.log_on_all_ranks
            self.log_on_all_ranks = True
            self.error(msg, *args, **kwargs)
            self.log_on_all_ranks = original_setting
        else:
            self.error(msg, *args, **kwargs)


# Set the custom logger class as default
logging.setLoggerClass(DistributedAwareLogger)


def get_logger(name: str, level: str | None = None) -> DistributedAwareLogger:
    """
    Get a distributed-aware logger that works with Hydra's colorlog.

    Args:
        name: Logger name (typically __name__)
        level: Optional log level override

    Returns:
        DistributedAwareLogger instance
    """
    logger = logging.getLogger(name)

    if level:
        logger.setLevel(getattr(logging, level.upper()))

    return logger


def load_metadata(metadata_path):
    """
    Load metadata from a JSON lines file.
    """
    metadata = []
    with open(metadata_path) as f:
        for line in f:
            data = json.loads(line)
            metadata += [data]  # Return the metadata as is
    return metadata


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if is_distributed():
        return dist.get_rank()
    else:
        return 0


def get_world_size():
    if is_distributed():
        return dist.get_world_size()
    else:
        return 1


def barrier():
    if is_distributed():
        dist.barrier()


def destroy_process_group():
    if is_distributed():
        dist.destroy_process_group()


class PathJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path | PathLike):
            return str(obj)
        return super().default(obj)


def dump_json(obj, **kwargs):
    """Wrapper for json.dumps that handles Path objects."""
    return json.dumps(obj, cls=PathJSONEncoder, **kwargs)


def configure_tokenizer_model(
    model_instance, tokenizer, special_tokens: list[str] = None
):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    _need_resize = False
    if tokenizer.unk_token is None and tokenizer.pad_token is None:
        # raw llama3
        print("adding a special padding token...")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        _need_resize = True

    if special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        _need_resize = True

    if _need_resize:
        model_instance.resize_token_embeddings(len(tokenizer))


def get_and_set_device(local_rank):
    # Set the device for this process
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    elif torch.backends.mps.is_available():
        device = torch.device(f"mps:{local_rank}")
    else:
        device = torch.device("cpu")

    return device


def get_device_ordinal(device: str | torch.device) -> int | str:
    device_str = str(device)
    if device_str.startswith("cuda:"):
        return int(device_str.split(":")[1])
    return "cpu"


RELEVANT_STEERING_KEYS = [
    "input_length",
    "steering_output_length",
    "steering_num_of_examples",
    "steering_factors",
    "steering_datasets",
    "temperature",
    "lm_model",
    "seed",
]
RELEVANT_LATENT_KEYS = [
    "input_length",
    "latent_num_of_examples",
    "lm_model",
    "seed",
]


def get_cache_key(
    config, concept_ids, metadata, is_latent=False, extra_keys: dict = {}
):
    """Create a hash key based on relevant inputs that would affect the steering data."""
    # Extract only the relevant keys from the inference config
    relevant_keys = RELEVANT_LATENT_KEYS if is_latent else RELEVANT_STEERING_KEYS

    # Convert OmegaConf to dict and extract only relevant keys
    config_dict = config.model_dump()
    processed_config = {k: config_dict[k] for k in relevant_keys if k in config_dict}

    if extra_keys:
        processed_config.update(extra_keys)

    cache_key_dict = {
        "config": processed_config,
        "concept_ids": sorted(concept_ids),  # Sort for consistency
        "metadata_hash": hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode()
        ).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(cache_key_dict, sort_keys=True).encode()
    ).hexdigest()


def combine_all_results(all_results: list[dict]):
    combined_results = {}

    def combine_recursive(results_list):
        if not results_list:
            return {}
        if isinstance(results_list[0], list):
            # Base case - we've reached a list, concatenate them
            combined = []
            for result in results_list:
                combined.extend(result)
            return combined
        elif not isinstance(results_list[0], dict):
            # Base case - we've reached a scalar, return it
            return results_list
        # Recursively combine nested dictionaries
        combined_dict = {}
        for key in results_list[0].keys():
            values_for_key = [r[key] for r in results_list if key in r]
            combined_dict[key] = combine_recursive(values_for_key)
        return combined_dict

    # Combine all results recursively
    if all_results:
        combined_results = [
            combine_recursive([result[i] for result in all_results])
            for i in range(len(all_results[0]))
        ]
    return combined_results
