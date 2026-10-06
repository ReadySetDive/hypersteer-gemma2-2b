#################################
#
# Constants.
#
#################################


from enum import Enum


class EXAMPLE_TAG(Enum):
    CONTROL = 0
    EXPERIMENT = 1


OPENAI_RATE_LIMIT = 10
PRICING_DOLLAR_PER_1M_TOKEN = {
    "gpt-4o-mini-2024-07-18": {"input": 0.150, "output": 0.600},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gpt-4o": {"input": 5.00, "output": 15.00},
}

UNIT_1M = 1_000_000

CHAT_MODELS = {
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
}

BASE_MODELS = {"google/gemma-2-2b", "meta-llama/Llama-3.1-8B"}

EMPTY_CONCEPT = "EEEEE"

MAX_RETRIES = 5
RETRY_DELAY = 1  # in seconds
CONFIG_FILE = "config.json"
METADATA_FILE = "metadata.jsonl"
STEERING_EXCLUDE_MODELS = {
    "IntegratedGradients",
    "InputXGradients",
    "PromptDetection",
    "BoW",
}
LATENT_EXCLUDE_MODELS = {
    "PromptSteering",
    "PromptBaseline",
    "DiReFT",
    "LoReFT",
    "LoRA",
    "SFT",
}
HYPERNETWORK_MODELS = {"HyperSteer"}
SPARSE_SELECTIVE_MODELS = set()  # No selective models currently implemented
LSREFT_MODELS = set()  # No LsReFT models currently implemented

LATENT_PROMPT_PREFIX = "Generate a random sentence."
EVAL_STATE_FILE = "evaluate_state.pkl"
INFER_STATE_FILE = "inference_state.pkl"

CONFIG_FILE = "config.json"
STATE_FILE = "train_state.pkl"
METADATA_FILE = "metadata.jsonl"
