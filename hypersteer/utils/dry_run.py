from functools import partial

import tiktoken
from openai import AsyncOpenAI

from hypersteer.utils.helpers import get_logger

# Initialize the logger
logger = get_logger(__name__)

# Simple global tracker
call_stats = {
    "calls": 0,
    "models": {},  # model -> {calls: int, input_tokens: int, output_tokens: int}
}


def count_tokens(text: str, model: str) -> int:
    try:
        encoder = tiktoken.encoding_for_model(model)
        return len(encoder.encode(str(text)))
    except Exception:
        return len(str(text)) // 4


async def dry_run_chat_completions(self, *args, **kwargs):
    """Lightweight tracking wrapper for chat completions"""
    model = kwargs.get("model", "gpt-3.5-turbo")
    messages = kwargs.get("messages", [])
    max_tokens = kwargs.get("max_tokens", 100)

    input_tokens = sum(count_tokens(m.get("content", ""), model) for m in messages)

    # Track stats
    call_stats["calls"] += 1
    if model not in call_stats["models"]:
        call_stats["models"][model] = {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    call_stats["models"][model]["calls"] += 1
    call_stats["models"][model]["input_tokens"] += input_tokens
    call_stats["models"][model]["output_tokens"] += max_tokens

    # Log first few calls
    if call_stats["calls"] <= 3:
        logger.warning(
            f"DryRun: API call to {model} (input tokens: {input_tokens}, max output: {max_tokens})"
        )

    mock_response = {
        "choices": [
            {
                "message": {"content": "DRY RUN RESPONSE", "role": "assistant"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "created": 1234567890,
        "model": model,
        "usage": {
            "completion_tokens": max_tokens,
            "prompt_tokens": input_tokens,
            "total_tokens": input_tokens + max_tokens,
        },
    }

    return type("MockResponse", (), {"to_dict": lambda: mock_response, **mock_response})


def patch_client(client: AsyncOpenAI) -> AsyncOpenAI:
    """Patch a client instance and return it"""
    # Create bound method for the instance
    bound_dry_run = partial(dry_run_chat_completions, client.chat.completions)

    # Patch the instance
    client.chat.completions.create = bound_dry_run

    logger.warning(f"Patched OpenAI client instance {id(client)}")
    return client
