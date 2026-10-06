#!/bin/bash
# One-time setup on a RunPod GPU pod (PyTorch template).
# Usage (from the repo root on the pod):
#   echo "HF_TOKEN=hf_xxx" > .env
#   bash scripts/runpod_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f .env ] || ! grep -q '^HF_TOKEN=' .env; then
    echo "Missing .env with HF_TOKEN=... (create it first; never commit it)" >&2
    exit 1
fi

# Keep model downloads on the persistent /workspace volume
grep -q '^HF_HOME=' .env || echo "HF_HOME=/workspace/hf_cache" >> .env

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# --frozen: install exactly what uv.lock pins without re-resolving; re-resolving
# fails because the dev-only `ai_commit` git dependency no longer exists upstream
grep -q '^UV_FROZEN=' .env || echo "UV_FROZEN=1" >> .env
# RunPod templates set HF_HUB_ENABLE_HF_TRANSFER=1, but hf_transfer isn't in uv.lock
grep -q '^HF_HUB_ENABLE_HF_TRANSFER=' .env || echo "HF_HUB_ENABLE_HF_TRANSFER=0" >> .env
uv sync --frozen

set -a; source .env; set +a
uv run python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available - wrong pod/template?"
print("GPU:", torch.cuda.get_device_name(0),
      f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")

# Fails fast if the Gemma license hasn't been accepted for this token
from huggingface_hub import snapshot_download
for repo in ["google/gemma-2-2b-it", "google/gemma-2-2b"]:
    snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
    print("downloaded", repo)
EOF

echo "Setup OK. Next: bash scripts/runpod_train.sh smoke"
