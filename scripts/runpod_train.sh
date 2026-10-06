#!/bin/bash
# Train HyperSteer (text-prompt, cross-attention hypernet) on Gemma-2-2B layer 20.
# Usage (from the repo root on the pod):
#   bash scripts/runpod_train.sh smoke   # ~5 concepts, 50 steps: checks the pipeline + timing
#   tmux new -s train                    # then inside tmux:
#   bash scripts/runpod_train.sh full    # all 500 concepts, 3 epochs (overnight)
#   nohup bash scripts/runpod_train.sh large > logs/large.out 2>&1 &
#                                        # concept16k subset: MAX_CONCEPTS (8000), EPOCHS (1), CKPT_EVERY (3000)
#   RESUME=1 nohup bash scripts/runpod_train.sh large > logs/large.out 2>&1 &
#                                        # continue the newest run from its latest checkpoint
#                                        # (same MODE and env as the original launch)
# Weights land in assets/checkpoints/train_<timestamp>/train/
#
# Auto-stop: `full` stops this pod when training ends (success OR crash) so the GPU
# stops billing; /workspace (and the checkpoint) survives a stop. `smoke` only verifies
# that auto-stop will work. Override with AUTO_STOP=0 or AUTO_STOP=1.
#
# HF upload: `full` uploads the finished run folder to a private Hugging Face repo
# (<your-hf-user>/$HF_REPO_NAME) before stopping. Needs a Write-capable HF_TOKEN.
# `smoke` only verifies write access. Override with HF_UPLOAD=0 or HF_UPLOAD=1.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
set -a; source .env; set +a
# RunPod injects a pod-scoped RUNPOD_API_KEY that runpodctl prefers over the key from
# `runpodctl config --apiKey`, and it lacks permission to get/stop the pod
unset RUNPOD_API_KEY
# SSH sessions don't inherit the container env; the web terminal does
RUNPOD_POD_ID=${RUNPOD_POD_ID:-$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^RUNPOD_POD_ID=//p')}

MODE=${1:-smoke}
shift || true

case "$MODE" in
    smoke) OVERRIDES=(dataset.train.max_concepts=5 train.n_steps=50); AUTO_STOP=${AUTO_STOP:-0}; HF_UPLOAD=${HF_UPLOAD:-0} ;;
    # Default validation runs the full 10% dev split every 100 steps (~9h of overhead at 500 concepts)
    # batch_size=16: the authors' GH200 setting (config/experiment/hypersteer_gh200.yaml), same lr;
    # ~6.1k steps total. Checkpoints (uploaded to HF if HF_UPLOAD=1) every 1500 steps.
    full)  OVERRIDES=(dataset.train.max_concepts=500 dataset.train.dev_size=0.02 train.batch_size=16 train.val_interval=500 train.checkpoint_per_step=1500); AUTO_STOP=${AUTO_STOP:-1}; HF_UPLOAD=${HF_UPLOAD:-1} ;;
    # First MAX_CONCEPTS concept ids of concept16k (includes the 500 above); higher ids stay
    # unseen = a held-out set in the training style. Fixed 1000-example dev set (2% would be
    # ~11k examples, ~5 min per validation). ~36k steps/epoch at 8000 concepts, batch 16.
    large) OVERRIDES=(dataset.train.hf_dataset_name=pyvene/axbench-concept16k dataset.train.max_concepts=${MAX_CONCEPTS:-8000} dataset.train.dev_size=1000 train.batch_size=16 train.n_epochs=${EPOCHS:-1} train.val_interval=2000 train.checkpoint_per_step=${CKPT_EVERY:-3000}); AUTO_STOP=${AUTO_STOP:-1}; HF_UPLOAD=${HF_UPLOAD:-1} ;;
    *) echo "usage: $0 [smoke|full|large] [extra hydra overrides...]" >&2; exit 1 ;;
esac
HF_REPO_NAME=${HF_REPO_NAME:-hypersteer-gemma2-2b-l20}

if [ -n "${RESUME_FROM:-}" ]; then
    # Explicit checkpoint dir <run>/train/checkpoints/step_N, e.g. downloaded from HF onto a
    # fresh pod (weights only -> resumes with a fresh optimizer, logged as a warning)
    echo "Resuming from $RESUME_FROM"
    OVERRIDES+=("train.resume_from=$RESUME_FROM")
elif [ "${RESUME:-0}" = 1 ]; then
    # Newest checkpoint dir that has optimizer state (written by Trainer.save_checkpoint)
    RESUME_FROM=$(ls -dt assets/checkpoints/train_*/train/checkpoints/step_*/ 2>/dev/null         | while read -r d; do [ -f "$d/trainer_state.pt" ] && { echo "${d%/}"; break; }; done || true)
    if [ -z "$RESUME_FROM" ]; then
        echo "RESUME=1 but no checkpoint with trainer_state.pt under assets/checkpoints" >&2
        exit 1
    fi
    echo "Resuming from $RESUME_FROM"
    OVERRIDES+=("train.resume_from=$RESUME_FROM")
fi

# Preflight: confirm the token can write by creating (or finding) the private repo
if HF_REPO_ID=$(uv run python - "$HF_REPO_NAME" <<'EOF'
import sys
from huggingface_hub import HfApi
api = HfApi()
repo_id = f"{api.whoami()['name']}/{sys.argv[1]}"
api.create_repo(repo_id, private=True, exist_ok=True)
print(f"HF upload preflight OK ({repo_id})", file=sys.stderr)
print(repo_id)
EOF
); then
    HF_UPLOAD_OK=1
    # Read by Trainer.save_checkpoint for mid-run uploads
    [ "$HF_UPLOAD" = 1 ] && export HF_CKPT_REPO="$HF_REPO_ID"
else
    echo "WARNING: can't create HF repo - token is probably Read-only; HF upload unavailable" >&2
    HF_UPLOAD_OK=0
fi
if [ "$HF_UPLOAD" = 1 ] && [ "$HF_UPLOAD_OK" = 0 ]; then
    echo "Refusing to start: checkpoint couldn't be uploaded. Fix token, or HF_UPLOAD=0." >&2
    exit 1
fi

# Preflight: confirm the pod can stop itself before spending hours training
if [ -z "${RUNPOD_POD_ID:-}" ] || ! command -v runpodctl >/dev/null 2>&1; then
    echo "WARNING: runpodctl or RUNPOD_POD_ID missing - auto-stop unavailable" >&2
    AUTO_STOP_OK=0
elif runpodctl get pod "$RUNPOD_POD_ID" >/dev/null 2>&1; then
    echo "Auto-stop preflight OK (pod $RUNPOD_POD_ID)"
    AUTO_STOP_OK=1
else
    echo "WARNING: 'runpodctl get pod' failed (API key?) - auto-stop unavailable" >&2
    AUTO_STOP_OK=0
fi
if [ "$AUTO_STOP" = 1 ] && [ "$AUTO_STOP_OK" = 0 ]; then
    echo "Refusing to start a long run that can't stop itself. Fix above, or AUTO_STOP=0." >&2
    exit 1
fi

stop_pod() {
    echo "Training exited ($1). Stopping pod in 60s - Ctrl+C to cancel."
    sleep 60
    runpodctl stop pod "$RUNPOD_POD_ID"
}

mkdir -p logs
LOG="logs/train_${MODE}_$(date +%Y%m%d_%H%M%S).log"

# run_eval_suite_at_end=false: the eval suite calls OpenAI (LLM judge); run it separately later.
# Pass wandb.log=true as an extra override if WANDB_API_KEY is in .env.
set +e
uv run python -m hypersteer.scripts.train \
    experiment=hypersteer \
    wandb.log=false \
    train.run_eval_suite_at_end=false \
    "${OVERRIDES[@]}" "$@" 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

echo "Exit status: $STATUS"
echo "Log: $LOG"
if [ -n "${RESUME_FROM:-}" ]; then
    RUN_DIR=${RESUME_FROM%/train/checkpoints/*}
else
    RUN_DIR=$(ls -dt assets/checkpoints/train_* 2>/dev/null | head -1 || true)
fi
echo "Run dir: ${RUN_DIR:-<none>}"

if [ "$HF_UPLOAD" = 1 ] && [ "$STATUS" = 0 ] && [ -n "$RUN_DIR" ]; then
    # Retry: a failed upload before auto-stop would strand the weights on a stopped pod
    for attempt in 1 2 3; do
        if uv run python - "$HF_REPO_NAME" "$RUN_DIR" "$LOG" <<'EOF'
import os, sys
from huggingface_hub import HfApi
api = HfApi()
name, run_dir, log = sys.argv[1:]
repo_id = f"{api.whoami()['name']}/{name}"
run = os.path.basename(run_dir)
# trainer_state.pt = optimizer state for resuming (~2x weights); keep it local
api.upload_folder(repo_id=repo_id, folder_path=run_dir, path_in_repo=run,
                  ignore_patterns=["**/trainer_state.pt*", "trainer_state.pt*"])
api.upload_file(repo_id=repo_id, path_or_fileobj=log, path_in_repo=f"{run}/{os.path.basename(log)}")
print(f"Uploaded to https://huggingface.co/{repo_id}/tree/main/{run}")
EOF
        then break; fi
        echo "Upload attempt $attempt failed" >&2
        if [ "$attempt" = 3 ]; then
            echo "Upload failed 3x - NOT auto-stopping so the checkpoint stays reachable." >&2
            AUTO_STOP=0
        else
            sleep 30
        fi
    done
fi

if [ "$AUTO_STOP" = 1 ]; then
    stop_pod "status $STATUS"
fi
exit "$STATUS"
