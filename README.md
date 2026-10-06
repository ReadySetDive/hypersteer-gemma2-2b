# HyperSteer on Gemma-2-2B: training + interactive demos

A reproduction of **HyperSteer** ([Sun et al., 2025, arXiv:2506.03292](https://arxiv.org/abs/2506.03292)) for CSCI 544 (Team 28). A hypernetwork reads a plain-English *steering prompt* (e.g. "references to organic compounds") and outputs a steering vector that is added to Gemma-2-2B-it's layer-20 residual stream, with no prompt tokens spent.

This repo contains:
- **Training** on a single RunPod A100, with Hugging Face checkpoint uploads, auto-stop, exact resume, and a laptop-side crash watchdog.
- **Two Gradio demos:** a side-by-side comparison app, and a *live* app that bends a generation mid-stream.
- **Trained weights** in the Hugging Face repo [`RSD002/hypersteer-gemma2-2b-l20`](https://huggingface.co/RSD002/hypersteer-gemma2-2b-l20) (private; ask for access). The demos download them automatically.

Built on the HyperSteer research code released in [stanfordnlp/axbench](https://github.com/stanfordnlp/axbench) (Apache-2.0), via the restructured [HyperStuff/hypersteer](https://github.com/HyperStuff/hypersteer) (the `hypersteer/` package and `config/`), with changes for single-GPU training, resumability and fast inference. See [Credits](#credits) and [`NOTICE`](NOTICE).

## Results

| Run (HF repo folder) | Training data | Held-out concepts |
|---|---|---|
| `train_20261004_230823162433` | 500 concepts (AxBench Concept500), 3 epochs | **no** visible steering |
| `train_20261005_032732706478` (**final**) | first 8,000 of AxBench Concept16k, 3 epochs (107,814 steps) | **steers visibly** (4 of 6 tested at ×1.5) |

Scaling the number of training concepts is what makes HyperSteer generalize to steering prompts it never saw, as the paper's Figure 2 claims. Sample table: [`results/results_table_8k_final.md`](results/results_table_8k_final.md). Strength ~1.0–1.5 is fluent, and ~2.0 over-steers into repetition.

## Demo (Windows + NVIDIA GPU, 8 GB is enough)

```powershell
# one time: CUDA env in demo\.venv-win (needs uv: https://docs.astral.sh/uv/)
powershell -ExecutionPolicy Bypass -File demo\setup_windows.ps1
demo\.venv-win\Scripts\huggingface-cli login        # account with access to the HF repo

.\demo\run_demo.ps1            # side-by-side app   -> http://127.0.0.1:7860
.\demo\run_live.ps1            # live steering app  -> http://127.0.0.1:7861
.\demo\run_demo.ps1 -Share     # + public https://*.gradio.live link (asks for a login)
```
The first launch downloads the final weights (~5 GB) to `assets\checkpoints\hf\`. On an 8 GB GPU, run one demo at a time. Details: [`demo/README.md`](demo/README.md).

**Linux / cloud GPU:**
```bash
uv sync --frozen
uv run --with "gradio>=4.44,<6" --with "huggingface-hub<1.0" python demo/fetch_and_run.py --fetch-only
uv run --with "gradio>=4.44,<6" --with "huggingface-hub<1.0" python demo/app.py --backend hypersteer \
    --run-dir assets/checkpoints/hf/train_20261005_032732706478 --quant none --hyper-quant none --share
```

## Training (RunPod, ~$10–30 per run)

1. Create a pod with an **A100 80GB**, the PyTorch template, and a **150 GB volume**. Then on the pod:
   ```bash
   cd /workspace && git clone <this repo> hypersteer && cd hypersteer
   read -s -p "HF write token: " T && echo "HF_TOKEN=$T" > .env && unset T   # Gemma license accepted
   runpodctl config --apiKey <your RunPod key>    # lets the pod stop itself when done
   bash scripts/runpod_setup.sh                   # uv env + Gemma download (~15 min)
   ```
2. Smoke test, then the real run:
   ```bash
   bash scripts/runpod_train.sh smoke                       # 5 concepts, 50 steps
   nohup bash scripts/runpod_train.sh large > logs/large.out 2>&1 &
   ```
   - `large` = the first `MAX_CONCEPTS` (8000) of Concept16k, `EPOCHS` (1), checkpoint every `CKPT_EVERY` (3000) steps. The final run used `EPOCHS=3 CKPT_EVERY=9000`.
   - Also: `full` = the 500-concept setup. Batch 16 is the fastest per example on an A100 (bigger batches were slower).
   - Checkpoints and the final weights upload to `<your-hf-user>/hypersteer-gemma2-2b-l20`, and the pod stops itself at the end, whether training succeeds or crashes.
3. **Resume** after a crash or stop: start the pod again, then run the same command with `RESUME=1` in front. It's exact: optimizer, LR schedule, RNG and data position all restore. `RESUME_FROM=<dir>` resumes from a checkpoint downloaded from HF onto a fresh pod (fresh optimizer).
4. **Optional laptop watchdog,** which restarts or rebuilds the pod after a crash:
   `python scripts/pod_watchdog.py --pod <id> --run train_<ts> --launch-env "EPOCHS=3 CKPT_EVERY=9000"`. Its rebuild path expects a code zip at the repo root and your HF write `.env` at `~/.runpod/hf_write.env`.

Each checkpoint is ~13 GB on the pod (5 GB weights + 8 GB optimizer state, two kept), so use a 150 GB volume. At ~0.5 s/step, one 8k-concept epoch takes ~5 h.

## Repo layout

```
hypersteer/          HyperSteer package (models, trainer with resume, data)    [upstream + changes]
config/              Hydra configs (experiment=hypersteer)                     [upstream]
scripts/             runpod_setup.sh, runpod_train.sh, pod_watchdog.py, pod_disk_guard.sh
demo/                app.py, live_app.py, fetch_and_run.py, sample_steering.py, PowerShell launchers
results/             sample outputs + results table for the final run
```

## Notes and known issues

- **The architecture differs from the paper.** The upstream code's default hypernet is 8 randomly initialized blocks with Gemma-7B-style dims (FFN 24576, 16 heads). The paper describes 22 blocks initialized from pretrained Gemma-2-2B. We kept the code default, and the paper's ablations suggest the paper's version generalizes better.
- **Negative strengths** collapse within a few tokens; training only ever used positive strengths.
- **Demo speed** on a laptop is CPU-bound (Python/HF per-token overhead), about 10 tokens/s. Fewer max tokens is the main lever.
- `uv sync --frozen` is required, because an upstream dev dependency (`ai_commit`) no longer resolves.

## Credits

- HyperSteer: Jiuding Sun, Sidharth Baskaran, Zhengxuan Wu, Michael Sklar, Christopher Potts, Atticus Geiger. *HyperSteer: Activation Steering at Scale with Hypernetworks*, arXiv:2506.03292 (2025).
- Base code: [AxBench](https://github.com/stanfordnlp/axbench) (Apache-2.0, where the HyperSteer code was released) via [HyperStuff/hypersteer](https://github.com/HyperStuff/hypersteer), using [pyvene](https://github.com/stanfordnlp/pyvene). Data: [`pyvene/axbench-concept16k`](https://huggingface.co/datasets/pyvene/axbench-concept16k). Model: Google Gemma-2-2B-it ([Gemma Terms of Use](https://ai.google.dev/gemma/terms)).

## License

Apache License 2.0, the same as the upstream AxBench code. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for attribution and a summary of our modifications.
