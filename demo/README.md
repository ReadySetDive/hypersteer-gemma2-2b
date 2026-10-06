# Demos

Both demos load Gemma-2-2B-it plus a trained HyperSteer hypernetwork, and steer by adding `strength × v` to layer 20's output at every position. That's the same operation as the paper's intervention, implemented as a plain PyTorch hook (≈4× faster than going through pyvene, with identical output).

## Setup (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File demo\setup_windows.ps1   # once; creates demo\.venv-win
demo\.venv-win\Scripts\huggingface-cli login                      # needs access to the HF repo + Gemma license
```
Weights come from the private HF repo `RSD002/hypersteer-gemma2-2b-l20`. The launch scripts download the final run on first use. Other runs and checkpoints:
```powershell
demo\.venv-win\Scripts\python demo\fetch_and_run.py --list
demo\.venv-win\Scripts\python demo\fetch_and_run.py --fetch-only --run train_20261005_032732706478 --step 63000
```

## 1. Side-by-side app: `run_demo.ps1` (port 7860)

Type a steering prompt and a user prompt, and compare up to four outputs: **No steering**, **HyperSteer**, **Instruction as prompt** (AxBench's prompting baseline), and **Prompt + HyperSteer**.
- Strength runs from −3 to 3 (default 1.3), and temperature can go down to 0 (greedy, deterministic). Panels render markdown.
- Built-in libraries of **held-in** (seen in training) and **held-out** (never seen) concepts.
- Unticking a panel skips its generation. Ctrl+Enter submits. "Combine a second instruction" adds a second vector.
- `-Share` makes a public `gradio.live` link with a login. `-Run train_<ts>` selects another downloaded run.

## 2. Live app: `run_live.ps1` (port 7861, single user)

One long, never-ending generation that you steer *while it writes*:
- **Strength slider:** takes effect on the next token.
- **Update steering prompt:** the hypernet recomputes the vector against the text so far, and it's swapped in between tokens.
- **Schedule** (for talks): lines like `0: 0`, `40: 1.5`, `150: concept=descriptions of cities with rich culture and history`.
- The output is tinted by concept (color) and strength (intensity), and you can hover a span to see it.
- End-of-turn tokens are blocked, so it runs until **Stop** or the token cap.

## Memory and speed (8 GB laptop GPU)

The defaults are Gemma in bf16 (`-Quant none`) and the hypernet in 4-bit (`-HyperQuant 4bit`), about 6.8 GB. The hypernet runs once per request, so quantizing it costs no speed. Generation is CPU-bound (Python per-token overhead) at about 10 tokens/s, so max tokens is the main speed lever. The two apps don't fit on an 8 GB GPU at the same time.

## Other tools

- `sample_steering.py`: batch samples (unsteered vs. several strengths) to `results/`, e.g. `--pairs results/pairs_6in_6out.json --factors 0,1.0,1.5,2.0`.
- `fetch_and_run.py`: list or download runs from HF, optionally launching `app.py`.
- On Linux, use the repo env (`uv sync --frozen`) and add `--with "gradio>=4.44,<6" --with "huggingface-hub<1.0"` to `uv run`.
