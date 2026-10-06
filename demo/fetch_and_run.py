"""Download trained HyperSteer weights from the Hugging Face repo and launch the demo.

    python demo/fetch_and_run.py --list                  # runs + checkpoint steps in the repo
    python demo/fetch_and_run.py --fetch-only            # newest run with final weights
    python demo/fetch_and_run.py --run train_<ts> --step 63000 --fetch-only
    python demo/fetch_and_run.py --quant none --hyper-quant 4bit --share   # download + launch

Weights land in assets/checkpoints/hf/<run> (or <run>_step<N>). Unknown flags are passed
through to demo/app.py. The default repo is private: `huggingface-cli login` with an account
that has access first.
"""

import argparse
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / "assets" / "checkpoints" / "hf"
WEIGHTS = "HyperSteer_weight.safetensors"


def index_runs(repo):
    """{run: {"final": bool, "steps": [N, ...]}} from the repo's file list."""
    runs = defaultdict(lambda: {"final": False, "steps": []})
    for f in HfApi().list_repo_files(repo):
        if m := re.fullmatch(rf"(train_[\d_]+)/train/{WEIGHTS}", f):
            runs[m[1]]["final"] = True
        elif m := re.fullmatch(rf"(train_[\d_]+)/train/checkpoints/step_(\d+)/{WEIGHTS}", f):
            runs[m[1]]["steps"].append(int(m[2]))
    return runs


def fetch(repo, run, step):
    """Download into assets/checkpoints/hf/ and return a run dir in the layout
    HyperSteerBackend expects: <dir>/config.yaml + <dir>/train/HyperSteer_weight.safetensors"""
    get = lambda f: Path(hf_hub_download(repo, f, local_dir=LOCAL))
    try:
        config = get(f"{run}/config.yaml")
    except EntryNotFoundError:
        # Runs still in progress only have checkpoints on HF (config.yaml is uploaded at the
        # end). The model architecture is the same across our runs, so borrow the newest
        # config from another run. Only cfg.model matters to HyperSteerBackend.
        donors = sorted(
            f.split("/")[0] for f in HfApi().list_repo_files(repo) if f.endswith("/config.yaml")
        )
        if not donors:
            raise
        print(f"WARNING: {run} has no config.yaml on HF yet; using {donors[-1]}/config.yaml")
        config = get(f"{donors[-1]}/config.yaml")
    if step is None:
        get(f"{run}/train/{WEIGHTS}")
        return LOCAL / run
    weights = get(f"{run}/train/checkpoints/step_{step}/{WEIGHTS}")
    run_dir = LOCAL / f"{run}_step{step}"
    (run_dir / "train").mkdir(parents=True, exist_ok=True)
    shutil.copy(config, run_dir / "config.yaml")
    if not (run_dir / "train" / WEIGHTS).exists():
        shutil.move(weights, run_dir / "train" / WEIGHTS)
    return run_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="RSD002/hypersteer-gemma2-2b-l20")
    p.add_argument("--run", help="train_<ts> (default: newest run with final weights)")
    p.add_argument("--step", type=int, help="mid-run checkpoint step instead of final weights")
    p.add_argument("--list", action="store_true")
    p.add_argument("--fetch-only", action="store_true")
    args, app_args = p.parse_known_args()

    runs = index_runs(args.repo)
    if args.list:
        for name, r in sorted(runs.items()):
            print(f"{name}  final={'yes' if r['final'] else 'no'}  steps={sorted(r['steps'])}")
        return

    run = args.run or max((n for n, r in runs.items() if r["final"]), default=None)
    if run not in runs:
        sys.exit(f"No run {run!r} in {args.repo}; try --list")
    if args.step is not None and args.step not in runs[run]["steps"]:
        sys.exit(f"{run} has no step {args.step}; available: {sorted(runs[run]['steps'])}")
    if args.step is None and not runs[run]["final"]:
        sys.exit(f"{run} has no final weights; pick --step from {sorted(runs[run]['steps'])}")

    run_dir = fetch(args.repo, run, args.step)
    print(f"Run dir: {run_dir}")
    if args.fetch_only:
        return
    cmd = [sys.executable, "demo/app.py", "--backend", "hypersteer", "--run-dir", str(run_dir)]
    sys.exit(subprocess.call(cmd + app_args, cwd=ROOT))


if __name__ == "__main__":
    main()
