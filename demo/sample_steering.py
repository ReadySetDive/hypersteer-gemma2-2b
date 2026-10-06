"""Batch-sample HyperSteer steered vs. unsteered outputs (same loader as demo/app.py).

    demo\\.venv-win\\Scripts\\python demo\\sample_steering.py --run-dir <run> [--quant 4bit]

Writes results/<run>_<time>.json and .md (e.g. --pairs results/pairs_6in_6out.json).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from app import HyperSteerBackend  # noqa: E402

# (concept, prompt, in training set?)  Training concepts are verbatim from
# axbench-concept500 2b/l20/train; the held-out ones match no training concept.
PAIRS = [
    ("references to files and file-related operations",
     "How do I make a good cup of coffee?", True),
    ("technical terms and error messages related to Python and Django programming",
     "Write a short poem about autumn.", True),
    ("terms related to governance, citizenship, and roles in societal structures",
     "Describe your ideal weekend.", True),
    ("phrases related to health communication and discussion",
     "Give me three tips for learning a new language.", True),
    ("references to the Golden Gate Bridge",
     "How do I make a good cup of coffee?", False),
    ("Cruel, extremely mean",
     "Tell me about your favorite animal.", False),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--quant", default="4bit", choices=["none", "8bit", "4bit"])
    p.add_argument("--factors", default="1.0,1.5,2.0")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pairs", help="JSON list of [concept, prompt, in_training_set] (default: PAIRS)")
    args = p.parse_args()
    pairs = [tuple(x) for x in json.loads(Path(args.pairs).read_text())] if args.pairs else PAIRS
    factors = [float(f) for f in args.factors.split(",")]

    t0 = time.time()
    backend = HyperSteerBackend(args.run_dir, quant=args.quant)
    backend.skip_prompted = True
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)

    rows = []
    for concept, prompt, in_train in pairs:
        row = {"concept": concept, "prompt": prompt, "in_training_set": in_train, "steered": {}}
        for f in factors:
            torch.manual_seed(args.seed)
            r = backend.generate(prompt, concept, f, args.max_new_tokens, args.temperature)
            row.setdefault("unsteered", r.unsteered)
            row["steered"][f] = {"text": r.steered, "info": r.info}
            print(f"[{concept[:40]}] x{f}: {r.info}", flush=True)
        rows.append(row)

    out = Path(__file__).parent.parent / "results"
    out.mkdir(exist_ok=True)
    stem = out / f"{Path(args.run_dir).name}_{time.strftime('%Y%m%d_%H%M%S')}"
    settings = {k: v for k, v in vars(args).items()} | {"factors": factors}
    stem.with_suffix(".json").write_text(json.dumps({"settings": settings, "rows": rows}, indent=2))

    md = [f"# HyperSteer samples: {Path(args.run_dir).name}", "", f"Settings: `{settings}`", ""]
    for row in rows:
        md += [f"## {row['concept']} ({'train' if row['in_training_set'] else 'HELD-OUT'})",
               f"**Prompt:** {row['prompt']}", "", f"**Unsteered:** {row['unsteered']}", ""]
        for f, s in row["steered"].items():
            md += [f"**x{f}** ({s['info']}): {s['text']}", ""]
    stem.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {stem}.json / .md")


if __name__ == "__main__":
    main()
