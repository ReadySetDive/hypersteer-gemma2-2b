"""HyperSteer live demo: bend a generation while it is being written (single user).

Streams one long generation and lets you change the steering *during* it:
  - the strength slider takes effect on the next token
  - "Apply concept" swaps in a new steering vector, computed by the hypernet against the
    text so far (prompt + everything generated)
  - an optional schedule ("40: 1.5", "80: concept=...") makes a run repeatable for talks
The end-of-turn tokens are blocked, so it keeps writing until Stop or the token cap.
Each span of output is tinted by the concept (hue) and strength (opacity) it was written under.

    demo\\.venv-win\\Scripts\\python demo\\live_app.py --run-dir <run> [--share]

Single user by design: the controls are global, and only one generation runs at a time.
"""

from __future__ import annotations

import argparse
import html
import os
import sys
import threading
import time
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent))
from app import HyperSteerBackend, sampling  # noqa: E402

DEFAULT_PROMPT = (
    "Write a long, detailed story about a road trip across the country with an old friend. "
    "Describe the places, the people you meet, the food, and the conversations along the way."
)
DEFAULT_CONCEPT = "references to organic compounds or organic materials"
HUES = [210, 30, 140, 280, 0, 60, 180, 320]  # one per concept, in order of first use


class LiveSteerer:
    """Owns the model, the steering hook and the mutable steering state."""

    def __init__(self, backend: HyperSteerBackend):
        self.b = backend
        self.torch = backend.torch
        self.model, self.tok = backend.model, backend.tok
        self.layer = backend.hs.layer
        self.factor = 0.0
        self.concept = DEFAULT_CONCEPT
        self.pending_concept = None  # applied at the next token, against the text so far
        self.v_unit = None           # steering vector at strength 1
        self.hook_on = False
        self.stop = threading.Event()
        self.running = threading.Lock()
        self.segments = []           # [concept, factor, [token ids]] in generation order
        self.schedule = {}           # step -> list of ("factor", x) / ("concept", text)
        self.step = 0
        self.model.model.layers[self.layer].register_forward_hook(self._hook)
        # Never let the reply end or start a new turn
        self.blocked = sorted({i for i in (
            self.tok.eos_token_id, self.tok.pad_token_id,
            self.tok.convert_tokens_to_ids("<end_of_turn>"),
            self.tok.convert_tokens_to_ids("<start_of_turn>"),
        ) if isinstance(i, int) and i >= 0})

    # -- steering ----------------------------------------------------------------------
    def _hook(self, module, args, out):
        if not self.hook_on or self.v_unit is None or self.factor == 0:
            return None
        add = (self.factor * self.v_unit)[:, None, :]
        if isinstance(out, tuple):
            return (out[0] + add.to(out[0].dtype),) + tuple(out[1:])
        return out + add.to(out.dtype)

    def vector_for(self, ids, concept):
        """Hypernet vector (strength 1) for `concept`, conditioned on token ids `ids`
        (the prompt, or prompt + generated text). The hook is off for this forward pass."""
        torch, b = self.torch, self.b
        was_on, self.hook_on = self.hook_on, False
        try:
            with torch.no_grad():
                c = b.hs.base_model_tokenizer(concept, return_tensors="pt",
                                              add_special_tokens=True).to(ids.device)
                hidden = self.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                    output_hidden_states=True).hidden_states[self.layer]
                return b.hs.concept_embedding(
                    input_ids=None,
                    inputs_embeds=self.model.model.embed_tokens(c["input_ids"]),
                    attention_mask=c["attention_mask"],
                    base_encoder_hidden_states=hidden,
                    base_encoder_attention_mask=torch.ones_like(ids),
                    output_hidden_states=False,
                ).last_hidden_state
        finally:
            self.hook_on = was_on

    # -- per-step callbacks (outside the model forward, so recomputing v is safe) -------
    def on_step(self, input_ids, scores):
        """LogitsProcessor: runs before each new token is sampled."""
        for kind, val in self.schedule.pop(self.step, []):
            if kind == "factor":
                self.factor = val
            else:
                self.pending_concept = val
        if self.pending_concept is not None:
            self.concept, self.pending_concept = self.pending_concept, None
            self.v_unit = self.vector_for(input_ids[:1], self.concept)
        scores[:, self.blocked] = -float("inf")
        return scores

    def on_token(self, input_ids, scores, **kwargs):
        """StoppingCriteria: runs after each token is appended; records which steering
        produced it."""
        tok = int(input_ids[0, -1])
        key = (self.concept, round(self.factor, 2))
        if self.segments and tuple(self.segments[-1][:2]) == key:
            self.segments[-1][2].append(tok)
        else:
            self.segments.append([key[0], key[1], [tok]])
        self.step += 1
        return self.stop.is_set()

    # -- generation ----------------------------------------------------------------------
    def generate(self, prompt, max_new_tokens, temperature, rep_penalty):
        from transformers import (LogitsProcessor, LogitsProcessorList, StoppingCriteria,
                                  StoppingCriteriaList)

        steerer, torch = self, self.torch

        class Steer(LogitsProcessor):
            def __call__(self, input_ids, scores):
                return steerer.on_step(input_ids, scores)

        class Recorder(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                done = steerer.on_token(input_ids, scores)
                return torch.full((input_ids.shape[0],), done, dtype=torch.bool,
                                  device=input_ids.device)

        chat = self.b._chat(prompt)
        inputs = self.tok(chat, return_tensors="pt").to(self.model.device)
        self.segments, self.step = [], 0
        self.stop.clear()
        self.pending_concept = None
        self.v_unit = self.vector_for(inputs["input_ids"], self.concept)
        self.hook_on = True
        try:
            with self.torch.no_grad():
                self.model.generate(
                    **inputs, max_new_tokens=int(max_new_tokens),
                    repetition_penalty=float(rep_penalty), **sampling(float(temperature)),
                    logits_processor=LogitsProcessorList([Steer()]),
                    stopping_criteria=StoppingCriteriaList([Recorder()]),
                )
        finally:
            self.hook_on = False

    def render(self):
        """Segments -> HTML: hue = concept, opacity = |strength|, dashed = negative."""
        concepts, parts = [], []
        for concept, factor, toks in list(self.segments):
            if concept not in concepts:
                concepts.append(concept)
            hue = HUES[concepts.index(concept) % len(HUES)]
            alpha = min(abs(factor) / 2.5, 0.65)
            # Colors come from CSS (.steer / .dark .steer) so they suit light and dark mode
            cls = "steer neg" if factor < 0 else "steer"
            text = html.escape(self.tok.decode(toks, skip_special_tokens=True))
            parts.append(f'<span class="{cls}" title="{html.escape(concept)} @ x{factor:g}" '
                         f'style="--h:{hue};--a:{alpha:.2f}">{text.replace(chr(10), "<br>")}</span>')
        legend = " &nbsp; ".join(
            f'<span class="steer" style="--h:{HUES[i % len(HUES)]};--a:0.6;padding:0 6px">'
            f'{html.escape(c)}</span>'
            for i, c in enumerate(concepts))
        body = "".join(parts) or "<i>…</i>"
        return (f'<div style="font-size:0.85em;margin-bottom:8px">{legend}</div>'
                f'<div style="line-height:1.7;font-size:1.05em">{body}</div>')


def parse_schedule(text):
    """Lines like '40: 1.5' (strength at token 40) or '80: concept=...'."""
    schedule = {}
    for line in text.splitlines():
        if ":" not in line.strip():
            continue
        step, action = line.split(":", 1)
        step, action = int(step.strip()), action.strip()
        if action.lower().startswith("concept="):
            item = ("concept", action.split("=", 1)[1].strip())
        else:
            item = ("factor", float(action))
        schedule.setdefault(step, []).append(item)
    return schedule


def build_ui(live: LiveSteerer):
    def start(prompt, concept, factor, max_tokens, temperature, rep_penalty, schedule_text):
        if not live.running.acquire(blocking=False):
            raise gr.Error("A generation is already running (single-user demo) - press Stop.")
        try:
            live.concept, live.factor = concept.strip() or DEFAULT_CONCEPT, float(factor)
            live.schedule = parse_schedule(schedule_text or "")
            worker = threading.Thread(
                target=live.generate, args=(prompt, max_tokens, temperature, rep_penalty))
            t0 = time.time()
            worker.start()
            while worker.is_alive():
                time.sleep(0.15)
                yield live.render(), f"{live.step} tokens · {time.time() - t0:.0f}s · " \
                                     f"now: {live.concept} @ x{live.factor:g}"
            worker.join()
            yield live.render(), f"done · {live.step} tokens · {time.time() - t0:.0f}s"
        finally:
            live.running.release()

    def set_factor(f):
        live.factor = float(f)

    def apply_concept(c):
        if c.strip():
            live.pending_concept = c.strip()

    def stop():
        live.stop.set()

    # Tints: pastel behind dark text in light mode; deeper, less saturated behind light
    # text in dark mode (bright pastels under white text were hard to read)
    css = """
    .steer { background: hsla(var(--h), 85%, 62%, var(--a)); border-radius: 3px; }
    .dark .steer { background: hsla(var(--h), 55%, 38%, calc(var(--a) * 1.3));
                   color: #f3f4f6; }
    .steer.neg { outline: 1px dashed #e05252; }
    """
    with gr.Blocks(title="HyperSteer Live", css=css) as demo:
        gr.Markdown(
            "# HyperSteer Live\n"
            "Start a long generation, then **drag the strength** or **change the steering prompt** "
            "while it writes (edit it, then **Update steering prompt**); changes hit the next token. Text is tinted by the concept "
            "(color) and strength (intensity) it was written under; hover a span for details. "
            "It never ends on its own - press **Stop**.")
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(DEFAULT_PROMPT, label="Prompt (long, open-ended works best)",
                                    lines=4)
                concept = gr.Textbox(DEFAULT_CONCEPT, label="Steering prompt (concept)")
                apply_btn = gr.Button("Update steering prompt (takes effect mid-generation)")
                factor = gr.Slider(-3.0, 3.0, value=0.0, step=0.1, label="Strength (live)")
                with gr.Row():
                    go = gr.Button("Start", variant="primary")
                    stop_btn = gr.Button("Stop", variant="stop")
                with gr.Accordion("Schedule (repeatable runs)", open=False):
                    schedule = gr.Textbox(
                        label="One change per line: token: strength  or  token: concept=...",
                        placeholder="0: 0\n40: 1.5\n120: concept=cooking instructions and "
                                    "process-related terms\n200: 0", lines=5)
                with gr.Accordion("Generation settings", open=False):
                    max_tokens = gr.Slider(64, 2000, value=600, step=32, label="Token cap")
                    temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="Temperature")
                    rep_penalty = gr.Slider(1.0, 1.5, value=1.1, step=0.05,
                                            label="Repetition penalty (keeps long runs fresh)")
            with gr.Column(scale=2):
                status = gr.Markdown()
                out = gr.HTML()

        go.click(start, [prompt, concept, factor, max_tokens, temperature, rep_penalty,
                         schedule], [out, status])
        # Separate, unlimited-concurrency events so they run *while* start is streaming
        factor.change(set_factor, factor, None, concurrency_limit=None,
                      trigger_mode="always_last")
        apply_btn.click(apply_concept, concept, None, concurrency_limit=None)
        concept.submit(apply_concept, concept, None, concurrency_limit=None)
        stop_btn.click(stop, None, None, concurrency_limit=None)
    return demo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--quant", choices=["none", "8bit", "4bit"], default="none")
    p.add_argument("--hyper-quant", choices=["none", "8bit", "4bit"], default="4bit")
    p.add_argument("--share", action="store_true")
    p.add_argument("--port", type=int, default=7861)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()

    live = LiveSteerer(HyperSteerBackend(args.run_dir, quant=args.quant,
                                         hyper_quant=args.hyper_quant))
    auth = tuple(os.environ["DEMO_AUTH"].split(":", 1)) if os.environ.get("DEMO_AUTH") else None
    build_ui(live).queue().launch(server_name=args.host, server_port=args.port,
                                  share=args.share, auth=auth)


if __name__ == "__main__":
    main()
