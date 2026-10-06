# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "gradio>=4.44,<6",
#     "torch>=2.4",
#     "transformers>=4.45,<5",
#     "accelerate",
# ]
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# ///
"""Interactive HyperSteer demo: steer Gemma-2-2B-it with a plain-English instruction.

Gemma mode (local GPU, no hypernetwork): real Gemma-2-2B-it generations; the hypernetwork is
stubbed by a pluggable vector source ("zero" = pass-through, "actdiff" = cheap heuristic):
    uv run demo/app.py                       # zero vector: HyperSteer column == no steering
    uv run demo/app.py --vector actdiff      # activation-difference vector, actually steers
Real mode (GPU pod, repo's venv + gradio overlay, after training):
    uv run --with "gradio>=4.44,<6" python demo/app.py --backend hypersteer \
        --run-dir assets/checkpoints/train_<ts> --share
Real mode on an 8 GB Windows laptop GPU (env from demo/setup_windows.ps1):
    demo/.venv-win/Scripts/python demo/app.py --backend hypersteer --quant 4bit --run-dir <run>
(gradio is deliberately NOT added to pyproject.toml: the pod installs with --frozen.)

Every request produces three outputs side by side:
  1. no steering            - plain Gemma-2-2B-it
  2. HyperSteer             - steering vector from the hypernetwork added at layer 20
  3. instruction as prompt  - the steering instruction prepended to the prompt (AxBench's
                              prompting baseline), i.e. what HyperSteer saves context tokens vs.
"""

from __future__ import annotations

import argparse
import os
import random
import textwrap
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import gradio as gr

# Factor = scalar multiplier on the generated vector: residual += factor * v (see
# HyperAdditiveIntervention.forward). Configs sweep 0.6-1.4 and factor selection goes to 2-3.
# Negative factors subtract the vector (steer away from the concept)
FACTOR_MIN, FACTOR_MAX, FACTOR_DEFAULT = -3.0, 3.0, 1.3

# AxBench's prompting baseline template (hypersteer/data/axbench.py T_PROMPT_STEERING)
PROMPT_STEERING = (
    "You must answer the question with content related to %s even if it is not "
    "related to the question or it does not make sense."
)

# AxBench-style concepts are short descriptions of a feature, not imperative instructions.
# From pyvene/axbench-concept16k (2b/l20): the 8k-concept run trained on concept ids
# 0-7999, so "held-in" = seen in training, "held-out" = ids >= 8000, never seen.
HELD_IN = [
    ["terms related to petroleum and energy extraction", "Write a short poem about autumn.", FACTOR_DEFAULT],
    ["concepts related to spiritual beliefs and practices", "How do I make a good cup of coffee?", FACTOR_DEFAULT],
    ["phrases indicating uncertainty or ambiguity", "Give me three tips for learning a new language.", FACTOR_DEFAULT],
    ["positive descriptors and characteristics related to individuals", "Describe your ideal weekend.", FACTOR_DEFAULT],
    ["terms related to experimental procedures and measurements", "How do I bake chocolate chip cookies?", FACTOR_DEFAULT],
    ["key phrases related to corporate communications and public statements", "Write a text to my friend about weekend plans.", FACTOR_DEFAULT],
    ["mentions of guest appearances and roles in various shows", "What should I pack for a beach trip?", FACTOR_DEFAULT],
]
HELD_OUT = [
    ["terms related to manufacturing processes and material treatments", "How do I make a good cup of coffee?", FACTOR_DEFAULT],
    ["references to safety regulations and management practices in various industries", "Give me three tips for learning a new language.", FACTOR_DEFAULT],
    ["references to organic compounds or organic materials", "Describe your ideal weekend.", FACTOR_DEFAULT],
    ["cooking instructions and process-related terms", "Write a short poem about autumn.", FACTOR_DEFAULT],
    ["chemical components and their concentrations in various substances", "How do I bake chocolate chip cookies?", FACTOR_DEFAULT],
    ["descriptions of cities with rich culture and history", "Write a text to my friend about weekend plans.", FACTOR_DEFAULT],
    ["information pertaining to card games and gameplay mechanics", "What should I pack for a beach trip?", FACTOR_DEFAULT],
]


OFF = "(off)"
ALL_OUTPUTS = ("unsteered", "steered", "prompted", "both")  # both = prompt + HyperSteer


def sampling(temperature):
    """generate() kwargs; temperature 0 = greedy decoding (deterministic)."""
    return ({"do_sample": True, "temperature": temperature} if temperature > 0
            else {"do_sample": False})


@dataclass
class SteerResult:
    unsteered: str
    steered: str
    prompted: str
    info: str = ""
    both: str = OFF  # instruction in the prompt AND the steering vector


class SteeringBackend(ABC):
    name: str

    @abstractmethod
    def generate(
        self,
        prompt: str,
        steering_text: str,
        factor: float,
        max_new_tokens: int,
        temperature: float,
        second_steering_text: str = "",
        second_factor: float = 0.0,
        outputs: tuple | None = None,  # subset of ALL_OUTPUTS to generate; None = all
    ) -> SteerResult: ...


class GemmaBackend(SteeringBackend):
    """Real Gemma-2-2B-it generation with the hypernetwork stubbed out. A forward hook on
    decoder layer `layer` does residual += factor * v at every prompt and generated position,
    same as HyperAdditiveIntervention. Where v comes from is the `vector` source:
      zero     pass-through stand-in for the hypernet; HyperSteer column == no steering
      actdiff  mean layer-`layer` activation of the concept text minus that of a neutral
               text, rescaled to the typical residual norm (a CAA-style heuristic, not
               HyperSteer). Gives the UI something that visibly steers.
    Unsteered and steered generations share a seed, so a zero vector reproduces the
    unsteered text exactly."""

    NEUTRAL = "Let's talk about something."
    CONCEPT = "Let's talk about {}."
    # ||v|| = SCALE * mean residual norm, tuned by hand so factor ~1 steers fluently and
    # factor ~2 over-steers (same feel as the slider's HyperSteer range)
    SCALE = 0.6

    def __init__(self, vector: str = "zero", model_name: str = "google/gemma-2-2b-it",
                 layer: int = 20, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.vector = vector
        self.layer = layer
        self.name = f"{model_name}, layer {layer}, steering vector: {vector}"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        # eager attention: Gemma-2 uses logit softcapping, which sdpa doesn't implement
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device,
            attn_implementation="eager",
        ).eval()
        self._v = None  # vector the hook adds; None = hook is a no-op
        self.model.model.layers[layer].register_forward_hook(self._hook)
        if vector == "actdiff":
            self._neutral_mean, self._resid_norm = self._layer_stats(self.NEUTRAL)

    def _hook(self, module, args, output):
        if self._v is None:
            return None
        if isinstance(output, tuple):
            return (output[0] + self._v,) + tuple(output[1:])
        return output + self._v

    def _layer_stats(self, text):
        """Mean and mean norm of layer-`layer` outputs over text tokens (BOS excluded: its
        activation norm is an outlier in Gemma)."""
        ids = self.tok(text, return_tensors="pt").input_ids.to(self.model.device)
        with self.torch.no_grad():
            hs = self.model(ids, output_hidden_states=True).hidden_states[self.layer + 1][0, 1:]
        hs = hs.float()
        return hs.mean(0), hs.norm(dim=-1).mean().item()

    def _concept_vector(self, concept):
        if self.vector == "zero":
            return None
        mean, _ = self._layer_stats(self.CONCEPT.format(concept))
        d = mean - self._neutral_mean
        return d / d.norm() * self.SCALE * self._resid_norm

    def _chat_ids(self, user_text):
        return self.tok.apply_chat_template(
            [{"role": "user", "content": user_text}], add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

    def _generate(self, ids, max_new_tokens, temperature, seed, v=None):
        self._v = v
        try:
            self.torch.manual_seed(seed)
            with self.torch.no_grad():
                out = self.model.generate(ids, max_new_tokens=max_new_tokens,
                                          **sampling(temperature))
        finally:
            self._v = None
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    def generate(self, prompt, steering_text, factor, max_new_tokens, temperature,
                 second_steering_text="", second_factor=0.0, outputs=None):
        outputs = outputs or ALL_OUTPUTS
        t0 = time.time()
        seed = random.randrange(2**31)
        v = None
        if self.vector != "zero":
            v = factor * self._concept_vector(steering_text)
            if second_steering_text.strip() and second_factor > 0:
                v = v + second_factor * self._concept_vector(second_steering_text)
            v = v.to(self.model.dtype)

        ids = self._chat_ids(prompt)
        unsteered = (self._generate(ids, max_new_tokens, temperature, seed)
                     if "unsteered" in outputs else OFF)
        steered = (self._generate(ids, max_new_tokens, temperature, seed, v=v)
                   if "steered" in outputs else OFF)
        prompted = OFF if "prompted" not in outputs else self._generate(
            self._chat_ids(f"{PROMPT_STEERING % steering_text}\n\nQuestion: {prompt}"),
            max_new_tokens, temperature, seed,
        )
        info = f"{time.time() - t0:.1f}s, seed={seed}"
        if self.vector == "zero":
            info += " · zero vector (hypernet stubbed), so HyperSteer == no steering"
        return SteerResult(unsteered, steered, prompted, info=info)


class HyperSteerBackend(SteeringBackend):
    """Real backend, mirroring scripts/inference.py and HyperSteer.predict_step. Tested on
    an 8 GB laptop 4070 with --quant 4bit (~3.8 GB peak, ~12 s/request at 48 tokens) and
    --quant 8bit (~5.6 GB peak, ~35 s/request: bnb int8 matmuls are slow). Unquantized
    (pod) mode is still untested.

    run_dir layout from scripts/train.py:
        <run_dir>/config.yaml                       (ExperimentConfig dump)
        <run_dir>/train/HyperSteer_weight.safetensors
    """

    name = "hypersteer"

    def __init__(self, run_dir: str, device: str = "cuda", quant: str = "none",
                 hyper_quant: str | None = None):
        # quant: the target Gemma (runs on every generated token - quantizing it costs
        # speed); hyper_quant: the hypernet (runs once per request), defaults to quant
        hyper_quant = quant if hyper_quant is None else hyper_quant
        import pyvene
        import torch
        from omegaconf import OmegaConf
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        from hypersteer import get_model
        from hypersteer.utils.configs import ExperimentConfig, config_to_pydantic
        from hypersteer.utils.helpers import configure_tokenizer_model
        from hypersteer.utils.model_utils import get_prefix_length
        from hypersteer.utils.patch import monkeypatch_ax_model_generate

        import warnings

        warnings.filterwarnings("ignore", message="MatMul8bitLt: inputs will be cast")
        self.torch = torch
        run_dir = Path(run_dir)
        # config.yaml from the Linux pod tags dump_dir as pathlib.PosixPath, which can't be
        # instantiated on Windows; omegaconf looks the class up at load time, so alias it
        import pathlib
        posix_path = pathlib.PosixPath
        pathlib.PosixPath = pathlib.Path
        try:
            cfg = config_to_pydantic(OmegaConf.load(run_dir / "config.yaml"), ExperimentConfig)
        finally:
            pathlib.PosixPath = posix_path
        self.cfg = cfg
        self.name = (f"hypersteer (Gemma {quant if quant != 'none' else 'bf16'}, "
                     f"hypernet {hyper_quant if hyper_quant != 'none' else 'bf16'})")

        # Target (steered) model: gemma-2-2b-it. The attn hypernet is built from config and
        # loaded from the checkpoint; only the *tokenizer* of gemma-2-2b (base_model_name)
        # is needed, not its weights.
        self.tok = AutoTokenizer.from_pretrained(cfg.model.target_model_name, model_max_length=512)
        if quant == "none":
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model.target_model_name, torch_dtype=torch.bfloat16
            ).eval().to(device)
        else:
            # Quantized on load from the regular HF cache. bnb models are placed by device_map
            # and raise on .to(), so pyvene's set_device must not move the model (below).
            qcfg = {
                "8bit": BitsAndBytesConfig(load_in_8bit=True),
                "4bit": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.bfloat16),
            }[quant]
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model.target_model_name, torch_dtype=torch.bfloat16,
                quantization_config=qcfg, device_map={"": device},
            ).eval()
            set_device = pyvene.IntervenableModel.set_device
            pyvene.IntervenableModel.set_device = (
                lambda ax_model, dev, set_model=True: set_device(ax_model, dev, set_model=False)
            )
        configure_tokenizer_model(self.model, self.tok)

        # Mirrors inference.py:334-350 (it passes model_config as training_args too)
        self.hs = get_model(
            "HyperSteer", model=self.model, tokenizer=self.tok, device=device,
            training_args=cfg.model, model_config=cfg.model,
        )
        self._load_hypernet(run_dir / "train", device, hyper_quant)
        self.hs.ax.eval()
        self.hs.ax.to(torch.bfloat16)
        self.hs.concept_embedding.eval()
        self.hs.ax_model = monkeypatch_ax_model_generate(self.hs.ax_model)  # inference.py:156

        self.prefix_length = get_prefix_length(self.tok)  # chat-model prefix (inference.py:262)
        self.dump_dir = Path("assets/cache/demo")
        self.quant = hyper_quant  # hot reload loads into the hypernet
        self.ckpt_label = run_dir.name
        self.lock = threading.Lock()  # generate vs. hot weight reload

    def reload_weights(self, weights: Path, label: str):
        """Swap in another checkpoint's hypernet weights in place (same architecture), so a
        running demo - and its share URL - picks up new training checkpoints."""
        from safetensors.torch import load_file

        if self.quant != "none":
            raise NotImplementedError("hot reload needs --quant none")
        state = {k: v for k, v in load_file(str(weights), device="cpu").items()
                 if not k.startswith("embed_tokens.")}
        with self.lock:
            self.hs.concept_embedding.load_state_dict(state, strict=False)
            self.ckpt_label = label
        del state

    def watch_checkpoints(self, ckpt_root: Path, total_steps: int, interval: int = 60):
        """Background thread: load the newest complete checkpoint under ckpt_root
        (<run>/train/checkpoints; complete = trainer_state.pt written after the weights),
        or the run's final weights once training ends."""
        name = f"{self.hs}_weight.safetensors"
        final = ckpt_root.parent / name
        loaded = None

        def newest():
            if final.exists():
                return "final", final, "final weights (training complete)"
            steps = sorted((int(d.name.split("_")[1]), d) for d in ckpt_root.glob("step_*")
                           if (d / "trainer_state.pt").exists() and (d / name).exists())
            if not steps:
                return None
            n, d = steps[-1]
            return n, d / name, f"step {n:,} / {total_steps:,} ({100 * n / total_steps:.0f}% of training)"

        while True:
            try:
                found = newest()
                if found and found[0] != loaded:
                    print(f"[watch] loading {found[1]}", flush=True)
                    self.reload_weights(found[1], found[2])
                    loaded = found[0]
                    print(f"[watch] now serving {found[2]}", flush=True)
            except Exception as e:
                print(f"[watch] error (keeps old weights): {e}", flush=True)
            time.sleep(interval)

    def _load_hypernet(self, train_dir: Path, device: str, quant: str):
        """HyperSteer.load, fitted to an 8 GB GPU. load() builds the hypernet (~2.6B params)
        on the GPU and loads the state dict onto the GPU too (a ~2x transient peak). Here:
        - built on CPU in bf16 without random init (every weight comes from the checkpoint)
        - its own embed_tokens (256k x d, ~0.6B params) is dropped: predict_step embeds
          concepts with the TARGET model's embed_tokens and passes inputs_embeds
        - with quant != none, the decoder layers' Linears are bnb-quantized like the target
        bf16 sizes: ~3.9 GB without embed_tokens, ~2 GB at 8bit, ~1.1 GB at 4bit."""
        from contextlib import contextmanager

        import hypersteer.models.hypersteer as hs_mod
        from safetensors.torch import load_file
        from transformers.modeling_utils import no_init_weights

        torch = self.torch
        set_default_device = hs_mod.set_default_device
        hs_mod.set_default_device = contextmanager(lambda _device: (yield))  # build on CPU
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with no_init_weights():
                self.hs.make_model(mode="steering")
        finally:
            hs_mod.set_default_device = set_default_device
            torch.set_default_dtype(default_dtype)

        hypernet = self.hs.concept_embedding
        state = load_file(str(train_dir / f"{self.hs}_weight.safetensors"), device="cpu")
        if self.cfg.model.hypernet_type == "attn":
            hypernet.embed_tokens = torch.nn.Embedding(1, hypernet.config.hidden_size)
            state = {k: v for k, v in state.items() if not k.startswith("embed_tokens.")}
            missing, unexpected = hypernet.load_state_dict(state, strict=False)
            assert missing == ["embed_tokens.weight"] and not unexpected, (missing, unexpected)
        else:
            hypernet.load_state_dict(state)
        del state
        hypernet.to(dtype=torch.bfloat16)
        if quant != "none":
            self._quantize_linears(hypernet.layers, quant)
        hypernet.to(device)  # bnb quantizes weights on this move

        head = train_dir / f"{self.hs}_selection_head.safetensors"
        if self.cfg.model.use_selection_head and hasattr(self.hs.ax, "selection_head"):
            if head.exists():
                self.hs.ax.selection_head.load_state_dict(load_file(str(head), device=device))

    def _quantize_linears(self, module, quant: str):
        """Swap every nn.Linear under `module` for its bitsandbytes counterpart, carrying
        over the loaded (CPU) weights; quantization happens on the later .to(cuda)."""
        import bitsandbytes as bnb

        nn = self.torch.nn
        for name, child in module.named_children():
            if not isinstance(child, nn.Linear):
                self._quantize_linears(child, quant)
                continue
            has_bias = child.bias is not None
            if quant == "8bit":
                q = bnb.nn.Linear8bitLt(child.in_features, child.out_features, bias=has_bias,
                                        has_fp16_weights=False)
                q.weight = bnb.nn.Int8Params(child.weight.data, requires_grad=False,
                                             has_fp16_weights=False)
            else:
                q = bnb.nn.Linear4bit(child.in_features, child.out_features, bias=has_bias,
                                      compute_dtype=self.torch.bfloat16, quant_type="nf4")
                q.weight = bnb.nn.Params4bit(child.weight.data, requires_grad=False,
                                             quant_type="nf4")
            if has_bias:
                q.bias = nn.Parameter(child.bias.data, requires_grad=False)
            setattr(module, name, q)

    def _chat(self, user_text: str) -> str:
        # Same formatting as AxBenchDatasetFactory.create_eval_ds: chat template with
        # generation prompt, BOS stripped ([1:]) because the tokenizer re-adds it
        ids = self.tok.apply_chat_template(
            [{"role": "user", "content": user_text}], tokenize=True, add_generation_prompt=True
        )[1:]
        return self.tok.decode(ids)

    def _plain_generate(self, text, max_new_tokens, temperature):
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      **sampling(temperature))
        return self.tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def _steering_vector(self, chat_prompt, concept):
        """v for one concept: the "Concept embedding (v)" block of HyperSteer.predict_step
        (concept embedded with the target's embed_tokens, cross-attending to the target's
        layer-`layer` residual stream on the prompt). Returns [1, d]."""
        torch = self.torch
        dev = self.model.device
        with torch.no_grad():
            inputs = self.tok(chat_prompt, return_tensors="pt").to(dev)
            c = self.hs.base_model_tokenizer(concept, return_tensors="pt",
                                             add_special_tokens=True).to(dev)
            base_hidden = self.model(**inputs, output_hidden_states=True).hidden_states[self.hs.layer]
            return self.hs.concept_embedding(
                input_ids=None,
                inputs_embeds=self.model.model.embed_tokens(c["input_ids"]),
                attention_mask=c["attention_mask"],
                base_encoder_hidden_states=base_hidden,
                base_encoder_attention_mask=inputs["attention_mask"],
                output_hidden_states=False,
            ).last_hidden_state

    def _steered_generate(self, chat_prompt, v, max_new_tokens, temperature):
        """Generate with residual += v at the output of decoder layer `layer`, at every
        prompt and generated position: what HyperAdditiveIntervention does inside
        pyvene's generate (base + mag * v), minus pyvene's per-token overhead (~15x)."""
        def add_v(module, args, out):
            if isinstance(out, tuple):
                return (out[0] + v[:, None, :].to(out[0].dtype),) + tuple(out[1:])
            return out + v[:, None, :].to(out.dtype)

        handle = self.model.model.layers[self.hs.layer].register_forward_hook(add_v)
        try:
            return self._plain_generate(chat_prompt, max_new_tokens, temperature)
        finally:
            handle.remove()

    def generate(self, prompt, steering_text, factor, max_new_tokens, temperature,
                 second_steering_text="", second_factor=0.0, outputs=None):
        outputs = outputs or ALL_OUTPUTS
        if getattr(self, "skip_prompted", False):
            outputs = tuple(o for o in outputs if o != "prompted")

        t0 = time.time()
        chat_prompt = self._chat(prompt)
        combine = bool(second_steering_text.strip()) and second_factor != 0
        info_combine = f" · combined with x{second_factor:g} '{second_steering_text}'" if combine else ""

        prompted_chat = self._chat(f" {PROMPT_STEERING % steering_text}\n\nQuestion: {prompt}")

        def vector(chat):
            # v depends on the prompt too (the hypernet cross-attends to its activations)
            v = factor * self._steering_vector(chat, steering_text)
            if combine:
                v = v + second_factor * self._steering_vector(chat, second_steering_text)
            return v

        ckpt = self.ckpt_label
        steered = both = OFF
        if "steered" in outputs or "both" in outputs:
            with self.lock:  # vs. hot weight reload
                v = vector(chat_prompt) if "steered" in outputs else None
                v_both = vector(prompted_chat) if "both" in outputs else None
                ckpt = self.ckpt_label
            if v is not None:
                steered = self._steered_generate(chat_prompt, v, max_new_tokens, temperature)
            if v_both is not None:
                both = self._steered_generate(prompted_chat, v_both, max_new_tokens, temperature)
        unsteered = (self._plain_generate(chat_prompt, max_new_tokens, temperature)
                     if "unsteered" in outputs else OFF)
        prompted = (self._plain_generate(prompted_chat, max_new_tokens, temperature)
                    if "prompted" in outputs else OFF)
        info = f"{time.time() - t0:.1f}s · checkpoint: {ckpt}{info_combine}"
        return SteerResult(unsteered, steered, prompted, info=info, both=both)


def build_ui(backend: SteeringBackend, show_prompted: bool = True) -> gr.Blocks:
    def run(steering_text, prompt, factor, max_new_tokens, temperature, steer2, factor2,
            show_base, show_steer, show_prompt, show_both):
        if not prompt.strip():
            raise gr.Error("Enter a prompt.")
        if not steering_text.strip():
            raise gr.Error("Enter a steering instruction.")
        flags = (show_base, show_steer, show_prompt, show_both)
        outputs = tuple(o for o, on in zip(ALL_OUTPUTS, flags) if on)
        if not outputs:
            raise gr.Error("Tick at least one output.")
        r = backend.generate(prompt, steering_text, factor, int(max_new_tokens), temperature,
                             steer2, factor2, outputs=outputs)
        return r.unsteered, r.steered, r.prompted, r.both, r.info

    # Phones: stack the output panels and the two example tables in one column
    css = """
    @media (max-width: 768px) {
      .stack-mobile { flex-direction: column !important; }
      .stack-mobile > * { min-width: 100% !important; width: 100% !important; }
    }
    .out-md { min-height: 180px; }
    """
    # Ctrl/Cmd+Enter in any text box clicks Generate
    js = """() => document.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const b = document.querySelector('#generate-btn');
        if (b) { e.preventDefault(); b.click(); }
      }
    })"""
    with gr.Blocks(title="HyperSteer", css=css, js=js) as demo:
        gr.Markdown(textwrap.dedent(f"""\
            # HyperSteer
            Describe a behavior in plain English. A hypernetwork turns it into a steering
            vector added to Gemma-2-2B-it's layer-20 residual stream, with no prompt tokens spent.
            Backend: **{backend.name}**"""))
        # Current checkpoint (hot-reloaded during training): refreshed on page load and
        # every 60 s
        ckpt_md = gr.Markdown()
        ckpt_text = lambda: f"**Serving:** {getattr(backend, 'ckpt_label', backend.name)}"
        demo.load(ckpt_text, None, ckpt_md)
        if hasattr(gr, "Timer"):
            gr.Timer(60).tick(ckpt_text, None, ckpt_md)
        with gr.Row():
            with gr.Column(scale=1):
                steering = gr.Textbox(label="Steering instruction (concept)",
                                      placeholder="e.g. references to the Golden Gate Bridge")
                prompt = gr.Textbox(label="Prompt", lines=3,
                                    placeholder="e.g. How do I make a good cup of coffee?")
                factor = gr.Slider(FACTOR_MIN, FACTOR_MAX, value=FACTOR_DEFAULT, step=0.1,
                                   label="Steering strength (factor)",
                                   info="Multiplier on the vector. ~1 is the training scale; "
                                        "too high breaks fluency.")
                with gr.Accordion("Generation settings", open=False):
                    max_tokens = gr.Slider(16, 256, value=64, step=16, label="Max new tokens")
                    temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="Temperature",
                                            info="0 = greedy, same output every time")
                with gr.Accordion("Combine a second instruction (experimental)", open=False):
                    steer2 = gr.Textbox(label="Second steering instruction")
                    factor2 = gr.Slider(FACTOR_MIN, FACTOR_MAX, value=0.0, step=0.1,
                                        label="Second strength (0 = off)")
                with gr.Row():
                    show_base = gr.Checkbox(True, label="No steering")
                    show_steer = gr.Checkbox(True, label="HyperSteer")
                    show_prompt = gr.Checkbox(show_prompted, label="Instruction as prompt")
                    show_both = gr.Checkbox(show_prompted, label="Prompt + HyperSteer")
                go = gr.Button("Generate", variant="primary", elem_id="generate-btn")
            with gr.Column(scale=2):
                # Markdown panels (Gemma answers in markdown: bold, lists, headers);
                # 2x2 grid on desktop, one column on phones
                def panel(title, visible=True):
                    with gr.Column(variant="panel", min_width=240, visible=visible) as col:
                        gr.Markdown(f"**{title}**")
                        out = gr.Markdown(elem_classes="out-md")
                    return col, out

                with gr.Row(elem_classes="stack-mobile"):
                    col_base, out_base = panel("No steering")
                    col_steer, out_steer = panel("HyperSteer")
                with gr.Row(elem_classes="stack-mobile"):
                    col_prompt, out_prompt = panel("Instruction as prompt", show_prompted)
                    col_both, out_both = panel("Prompt + HyperSteer", show_prompted)
                info = gr.Markdown()
        with gr.Row(elem_classes="stack-mobile"):
            gr.Examples(HELD_IN, inputs=[steering, prompt, factor],
                        label="Held-in concepts (seen in training)")
            gr.Examples(HELD_OUT, inputs=[steering, prompt, factor],
                        label="Held-out concepts (never seen in training)")

        # Unticked outputs are skipped (no generation) and their panel hidden
        for box, col in ((show_base, col_base), (show_steer, col_steer),
                         (show_prompt, col_prompt), (show_both, col_both)):
            box.change(lambda on: gr.update(visible=on), box, col)
        inputs = [steering, prompt, factor, max_tokens, temperature, steer2, factor2,
                  show_base, show_steer, show_prompt, show_both]
        outputs = [out_base, out_steer, out_prompt, out_both, info]
        go.click(run, inputs, outputs)
        prompt.submit(run, inputs, outputs)
    return demo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["gemma", "hypersteer"], default="gemma")
    p.add_argument("--vector", choices=["zero", "actdiff"], default="zero",
                   help="gemma backend: stand-in for the hypernet's steering vector")
    p.add_argument("--run-dir", help="assets/checkpoints/train_<ts> (hypersteer backend)")
    p.add_argument("--quant", choices=["none", "8bit", "4bit"], default="none",
                   help="hypersteer backend: bitsandbytes-quantize the target model (8 GB GPUs)")
    p.add_argument("--hyper-quant", choices=["none", "8bit", "4bit"], default=None,
                   help="hypernet quantization (default: same as --quant). On 8 GB: "
                        "--quant none --hyper-quant 4bit keeps per-token generation in bf16")
    p.add_argument("--no-prompted", action="store_true",
                   help="hide the 'Instruction as prompt' panel and skip its generation (faster)")
    p.add_argument("--share", action="store_true", help="public gradio.live link (for the pod)")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--watch-ckpts", help="<run>/train/checkpoints dir: hot-load each new "
                   "checkpoint while training runs (needs --quant none)")
    p.add_argument("--total-steps", type=int, default=0, help="for the progress label")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to expose on the pod")
    args = p.parse_args()

    if args.backend == "hypersteer":
        if not args.run_dir:
            p.error("--run-dir is required for --backend hypersteer")
        backend = HyperSteerBackend(args.run_dir, quant=args.quant, hyper_quant=args.hyper_quant)
        backend.skip_prompted = args.no_prompted
        if args.watch_ckpts:
            threading.Thread(
                target=backend.watch_checkpoints,
                args=(Path(args.watch_ckpts), args.total_steps), daemon=True,
            ).start()
    else:
        backend = GemmaBackend(vector=args.vector)
    # DEMO_AUTH="user:password" in the environment adds a login page (keeps the password
    # out of the command line); recommended with --share
    auth = tuple(os.environ["DEMO_AUTH"].split(":", 1)) if os.environ.get("DEMO_AUTH") else None
    build_ui(backend, show_prompted=not args.no_prompted).launch(
        server_name=args.host, server_port=args.port, share=args.share, auth=auth)


if __name__ == "__main__":
    main()
