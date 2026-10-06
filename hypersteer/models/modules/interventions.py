from dataclasses import dataclass
from typing import Any

import torch
from pyvene import (
    DistributedRepresentationIntervention,
    InterventionOutput,
    SourcelessIntervention,
    TrainableIntervention,
)
from torch import nn
from torch.nn import functional as F


@dataclass
class PayloadInterventionOutput(InterventionOutput):
    """
    Output of the IntervenableModel, including original outputs, intervened outputs, and collected activations.
    """

    output: Any | None = None
    latent: Any | None = None
    payload: Any | None = None


class SelectionHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        use_ln: bool = True,
        start_temperature: float = 1.0,
        end_temperature: float = 0.1,
        learnable_temperature: bool = False,
        add_gumbel_noise: bool = False,
        threshold: float = 0.5,
        straight_through: bool = True,
    ):
        super().__init__()
        self.proj = nn.Linear(hidden_size * 2, 1)
        self.ln = None
        if use_ln:
            self.ln = nn.LayerNorm((hidden_size * 2,))

        self.start_temperature = start_temperature
        self.end_temperature = end_temperature
        self.learnable_temperature = learnable_temperature
        self.add_gumbel_noise = add_gumbel_noise
        self.register_buffer(
            "_step", torch.tensor(0, dtype=torch.int32, requires_grad=False)
        )
        self._temperature = nn.Parameter(
            torch.tensor(start_temperature, requires_grad=self.learnable_temperature)
        )
        self.threshold = threshold
        self.straight_through = straight_through

    def get_temperature(self) -> torch.Tensor:
        return self._temperature

    @torch.no_grad()
    def step_temperature(self, total_steps: int):
        """
        Linearly anneals the temperature from start_temperature to end_temperature over total_steps.
        Should be called at each training step.
        """
        self._step.add_(1)
        step = min(self._step.item(), total_steps)
        new_temp = self.start_temperature + (
            self.end_temperature - self.start_temperature
        ) * (step / total_steps)
        self._temperature.fill_(new_temp)

    def forward(self, x, v, hard_mask=False, eps=1e-6):
        latent = torch.cat([x, v.unsqueeze(1).expand_as(x)], dim=-1)
        if self.ln:
            latent = self.ln(latent)

        with torch.no_grad():
            x_stats = {
                "min": x.min().item(),
                "max": x.max().item(),
                "mean": x.mean().item(),
            }
            v_stats = {
                "min": v.min().item(),
                "max": v.max().item(),
                "mean": v.mean().item(),
            }
            print(f"x stats: {x_stats}")
            print(f"v stats: {v_stats}")

        _temperature = self._temperature.clip(
            min=self.end_temperature - eps, max=self.start_temperature + eps
        )

        if not self.learnable_temperature:
            _temperature = _temperature.detach()

        logits = self.proj(latent)

        if self.add_gumbel_noise:
            # draw uniform noise so that 0 < noise < 1 and log() is always defined
            noise = torch.rand_like(logits).clamp(min=eps, max=1 - eps)
            logistic_noise = torch.log(noise) - torch.log(1 - noise)

            out = F.sigmoid((logits + logistic_noise) / _temperature)
        else:
            out = F.sigmoid(logits / _temperature)

        if self.straight_through:
            out = (out > self.threshold).to(out.dtype) + out - out.detach()
        elif hard_mask:
            out = (out > self.threshold).to(out.dtype)
        return out


class HyperAdditiveIntervention(
    SourcelessIntervention, TrainableIntervention, DistributedRepresentationIntervention
):
    def __init__(self, **kwargs):
        # Note that we initialise these to zeros because we're loading in pre-trained weights.
        # If you want to train your own SAEs then we recommend using blah
        super().__init__(**kwargs, keep_last_dim=True)
        self.low_rank_dimension = kwargs["low_rank_dimension"]
        self.v: torch.Tensor = None
        self.use_selection = kwargs.get("use_selection_head", False)

        if self.use_selection:
            self.selection_head = SelectionHead(
                self.embed_dim,
                use_ln=kwargs.get("use_ln", True),
                start_temperature=kwargs.get("selection_head_start_temperature", 1.0),
                end_temperature=kwargs.get("selection_head_end_temperature", 0.1),
                learnable_temperature=kwargs.get(
                    "selection_head_learnable_temperature", False
                ),
                add_gumbel_noise=kwargs.get("selection_head_add_gumbel_noise", False),
                threshold=kwargs.get("selection_head_threshold", 0.5),
                straight_through=kwargs.get("selection_head_straight_through", False),
            )

    def _update_v(self, new_vect: torch.Tensor):
        self.v = new_vect

    def _reset_v(self):
        self.v = None

    def forward(self, base, source=None, subspaces=None):
        mag = subspaces["mag"][:, None, None] if subspaces and "mag" in subspaces else 1
        threshold = subspaces.get("inference_binarize_mask", False)
        mask = (
            self.selection_head(base, self.v, hard_mask=threshold)
            if self.use_selection
            else 1
        )
        output = base + mask * mag * self.v.unsqueeze(dim=1)

        return PayloadInterventionOutput(
            output=output,
            payload={
                "mask": mask,
            },
        )
