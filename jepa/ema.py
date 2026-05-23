"""Exponential-moving-average teacher.

Adapted from Point-JEPA's `pointjepa/modules/EMA.py` — plain PyTorch, framework
agnostic. The teacher is a frozen deep copy of the online (student) model whose
parameters track the student with a momentum `tau` that is ramped from
`tau_min` to `tau_max` over `tau_steps` optimiser steps (I-JEPA / Point-JEPA
recipe).
"""

import copy

import torch
import torch.nn as nn


class EMA(nn.Module):
    """Frozen EMA copy of a model.

    Args:
        model:     the online model to mirror.
        tau_min:   initial momentum (slow teacher, stable early targets).
        tau_max:   final momentum (teacher nearly frozen late in training).
        tau_steps: number of optimiser steps over which tau ramps min->max.
    """

    def __init__(self, model: nn.Module, tau_min: float = 0.9998,
                 tau_max: float = 0.99999, tau_steps: int = 1):
        super().__init__()
        self.ema_model = copy.deepcopy(model)
        self.ema_model.requires_grad_(False)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.tau_steps = max(int(tau_steps), 1)
        # buffer so the step counter is saved/restored with checkpoints
        self.register_buffer("step", torch.zeros(1, dtype=torch.long))

    def get_current_decay(self) -> float:
        s = int(self.step.item())
        if s >= self.tau_steps:
            return self.tau_max
        return self.tau_min + (self.tau_max - self.tau_min) * s / self.tau_steps

    @torch.no_grad()
    def update(self, online_model: nn.Module) -> float:
        """Pull EMA parameters toward `online_model`. Call after optimizer.step()."""
        tau = self.get_current_decay()
        for ema_p, online_p in zip(self.ema_model.parameters(),
                                   online_model.parameters()):
            ema_p.mul_(tau).add_(online_p.detach(), alpha=1.0 - tau)
        # buffers (e.g. BatchNorm running stats) are copied verbatim
        for ema_b, online_b in zip(self.ema_model.buffers(),
                                   online_model.buffers()):
            ema_b.copy_(online_b)
        self.step += 1
        return tau

    def forward(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)
