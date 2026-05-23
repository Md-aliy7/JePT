"""Shared helpers for the JePT training engine."""

import math

import torch


def to_device(batch: dict, device) -> dict:
    """Move every tensor in a collated batch dict to `device` (lists kept)."""
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def weight_decay_groups(named_params, weight_decay: float):
    """Split parameters into AdamW groups with weight decay applied correctly.

    1-D parameters — LayerNorm / BatchNorm weights and all biases — are placed
    in a no-decay group (`weight_decay=0`). Decaying them is harmful and is
    excluded by every transformer SSL recipe (I-JEPA, DINO, Point-JEPA).

    Returns a list of param-group dicts ready for `torch.optim.AdamW`. Groups
    with no parameters are dropped so the optimiser never sees an empty group.
    """
    decay, no_decay = [], []
    for _, p in named_params:
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int,
                            eta_min_ratio: float = 1e-3):
    """LambdaLR: linear warm-up then cosine decay to `eta_min_ratio * base_lr`.

    Plain-PyTorch replacement for Point-JEPA's `pl_bolts` scheduler — keeps the
    project dependency-free of PyTorch Lightning.
    """
    warmup_steps = max(int(warmup_steps), 1)
    total_steps = max(int(total_steps), warmup_steps + 1)

    def fn(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)
