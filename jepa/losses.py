"""JEPA latent-prediction loss and representation-collapse monitors."""

import torch
import torch.nn as nn


class JEPALoss(nn.Module):
    """Smooth-L1 (Huber) loss between predicted and target latent features.

    JEPA predicts in *latent* space (no pixel/point reconstruction). Smooth-L1
    (Point-JEPA default, `beta=2`) is robust to outliers and avoids forcing the
    model to match low-level detail.
    """

    def __init__(self, beta: float = 2.0):
        super().__init__()
        self.fn = nn.SmoothL1Loss(beta=beta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.fn(pred, target)


@torch.no_grad()
def token_std(x: torch.Tensor) -> float:
    """Mean per-channel std of a token batch — a representation-collapse probe.

    If this collapses toward 0 the encoder is producing constant features.
    Healthy target std stays well above 0; predictions should not vanish.
    """
    if x.numel() == 0:
        return 0.0
    return float(x.detach().float().std(dim=0).mean())
