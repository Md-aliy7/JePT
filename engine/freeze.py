"""Backbone freeze-policy controller for downstream fine-tuning.

Research summary (why three modes are kept)
--------------------------------------------
The self-supervised literature does not pick a single freeze policy — the right
choice depends on the label budget and task:

* **Linear probe (frozen backbone)** — DINOv3 explicitly freezes the backbone
  for dense tasks (segmentation, depth) and trains only lightweight heads. It
  is the standard *diagnostic* of representation quality and is the safest
  choice when very few labels are available (least overfitting).
* **Full fine-tune (unfrozen, low LR)** — generally the best *final* accuracy,
  and usually necessary for 3D detection, which transfers less readily than
  classification. Risk: overfitting a large backbone on a tiny labeled set.
* **Staged unfreeze** — Point-JEPA freezes the encoder for the first N epochs
  then unfreezes it. A stable compromise: heads adapt first, then the backbone
  is gently tuned.

Conclusion: keep all three, switchable via `FREEZE_MODE`; default to
`staged_unfreeze`. This controller owns optimiser (re)construction because the
set of trainable parameters changes when the backbone is (un)frozen.
"""

import torch

from .common import weight_decay_groups

_BACKBONE_PREFIXES = ("backbone.", "seg_backbone.", "det_backbone.")
_BACKBONE_NAMES = ("backbone", "seg_backbone", "det_backbone")


class FreezeController:
    """Applies a freeze policy each epoch and (re)builds the optimiser.

    Usage (per epoch, after `model.train()`):
        optimizer = controller.apply(epoch)
    """

    def __init__(self, model, cfg):
        self.model = model
        self.mode = getattr(cfg, "FREEZE_MODE", "staged_unfreeze")
        self.freeze_epochs = int(getattr(cfg, "FREEZE_EPOCHS", 20))
        self.base_lr = float(getattr(cfg, "LEARNING_RATE", 1e-3))
        self.backbone_lr_scale = float(getattr(cfg, "BACKBONE_LR_SCALE", 0.1))
        self.weight_decay = float(getattr(cfg, "WEIGHT_DECAY", 0.01))
        assert self.mode in ("linear_probe", "full_finetune", "staged_unfreeze"), \
            f"Invalid FREEZE_MODE '{self.mode}'"
        self._frozen = None          # current backbone state (None = uninitialised)
        self._optimizer = None

    # ------------------------------------------------------------------
    @staticmethod
    def _is_backbone(param_name: str) -> bool:
        return param_name.startswith(_BACKBONE_PREFIXES)

    def _set_frozen(self, frozen: bool):
        """Set requires_grad and train/eval mode of every backbone."""
        for name, param in self.model.named_parameters():
            if self._is_backbone(name):
                param.requires_grad = not frozen
        # frozen backbone -> .eval() so BatchNorm running stats stop updating
        for name, module in self.model.named_children():
            if name in _BACKBONE_NAMES:
                module.eval() if frozen else module.train()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """AdamW: reduced LR for the backbone, weight decay off for 1-D params.

        Four groups result — {backbone, head} x {decay, no-decay} — so a frozen
        backbone simply contributes no groups, and LayerNorm/bias params are
        never weight-decayed (transformer SSL / fine-tuning best practice).
        """
        groups = []
        for is_backbone, lr in ((True, self.base_lr * self.backbone_lr_scale),
                                (False, self.base_lr)):
            named = [(n, p) for n, p in self.model.named_parameters()
                     if p.requires_grad and self._is_backbone(n) == is_backbone]
            for g in weight_decay_groups(named, self.weight_decay):
                g["lr"] = lr
                groups.append(g)
        return torch.optim.AdamW(groups, lr=self.base_lr)

    # ------------------------------------------------------------------
    def want_frozen(self, epoch: int) -> bool:
        if self.mode == "linear_probe":
            return True
        if self.mode == "full_finetune":
            return False
        return epoch < self.freeze_epochs       # staged_unfreeze

    def apply(self, epoch: int) -> torch.optim.Optimizer:
        """Enforce the policy for `epoch` and return the optimiser to use.

        Re-asserts the backbone train/eval mode every call (a preceding
        `model.train()` would otherwise un-freeze BatchNorm). Rebuilds the
        optimiser only when the frozen/unfrozen state actually changes.
        """
        frozen = self.want_frozen(epoch)
        changed = frozen != self._frozen
        self._set_frozen(frozen)                # always re-assert
        self._frozen = frozen
        if changed or self._optimizer is None:
            self._optimizer = self._build_optimizer()
            state = "FROZEN" if frozen else "TRAINABLE"
            print(f"[FreezeController] epoch {epoch}: backbone {state} "
                  f"(mode={self.mode}) — optimiser rebuilt")
        return self._optimizer
