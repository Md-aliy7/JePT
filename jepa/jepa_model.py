"""JePTModel — JEPA self-supervised pretraining wrapper around LitePT.

Design (group-pooled, black-box LitePT)
---------------------------------------
The approved plan proposed slicing LitePT's internal voxel stages. During
implementation that proved fragile (it entangles GridPooling, sparse-conv
indice keys and re-serialization). JePT instead treats LitePT as a **black
box** and runs JEPA over *group-pooled tokens*:

  1. Points are bucketed into coarse voxel groups (`tokenizer.compute_groups`).
  2. Per sample, groups are split into contiguous context / target blocks.
  3. **Teacher** (EMA LitePT) encodes the *full* cloud; per-point features are
     pooled per group -> target token latents.
  4. **Student** (online LitePT) encodes only the *context* points; pooled per
     group -> context token latents.
  5. The **predictor** predicts target token latents from context tokens +
     target 3D positions; smooth-L1 loss in latent space.

This pretrains the entire LitePT encoder *and* decoder, and the resulting
`student` state-dict drops straight into pyLitePT's `LitePTUnifiedCustom`
backbone with zero key remapping (both are `LitePT(enc_mode=False)`).
"""

import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.litept.litept import LitePT
from models.unified import MODEL_CONFIGS

from .ema import EMA
from .losses import JEPALoss, token_std
from .predictor import JEPAPredictor
from .sampler import sample_blocks
from .tokenizer import compute_groups, scatter_mean

_ORDER = ("z", "z-trans", "hilbert", "hilbert-trans")


def build_litept(variant: str, in_channels: int) -> Tuple[LitePT, int]:
    """Construct a LitePT backbone for `variant` and return (model, feat_dim)."""
    if variant not in MODEL_CONFIGS:
        raise ValueError(f"Unknown MODEL_VARIANT '{variant}'. "
                         f"Options: {sorted(MODEL_CONFIGS)}")
    cfg = MODEL_CONFIGS[variant]
    model = LitePT(
        in_channels=in_channels,
        order=_ORDER,
        **cfg,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path=0.3,
        enc_mode=False,
    )
    feat_dim = cfg["dec_channels"][0] if len(cfg["dec_channels"]) > 0 \
        else cfg["enc_channels"][-1]
    return model, feat_dim


class JePTModel(nn.Module):
    """JEPA pretraining model: student LitePT + EMA teacher + predictor."""

    def __init__(
        self,
        variant: str = "small",
        in_channels: int = 6,
        group_size: float = 0.2,
        num_target_blocks: int = 4,
        target_ratio: Tuple[float, float] = (0.15, 0.20),
        min_tokens: int = 8,
        max_tokens: int = 2048,
        predictor_dim: int = 192,
        predictor_depth: int = 6,
        predictor_heads: int = 6,
        ema_tau_min: float = 0.9998,
        ema_tau_max: float = 0.99999,
        ema_tau_steps: int = 1,
        loss_beta: float = 2.0,
        seed: int = 0,
    ):
        super().__init__()
        self.variant = variant
        self.group_size = float(group_size)
        self.num_target_blocks = int(num_target_blocks)
        self.target_ratio = tuple(target_ratio)
        self.min_tokens = int(min_tokens)
        self.max_tokens = int(max_tokens)

        # online encoder (the backbone we keep) + frozen EMA teacher
        self.student, feat_dim = build_litept(variant, in_channels)
        self.feat_dim = feat_dim
        self.teacher = EMA(self.student, ema_tau_min, ema_tau_max, ema_tau_steps)

        self.predictor = JEPAPredictor(
            encoder_dim=feat_dim,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
        )
        self.loss_fn = JEPALoss(beta=loss_beta)
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    @staticmethod
    def _make_input(coord, feat, batch, grid_size) -> Dict:
        """Minimal LitePT input dict; Point derives grid_coord / offset."""
        return {"coord": coord, "feat": feat, "batch": batch,
                "grid_size": grid_size}

    @torch.no_grad()
    def update_teacher(self):
        """EMA step — call once after every optimiser step."""
        return self.teacher.update(self.student)

    # ------------------------------------------------------------------
    def forward(self, batch: Dict) -> Dict:
        """Run one JEPA step.

        Args:
            batch: collated dict with `coord` (N,3), `feat` (N,C), `batch` (N,),
                   `grid_size`.
        Returns:
            dict with `loss`, `pred_std`, `target_std`, `num_targets`,
            `num_context_groups`.
        """
        coord = batch["coord"]
        pbatch = batch["batch"]
        feat = batch["feat"]
        grid_size = batch["grid_size"]
        device = coord.device

        # ---- 1. group the points -------------------------------------
        group_id, group_center, group_batch = compute_groups(
            coord, pbatch, self.group_size)
        num_groups = group_center.shape[0]
        num_samples = int(pbatch.max().item()) + 1

        # ---- 2. per-sample context / target block selection ----------
        ctx_groups: List[torch.Tensor] = []
        tgt_blocks: List[List[torch.Tensor]] = []
        for b in range(num_samples):
            sample_groups = (group_batch == b).nonzero(as_tuple=True)[0]
            n = sample_groups.shape[0]
            if n < self.min_tokens:
                continue
            if n > self.max_tokens:                       # bound predictor cost
                keep = torch.randperm(n, device=device)[:self.max_tokens]
                sample_groups = sample_groups[keep]
                n = self.max_tokens
            centers = group_center[sample_groups]
            ctx_local, tgt_local = sample_blocks(
                centers, self.num_target_blocks, self.target_ratio, self._rng)
            if len(ctx_local) == 0:
                continue
            ctx_g = sample_groups[torch.tensor(ctx_local, device=device,
                                               dtype=torch.long)]
            tgt_g = [sample_groups[torch.tensor(t, device=device,
                                                dtype=torch.long)]
                     for t in tgt_local if len(t) > 0]
            if len(tgt_g) == 0:
                continue
            ctx_groups.append(ctx_g)
            tgt_blocks.append(tgt_g)

        # point-level mask of which points belong to a context group
        ctx_group_mask = torch.zeros(num_groups, dtype=torch.bool, device=device)
        for ctx_g in ctx_groups:
            ctx_group_mask[ctx_g] = True
        ctx_point_idx = ctx_group_mask[group_id].nonzero(as_tuple=True)[0]

        # degenerate batch (e.g. all clouds smaller than min_tokens):
        # return a zero loss that still carries grad so training does not crash
        if ctx_point_idx.numel() == 0 or len(ctx_groups) == 0:
            zero = sum(p.sum() for p in self.predictor.parameters()) * 0.0
            return {"loss": zero, "pred_std": 0.0, "target_std": 0.0,
                    "num_targets": 0, "num_context_groups": 0}

        # ---- 3. teacher encodes the FULL cloud -----------------------
        self.teacher.ema_model.eval()                     # no DropPath on targets
        with torch.no_grad():
            teacher_pt = self.teacher.ema_model(
                self._make_input(coord, feat, pbatch, grid_size))
            teacher_feat = teacher_pt.feat
        teacher_grp = scatter_mean(teacher_feat, group_id, num_groups)
        teacher_grp = F.layer_norm(teacher_grp, (teacher_grp.shape[-1],))

        # ---- 4. student encodes the CONTEXT subset -------------------
        sub_batch_raw = pbatch[ctx_point_idx]
        _, sub_batch = torch.unique(sub_batch_raw, return_inverse=True)
        student_pt = self.student(self._make_input(
            coord[ctx_point_idx], feat[ctx_point_idx], sub_batch, grid_size))
        student_grp = scatter_mean(
            student_pt.feat, group_id[ctx_point_idx], num_groups)

        # ---- 5. predict target latents from context ------------------
        preds, targets = [], []
        for ctx_g, blocks in zip(ctx_groups, tgt_blocks):
            ctx_tok = student_grp[ctx_g]
            ctx_ctr = group_center[ctx_g]
            for tgt_g in blocks:
                pred = self.predictor(ctx_tok, ctx_ctr, group_center[tgt_g])
                preds.append(pred)
                targets.append(teacher_grp[tgt_g])

        pred_all = torch.cat(preds, dim=0)
        tgt_all = torch.cat(targets, dim=0)
        loss = self.loss_fn(pred_all, tgt_all)

        return {
            "loss": loss,
            "pred_std": token_std(pred_all),
            "target_std": token_std(tgt_all),
            "num_targets": int(tgt_all.shape[0]),
            "num_context_groups": int(sum(c.numel() for c in ctx_groups)),
        }
