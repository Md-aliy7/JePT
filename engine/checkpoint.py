"""Checkpoint save / load for JePT."""

import os

import torch


def save_pretrain_checkpoint(path, jepa_model, optimizer, scheduler, epoch, cfg):
    """Persist a JEPA pretraining checkpoint.

    The `student` state-dict is the backbone consumed downstream by
    `models.unified.load_pretrained_backbone`.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "student": jepa_model.student.state_dict(),
        "predictor": jepa_model.predictor.state_dict(),
        "teacher": jepa_model.teacher.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "variant": jepa_model.variant,
        "feat_dim": jepa_model.feat_dim,
        # pretraining context — downstream transfer is only valid when the
        # fine-tuning config matches these (same backbone input statistics)
        "grid_size": getattr(cfg, "GRID_SIZE", None),
        "input_channels": getattr(cfg, "INPUT_CHANNELS", None),
    }, path)


def save_finetune_checkpoint(path, model, optimizer, epoch, metrics, meta=None):
    """Persist a downstream fine-tuning checkpoint.

    `meta` (variant / input_channels / class counts) is stored so tools such as
    `Custom/visualize.py` can rebuild the exact model without relying on the
    config still matching.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "metrics": metrics,
        "meta": meta or {},
    }, path)


def load_checkpoint(path, map_location="cpu"):
    """Load any checkpoint file."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location=map_location)
