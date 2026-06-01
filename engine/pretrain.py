"""Stage 1 — JEPA self-supervised pretraining loop (plain PyTorch).

Trains the LitePT backbone with no annotations. Produces checkpoints whose
`student` state-dict is loaded downstream by `load_pretrained_backbone`.
"""

import os

import torch
from torch.utils.data import DataLoader

from data import UnlabeledPointDataset, collate_point_batch
from jepa import JePTModel

from .checkpoint import save_pretrain_checkpoint
from .common import set_seed, to_device, warmup_cosine_scheduler, weight_decay_groups


def calibrate_group_size(dataset, n_sample=8, tokens_per_axis=12):
    """Pick a JEPA group (token) size from the dataset's spatial extent.

    Samples a few clouds, takes the median of their largest bounding-box
    edge, and divides by `tokens_per_axis`. This yields a comparable token
    count whether the data is small objects or large scenes.
    """
    import numpy as np
    n = len(dataset)
    idxs = np.unique(np.linspace(0, n - 1, min(n_sample, n)).astype(int))
    extents = []
    for i in idxs:
        coord = dataset[int(i)]["coord"]
        coord = coord.numpy() if hasattr(coord, "numpy") else np.asarray(coord)
        if coord.shape[0] > 0:
            extents.append(float((coord.max(0) - coord.min(0)).max()))
    if not extents:
        return 0.2
    return max(float(np.median(extents)) / tokens_per_axis, 1e-3)


def pretrain(cfg, max_steps_per_epoch=None, verbose=True):
    """Run JEPA pretraining.

    Args:
        cfg:                 pretraining config module (see configs/pretrain_config.py).
        max_steps_per_epoch: cap steps/epoch (used by the smoke test).
        verbose:             print per-epoch logs.
    Returns:
        dict history with per-epoch `loss`, `pred_std`, `target_std`.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(getattr(cfg, "USE_AMP", True)) and device.type == "cuda"
    set_seed(getattr(cfg, "SEED", 0))

    dataset = UnlabeledPointDataset(cfg.UNLABELED_DATA_PATH, cfg, split="train")
    if len(dataset) == 0:
        raise RuntimeError(f"No unlabeled clouds found in {cfg.UNLABELED_DATA_PATH}")
    loader = DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        collate_fn=collate_point_batch, num_workers=cfg.NUM_WORKERS,
        drop_last=len(dataset) >= cfg.BATCH_SIZE)

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * cfg.EPOCHS

    # GROUP_SIZE drives the JEPA token scale and is the one data-dependent
    # knob: auto-calibrate it from the data extent unless an explicit value
    # is given, so the same config works for objects and large scenes alike.
    group_size = getattr(cfg, "GROUP_SIZE", "auto")
    if group_size in (None, "auto", 0, 0.0):
        group_size = calibrate_group_size(dataset)
        print(f"[pretrain] auto-calibrated GROUP_SIZE = {group_size:.4f}")

    model = JePTModel(
        variant=cfg.MODEL_VARIANT,
        in_channels=cfg.INPUT_CHANNELS,
        group_size=group_size,
        num_target_blocks=cfg.NUM_TARGET_BLOCKS,
        target_ratio=tuple(cfg.TARGET_RATIO),
        min_tokens=cfg.MIN_TOKENS,
        max_tokens=cfg.MAX_TOKENS,
        predictor_dim=cfg.PREDICTOR_DIM,
        predictor_depth=cfg.PREDICTOR_DEPTH,
        predictor_heads=cfg.PREDICTOR_HEADS,
        ema_tau_min=cfg.EMA_TAU_MIN,
        ema_tau_max=cfg.EMA_TAU_MAX,
        ema_tau_steps=max(steps_per_epoch * cfg.EMA_TAU_EPOCHS, 1),
        loss_beta=cfg.LOSS_BETA,
        seed=cfg.SEED,
    ).to(device)

    # optimise the student + predictor; the EMA teacher is never optimised.
    # `params` (flat) is kept for grad clipping; the optimiser uses weight-decay
    # groups so LayerNorm/bias (1-D) params are excluded from weight decay.
    params = list(model.student.parameters()) + list(model.predictor.parameters())
    named = (list(model.student.named_parameters())
             + list(model.predictor.named_parameters()))
    optimizer = torch.optim.AdamW(
        weight_decay_groups(named, cfg.WEIGHT_DECAY), lr=cfg.LR)
    scheduler = warmup_cosine_scheduler(
        optimizer, warmup_steps=steps_per_epoch * cfg.WARMUP_EPOCHS,
        total_steps=total_steps, eta_min_ratio=cfg.ETA_MIN / cfg.LR)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {"loss": [], "pred_std": [], "target_std": []}
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    # optional resume from the rolling checkpoint
    start_epoch = 0
    resume_path = os.path.join(cfg.RESULTS_DIR, "last.pth")
    if getattr(cfg, "RESUME", False) and os.path.exists(resume_path):
        try:
            ck = torch.load(resume_path, map_location=device)
            model.student.load_state_dict(ck["student"])
            model.predictor.load_state_dict(ck["predictor"])
            model.teacher.load_state_dict(ck["teacher"])
            optimizer.load_state_dict(ck["optimizer"])
            if ck.get("scheduler") is not None:
                scheduler.load_state_dict(ck["scheduler"])
            start_epoch = int(ck["epoch"])
            print(f"[pretrain] resumed from {resume_path} at epoch {start_epoch}")
        except Exception as exc:                       # noqa: BLE001
            print(f"[pretrain] resume failed ({exc}) — starting from scratch")
            start_epoch = 0

    for epoch in range(start_epoch, cfg.EPOCHS):
        model.train()
        ep_loss = ep_pstd = ep_tstd = 0.0
        n = 0
        for step, batch in enumerate(loader):
            if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                break
            batch = to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(batch)
                loss = out["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            model.update_teacher()           # EMA step (after optimizer.step())

            ep_loss += float(loss.detach())
            ep_pstd += out["pred_std"]
            ep_tstd += out["target_std"]
            n += 1

        n = max(n, 1)
        history["loss"].append(ep_loss / n)
        history["pred_std"].append(ep_pstd / n)
        history["target_std"].append(ep_tstd / n)
        if verbose:
            print(f"[pretrain] epoch {epoch + 1}/{cfg.EPOCHS} "
                  f"loss={ep_loss / n:.4f} "
                  f"pred_std={ep_pstd / n:.4f} target_std={ep_tstd / n:.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        # periodic + rolling checkpoints
        if (epoch + 1) % getattr(cfg, "SAVE_EVERY", 10) == 0:
            save_pretrain_checkpoint(
                os.path.join(cfg.RESULTS_DIR, f"epoch_{epoch + 1}.pth"),
                model, optimizer, scheduler, epoch + 1, cfg)
        save_pretrain_checkpoint(
            os.path.join(cfg.RESULTS_DIR, "last.pth"),
            model, optimizer, scheduler, epoch + 1, cfg)

    return history
