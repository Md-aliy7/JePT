"""Stage 2 — downstream segmentation + detection fine-tuning (plain PyTorch).

Loads a JEPA-pretrained backbone (via `create_unified_model` /
`load_pretrained_backbone`), attaches pyLitePT's segmentation and detection
heads, and fine-tunes under one of three freeze policies (see engine/freeze.py).
"""

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import CustomDataset, collate_point_batch
from models.unified import create_unified_model

from .checkpoint import save_finetune_checkpoint
from .common import set_seed, to_device
from .freeze import FreezeController


def _check_label_ranges(ds, cfg, split):
    """Scan a labelled split and fail clearly on out-of-range labels.

    Without this, a `segment` value >= NUM_CLASSES_SEG or a `gt_boxes` class id
    >= NUM_CLASSES_DET crashes deep inside CrossEntropy / one-hot scatter with a
    cryptic index error. This makes JePT generic for any prepared 3D dataset:
    the failure names the offending scene and the fix.
    """
    import numpy as np

    ignore = set(getattr(cfg, "IGNORED_LABELS", [-1]) or [-1])
    n_seg = getattr(cfg, "NUM_CLASSES_SEG", 0)
    n_det = getattr(cfg, "NUM_CLASSES_DET", 0)
    for scene in ds.scenes:
        if os.path.isdir(scene):                    # NPY-folder scene
            seg_p = os.path.join(scene, "segment.npy")
            box_p = os.path.join(scene, "gt_boxes.npy")
        else:                                       # .ply scene + sidecar
            seg_p = None
            box_p = scene.replace(".ply", "_gt_boxes.npy")

        if n_seg > 0 and seg_p and os.path.exists(seg_p):
            s = np.load(seg_p).reshape(-1)
            bad = s[(s >= n_seg) | ((s < 0) & ~np.isin(s, list(ignore)))]
            if bad.size:
                raise ValueError(
                    f"[{split}] {scene}: segment label {int(bad.max())} is "
                    f"outside [0,{n_seg}). Set NUM_CLASSES_SEG to cover every "
                    f"label, or fix the data. (Allowed ignore labels: {sorted(ignore)})")

        if n_det > 0 and os.path.exists(box_p):
            b = np.load(box_p)
            if b.ndim == 2 and b.shape[0] > 0 and b.shape[1] > 7:
                lbl = b[:, 7].astype(np.int64)
                if lbl.min() < 0 or lbl.max() >= n_det:
                    raise ValueError(
                        f"[{split}] {scene}: gt_boxes class id "
                        f"{int(lbl.max())} is outside [0,{n_det}). Set "
                        f"NUM_CLASSES_DET correctly or fix the data "
                        f"(class ids must be 0-based — labelCloud exports them so).")


def _lr_factor(epoch, total_epochs, warmup):
    """Linear warm-up then cosine decay to 5% of base LR — a multiplicative
    factor applied to every optimiser param group each epoch.

    Stateless (a pure function of the epoch), so it survives the optimiser
    being rebuilt mid-training when the FreezeController (un)freezes the
    backbone — a stateful torch scheduler would not.
    """
    import math
    if warmup > 0 and epoch < warmup:
        return (epoch + 1) / (warmup + 1)
    progress = (epoch - warmup) / max(total_epochs - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))


def _build_det_config(cfg, train_ds):
    """Assemble the detection-head config, resolving 'auto' MEAN_SIZE."""
    det_config = dict(getattr(cfg, "DETECTION_CONFIG", {}) or {})
    if cfg.NUM_CLASSES_DET > 0 and det_config.get("MEAN_SIZE", "auto") == "auto":
        det_config["MEAN_SIZE"] = train_ds.calculate_mean_sizes(cfg.NUM_CLASSES_DET)
    return det_config


def _seg_loss_fn(cfg, train_ds, device):
    """CrossEntropy with (optionally auto) class weights and ignore index."""
    weights = cfg.CLASS_WEIGHTS
    if weights == "auto":
        weights = train_ds.calculate_class_weights(cfg.NUM_CLASSES_SEG)
    weight = torch.tensor(weights, dtype=torch.float32, device=device)
    ignore = cfg.IGNORED_LABELS[0] if getattr(cfg, "IGNORED_LABELS", None) else -1
    return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore)


def _combine_losses(cfg, model, seg_loss, det_loss):
    """Fuse seg + det losses (uncertainty weighting or static)."""
    if seg_loss is not None and det_loss is not None:
        if getattr(cfg, "LOSS_BALANCING_METHOD", "uncertainty") == "uncertainty":
            lv = model.log_vars                         # [log_var_seg, log_var_det]
            return (torch.exp(-lv[0]) * seg_loss + 0.5 * lv[0]
                    + torch.exp(-lv[1]) * det_loss + 0.5 * lv[1])
        return seg_loss + getattr(cfg, "DETECTION_LOSS_WEIGHT", 1.0) * det_loss
    return seg_loss if seg_loss is not None else det_loss


def finetune(cfg, max_steps_per_epoch=None, verbose=True, max_train_scenes=None):
    """Run downstream fine-tuning.

    Args:
        cfg:                finetune config module.
        max_steps_per_epoch: cap steps/epoch (smoke test).
        verbose:            print per-epoch logs.
        max_train_scenes:   if set, use only the first N labeled train scenes —
                            simulates the low-label regime JePT targets.
    Returns:
        (model, history) — `history` has per-epoch `train_seg_loss`,
        `train_det_loss`, `train_total`, `val_seg_acc`, `val_det_mAP50`.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(getattr(cfg, "USE_AMP", False)) and device.type == "cuda"
    fmt = getattr(cfg, "DATA_FORMAT", "npy")
    set_seed(getattr(cfg, "SEED", 0))

    train_ds = CustomDataset(cfg.DATA_PATH, "train", cfg, fmt)
    if len(train_ds.scenes) == 0:
        raise RuntimeError(
            f"No labeled train scenes found in {cfg.DATA_PATH}/train. "
            f"Expected NPY scene folders (coord.npy + segment.npy) or .ply files.")
    # fail early + clearly on a data/config feature-channel mismatch
    if train_ds.input_channels < cfg.INPUT_CHANNELS:
        raise RuntimeError(
            f"data provides {train_ds.input_channels} feature channels but "
            f"INPUT_CHANNELS={cfg.INPUT_CHANNELS}. Set INPUT_CHANNELS="
            f"{train_ds.input_channels} in finetune_config.py.")
    if max_train_scenes is not None and len(train_ds.scenes) > max_train_scenes:
        train_ds.scenes = train_ds.scenes[:max_train_scenes]
        print(f"[finetune] low-label regime: using "
              f"{len(train_ds.scenes)} labeled train scenes")
    val_ds = CustomDataset(cfg.DATA_PATH, "val", cfg, fmt)

    # generic-data safety: fail clearly (not cryptically) on label/config mismatch
    _check_label_ranges(train_ds, cfg, "train")
    _check_label_ranges(val_ds, cfg, "val")
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              collate_fn=collate_point_batch,
                              num_workers=cfg.NUM_WORKERS,
                              drop_last=len(train_ds) >= cfg.BATCH_SIZE)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            collate_fn=collate_point_batch,
                            num_workers=cfg.NUM_WORKERS)

    det_config = _build_det_config(cfg, train_ds)
    model = create_unified_model(cfg, cfg.INPUT_CHANNELS, det_config, device)

    has_seg = cfg.NUM_CLASSES_SEG > 0 and model.seg_head is not None
    has_det = cfg.NUM_CLASSES_DET > 0 and model.det_head is not None
    seg_loss_fn = _seg_loss_fn(cfg, train_ds, device) if has_seg else None

    controller = FreezeController(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = {"train_seg_loss": [], "train_det_loss": [],
               "train_total": [], "val_seg_acc": [], "val_det_mAP50": []}
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    best_metric = -1.0

    # optional resume (model weights + epoch; the freeze controller rebuilds
    # the optimiser, so optimiser momentum is intentionally not restored)
    start_epoch = 0
    resume_path = os.path.join(cfg.RESULTS_DIR, "last.pth")
    if getattr(cfg, "RESUME", False) and os.path.exists(resume_path):
        try:
            ck = torch.load(resume_path, map_location=device)
            model.load_state_dict(ck["model_state_dict"])
            start_epoch = int(ck["epoch"])
            _m = ck.get("metrics", {})
            best_metric = float(_m.get("selection_metric",
                                       _m.get("val_seg_acc", -1.0)))
            print(f"[finetune] resumed from {resume_path} at epoch {start_epoch}")
        except Exception as exc:                       # noqa: BLE001
            print(f"[finetune] resume failed ({exc}) — starting from scratch")
            start_epoch = 0

    # LR schedule (warm-up + cosine). pyLitePT fine-tunes with a scheduler;
    # JePT applies it as a stateless per-epoch factor so it survives the
    # FreezeController rebuilding the optimiser at the unfreeze epoch.
    warmup_epochs = max(1, cfg.EPOCHS // 20)
    last_optimizer = None
    base_lrs = []

    for epoch in range(start_epoch, cfg.EPOCHS):
        model.train()
        optimizer = controller.apply(epoch)         # enforce freeze policy

        # capture each param group's base LR whenever the optimiser is rebuilt,
        # then scale all groups by the schedule factor for this epoch
        if optimizer is not last_optimizer:
            last_optimizer = optimizer
            base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        lr_scale = _lr_factor(epoch, cfg.EPOCHS, warmup_epochs)
        for pg, base in zip(optimizer.param_groups, base_lrs):
            pg["lr"] = base * lr_scale

        ep_seg = ep_det = ep_total = 0.0
        n = 0
        for step, batch in enumerate(train_loader):
            if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                break
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(batch)
                seg_loss = None
                if has_seg:
                    seg_loss = seg_loss_fn(outputs["seg_logits"],
                                           batch["segment"].long())
                det_loss = None
                if has_det:
                    det_loss, _ = model.det_head.get_loss()
                total = _combine_losses(cfg, model, seg_loss, det_loss)

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                getattr(cfg, "GRAD_CLIP_NORM", 1.0))
            scaler.step(optimizer)
            scaler.update()

            ep_seg += float(seg_loss.detach()) if seg_loss is not None else 0.0
            ep_det += float(det_loss.detach()) if det_loss is not None else 0.0
            ep_total += float(total.detach())
            n += 1

        n = max(n, 1)
        history["train_seg_loss"].append(ep_seg / n)
        history["train_det_loss"].append(ep_det / n)
        history["train_total"].append(ep_total / n)

        # Fused single-pass val: seg accuracy + 3D-IoU mAP@0.5 in one loop.
        # mAP@0.5 is the pyLitePT-compatible detection metric — `best.pth` is
        # NOT chosen on seg accuracy alone (seg can saturate long before
        # detection converges).
        val_acc, val_det_mAP50 = _validate_combined(
            model, val_loader, device, has_seg, has_det, cfg)
        history["val_seg_acc"].append(val_acc)
        history["val_det_mAP50"].append(val_det_mAP50)

        if verbose:
            print(f"[finetune] epoch {epoch + 1}/{cfg.EPOCHS} "
                  f"seg_loss={ep_seg / n:.4f} det_loss={ep_det / n:.4f} "
                  f"total={ep_total / n:.4f} val_seg_acc={val_acc:.4f} "
                  f"val_det_mAP50={val_det_mAP50:.4f}")

        meta = {
            "variant": cfg.MODEL_VARIANT,
            "input_channels": cfg.INPUT_CHANNELS,
            "num_classes_seg": cfg.NUM_CLASSES_SEG,
            "num_classes_det": cfg.NUM_CLASSES_DET,
            "use_dual_path": getattr(cfg, "USE_DUAL_PATH_UNIFIED", False),
            "class_names": list(getattr(cfg, "CLASS_NAMES", [])),
            "grid_size": getattr(cfg, "GRID_SIZE", 0.02),
            "data_path": os.path.abspath(cfg.DATA_PATH),
        }
        # combined selection metric: seg accuracy + detection mAP@0.5
        # (both in [0,1]). Falls back to seg-only / det-only / loss when a
        # task is off. mAP@0.5 matches pyLitePT's evaluate.py unified score.
        if has_seg and has_det:
            metric = val_acc + val_det_mAP50
        elif has_seg:
            metric = val_acc
        elif has_det:
            metric = val_det_mAP50
        else:
            metric = -(ep_total / n)
        metrics = {"val_seg_acc": val_acc, "val_det_mAP50": val_det_mAP50,
                   "selection_metric": metric}
        if metric > best_metric:
            best_metric = metric
            save_finetune_checkpoint(
                os.path.join(cfg.RESULTS_DIR, "best.pth"),
                model, optimizer, epoch + 1, metrics, meta)
        save_finetune_checkpoint(
            os.path.join(cfg.RESULTS_DIR, "last.pth"),
            model, optimizer, epoch + 1, metrics, meta)

    return model, history


@torch.no_grad()
def _validate_combined(model, val_loader, device, has_seg, has_det, cfg):
    """One-pass val: seg point accuracy + detection mAP@0.5.

    Fuses the two metrics into a single sweep over the val loader so each
    epoch's validation runs ONE forward pass over the val split, not two.
    Returns `(val_seg_acc, val_det_mAP50)`.
    """
    model.eval()
    ignore = cfg.IGNORED_LABELS[0] if getattr(cfg, "IGNORED_LABELS", None) else -1

    correct = total = 0
    metric = None
    if has_det:
        from metrics.detection_metrics import DetectionMetrics
        from .evaluate import _det_add_batch
        class_names = list(getattr(cfg, "CLASS_NAMES", []))
        det_class_names = (class_names[:cfg.NUM_CLASSES_DET]
                           if class_names else None)
        metric = DetectionMetrics(num_classes=cfg.NUM_CLASSES_DET,
                                  iou_thresholds=[0.25, 0.5, 0.75],
                                  class_names=det_class_names)

    for batch in val_loader:
        batch = to_device(batch, device)
        out = model(batch)
        if has_seg:
            pred = out["seg_logits"].argmax(dim=1)
            gt = batch["segment"].long()
            valid = gt != ignore
            correct += int((pred[valid] == gt[valid]).sum())
            total += int(valid.sum())
        if has_det:
            _det_add_batch(out, batch, metric, nms_iou=0.2)

    val_acc = (correct / max(total, 1)) if has_seg else 0.0
    val_mAP50 = 0.0
    if has_det:
        val_mAP50 = float(metric.compute().get("mAP@0.5", 0.0))
    return val_acc, val_mAP50
