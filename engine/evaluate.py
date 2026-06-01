"""Evaluation metrics for JePT.

Segmentation — confusion-matrix mIoU + overall accuracy.
Detection    — real **3D IoU mAP** at IoU thresholds {0.25, 0.5, 0.75}, plus
              per-threshold recall. Backed by `metrics.detection_metrics.
              DetectionMetrics` — the exact metric used by pyLitePT, so JePT's
              detection numbers are directly comparable to pyLitePT's.
"""

import numpy as np
import torch

from .common import to_device


@torch.no_grad()
def evaluate_segmentation(model, loader, num_classes, device, ignore=-1):
    """Compute overall accuracy and mean IoU over a data loader.

    Uses a confusion matrix so mIoU is exact and order-independent. Classes
    absent from the evaluation set are excluded from the mIoU average.

    Returns:
        dict with `acc`, `miou`, `per_class_iou` (list of length num_classes).
    """
    model.eval()
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for batch in loader:
        batch = to_device(batch, device)
        logits = model(batch)["seg_logits"]
        pred = logits.argmax(dim=1).cpu()
        gt = batch["segment"].long().cpu()
        valid = (gt != ignore) & (gt >= 0) & (gt < num_classes)
        pred, gt = pred[valid], gt[valid]
        if gt.numel() == 0:
            continue
        idx = gt * num_classes + pred
        conf += torch.bincount(
            idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)

    inter = conf.diag().float()
    union = conf.sum(0).float() + conf.sum(1).float() - inter
    iou = inter / union.clamp(min=1.0)
    present = union > 0
    miou = float(iou[present].mean()) if present.any() else 0.0
    acc = float(inter.sum() / conf.sum().clamp(min=1))
    return {"acc": acc, "miou": miou, "per_class_iou": iou.tolist()}


def _det_add_batch(out, batch, metric, nms_iou):
    """Extract predictions + GT from one forward pass and feed them into a
    `DetectionMetrics` accumulator. Reused by both `evaluate_detection` and
    the fused per-epoch val helper in `engine/finetune.py`.

    `conf_thresh=0.0` — mAP integrates precision/recall over all scores, so
    a hard confidence cut would cap recall.
    """
    from Custom.postprocess import filter_and_nms

    gt = batch.get("gt_boxes")
    if gt is None:
        gt_boxes = np.zeros((0, 7), dtype=np.float32)
        gt_labels = np.zeros(0, dtype=np.int64)
    else:
        gt_np = gt.reshape(-1, gt.shape[-1]).cpu().numpy()
        keep = np.abs(gt_np[:, 3:6]).sum(1) > 1e-6       # drop padding rows
        gt_np = gt_np[keep]
        gt_boxes = gt_np[:, :7] if len(gt_np) else np.zeros((0, 7), np.float32)
        gt_labels = (gt_np[:, 7].astype(np.int64) if gt_np.shape[1] > 7
                     else np.zeros(len(gt_np), dtype=np.int64))

    if ("batch_box_preds" not in out
            or out["batch_box_preds"].numel() == 0):
        pred_boxes = np.zeros((0, 7), dtype=np.float32)
        pred_scores = np.zeros(0, dtype=np.float32)
        pred_labels = np.zeros(0, dtype=np.int64)
    else:
        cls_logits = out.get("batch_cls_preds")
        cls_logits = (cls_logits.detach().cpu()
                      if cls_logits is not None else None)
        res = filter_and_nms(
            out["batch_box_preds"].detach().cpu()[:, :7],
            out["point_cls_scores"].detach().cpu().reshape(-1),
            conf_thresh=0.0, iou_thresh=nms_iou, cls_preds=cls_logits)
        pred_boxes = res["boxes"]
        pred_scores = res["scores"]
        pred_labels = res["labels"].astype(np.int64)

    metric.add_batch(pred_boxes, pred_scores, pred_labels,
                     gt_boxes, gt_labels)


def _per_class_ap50(results, num_det, class_names):
    """Surface a fixed-length per-class AP@0.5 list for downstream reporting."""
    out = []
    for c in range(num_det):
        name = (class_names[c] if class_names and c < len(class_names)
                else f"class_{c}")
        out.append(float(results.get(f"AP_{name}@0.5", 0.0)))
    return out


@torch.no_grad()
def evaluate_detection(model, loader, num_det, device,
                       class_names=None, nms_iou=0.2,
                       iou_thresholds=(0.25, 0.5, 0.75)):
    """3D-IoU mAP detection evaluation (pyLitePT-compatible).

    Per scene: forward → per-class BEV NMS dedupe of the per-point box flood
    (`Custom.postprocess.filter_and_nms`) → `DetectionMetrics`, which computes
    per-class AP via area-under-PR-curve at each IoU threshold using full
    rotated-3D IoU.

    Returns a dict with `mAP@T`, `recall@T` for each `T` in `iou_thresholds`,
    plus `total_gt`, `total_pred`, and `per_class_AP@0.5` for the legend.
    """
    from metrics.detection_metrics import DetectionMetrics

    model.eval()
    metric = DetectionMetrics(num_classes=num_det,
                              iou_thresholds=list(iou_thresholds),
                              class_names=class_names)
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch)
        _det_add_batch(out, batch, metric, nms_iou)

    results = metric.compute()
    results["per_class_AP@0.5"] = _per_class_ap50(results, num_det, class_names)
    return results
