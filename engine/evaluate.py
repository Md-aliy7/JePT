"""Evaluation metrics for JePT — segmentation (acc / mIoU) and detection
(class-aware centre-distance recall / precision)."""

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


@torch.no_grad()
def evaluate_detection(model, loader, num_det, device,
                       conf_thresh=0.3, nms_iou=0.2):
    """Class-aware centre-distance detection recall / precision.

    A predicted box is a true positive for a ground-truth box if they share
    the class and the predicted centre lies within half the GT box diagonal
    of the GT centre. This is a lightweight, rotation-free proxy for mAP —
    enough to tell whether objects are being detected at all.

    Returns dict: `recall`, `precision`, `per_class_recall` (len num_det),
    `n_gt`, `n_pred`.
    """
    from Custom.postprocess import filter_and_nms
    model.eval()
    tp = fp = 0
    gt_total = np.zeros(num_det, dtype=np.int64)
    gt_hit = np.zeros(num_det, dtype=np.int64)

    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch)
        # ground-truth boxes for this single-scene batch
        gt = batch.get("gt_boxes")
        gt = gt.reshape(-1, gt.shape[-1]).cpu().numpy() if gt is not None \
            else np.zeros((0, 8), np.float32)
        gt = gt[np.abs(gt[:, 3:6]).sum(1) > 1e-6]          # drop padding rows

        if "batch_box_preds" not in out or out["batch_box_preds"].numel() == 0:
            for g in gt:
                gt_total[int(g[7]) % num_det] += 1
            continue
        cls_logits = out.get("batch_cls_preds")
        cls_logits = cls_logits.detach().cpu() if cls_logits is not None else None
        res = filter_and_nms(out["batch_box_preds"].detach().cpu()[:, :7],
                             out["point_cls_scores"].detach().cpu().reshape(-1),
                             conf_thresh=conf_thresh, iou_thresh=nms_iou,
                             cls_preds=cls_logits)
        pred_boxes, pred_lbl = res["boxes"], res["labels"]
        matched = np.zeros(len(pred_boxes), dtype=bool)

        for g in gt:
            cls = int(g[7]) % num_det
            gt_total[cls] += 1
            diag = float(np.linalg.norm(g[3:6]))
            hit = False
            for j, (pb, pl) in enumerate(zip(pred_boxes, pred_lbl)):
                if matched[j] or int(pl) != cls:
                    continue
                if np.linalg.norm(pb[:3] - g[:3]) <= 0.5 * diag:
                    matched[j] = True
                    hit = True
                    break
            if hit:
                gt_hit[cls] += 1
        tp += int(matched.sum())
        fp += int((~matched).sum())

    recall = float(gt_hit.sum() / max(gt_total.sum(), 1))
    precision = float(tp / max(tp + fp, 1))
    per_class = (gt_hit / np.maximum(gt_total, 1)).tolist()
    return {"recall": recall, "precision": precision,
            "per_class_recall": per_class,
            "n_gt": int(gt_total.sum()), "n_pred": tp + fp}
