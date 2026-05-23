"""
=============================================================================
DETECTION POST-PROCESSING
=============================================================================
Reusable post-processing utilities for 3D detection predictions.
Extracts confidence filtering + NMS logic used by both visualize.py and
detection.py.
"""

import torch
import numpy as np


def filter_and_nms(boxes, scores, conf_thresh=0.5, iou_thresh=0.1,
                   cls_preds=None, max_boxes=None, topk_fallback=0):
    """
    Apply confidence filtering and PER-CLASS NMS to detection predictions.

    Uses torchvision's axis-aligned BEV NMS on the AABB of rotated boxes
    for fast, GPU-compatible NMS. NMS is run PER CLASS (batched_nms) so a
    box of one class never suppresses an overlapping box of another class
    — without this, neighbouring objects of different classes drop out.

    Args:
        boxes: (N, 7+) [x, y, z, dx, dy, dz, heading, ...]
        scores: (N,) confidence scores
        conf_thresh: Minimum confidence threshold
        iou_thresh: IoU threshold for NMS suppression (within a class)
        cls_preds: (N, C) optional class logits for per-box class labels.
                   When given, NMS is class-aware.
        max_boxes: Optional maximum number of boxes to return
        topk_fallback: if >0 and the confidence filter keeps nothing, fall
                       back to the top-`topk_fallback` boxes by raw score.
                       Guards against the saturated-score regime wiping a
                       whole scene when the slider sits a hair too high.

    Returns:
        dict with keys:
            'boxes': (K, 7) filtered boxes
            'scores': (K,) filtered scores
            'labels': (K,) predicted class labels (0-based)
            'keep_indices': (K,) indices into the original input
    """
    if boxes is None or len(boxes) == 0:
        return {
            'boxes': np.zeros((0, 7), dtype=np.float32),
            'scores': np.zeros(0, dtype=np.float32),
            'labels': np.zeros(0, dtype=np.int64),
            'keep_indices': np.zeros(0, dtype=np.int64),
        }

    # Ensure float tensors
    boxes = boxes.float()
    scores = scores.float()
    if scores.dim() == 2:
        scores = scores.squeeze(1)

    # Per-box class labels (computed BEFORE NMS so NMS can be class-aware)
    have_cls = (cls_preds is not None and cls_preds.dim() == 2
                and cls_preds.shape[0] == boxes.shape[0])
    labels_all = (cls_preds.argmax(dim=-1) if have_cls
                  else torch.zeros(boxes.shape[0], dtype=torch.long))

    # 1. Confidence filter (with optional top-K fallback)
    mask = scores > conf_thresh
    if mask.sum() == 0:
        if topk_fallback > 0 and scores.numel() > 0:
            k = min(topk_fallback, scores.numel())
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask[torch.topk(scores, k).indices] = True
        else:
            return {
                'boxes': np.zeros((0, 7), dtype=np.float32),
                'scores': np.zeros(0, dtype=np.float32),
                'labels': np.zeros(0, dtype=np.int64),
                'keep_indices': np.zeros(0, dtype=np.int64),
            }

    boxes_f = boxes[mask]
    scores_f = scores[mask]
    labels_f = labels_all[mask]

    # 2. Compute AABB for BEV NMS
    x, y = boxes_f[:, 0], boxes_f[:, 1]
    dx, dy = boxes_f[:, 3], boxes_f[:, 4]
    heading = boxes_f[:, 6]

    cos_h = torch.abs(torch.cos(heading))
    sin_h = torch.abs(torch.sin(heading))

    w_aabb = dx * cos_h + dy * sin_h
    h_aabb = dx * sin_h + dy * cos_h

    x1 = x - w_aabb / 2
    y1 = y - h_aabb / 2
    x2 = x + w_aabb / 2
    y2 = y + h_aabb / 2

    boxes_bev = torch.stack([x1, y1, x2, y2], dim=1)

    # 3. PER-CLASS NMS via torchvision (batched_nms offsets boxes by class so
    #    cross-class boxes never suppress one another)
    import torchvision.ops
    keep_idx = torchvision.ops.batched_nms(
        boxes_bev, scores_f, labels_f, iou_threshold=iou_thresh)

    # sort survivors by score (batched_nms groups by class, not by score)
    keep_idx = keep_idx[torch.argsort(scores_f[keep_idx], descending=True)]

    if max_boxes is not None and len(keep_idx) > max_boxes:
        keep_idx = keep_idx[:max_boxes]

    # 4. Extract results
    result_boxes = boxes_f[keep_idx].cpu().numpy()
    result_scores = scores_f[keep_idx].cpu().numpy()
    result_labels = labels_f[keep_idx].cpu().numpy().astype(np.int64)

    # 5. Map indices back to original
    orig_indices = torch.where(mask)[0][keep_idx].cpu().numpy()

    return {
        'boxes': result_boxes,
        'scores': result_scores,
        'labels': result_labels,
        'keep_indices': orig_indices,
    }
