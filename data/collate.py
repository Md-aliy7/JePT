"""Batch collation for point clouds (ragged -> flat with offsets).

Adapted from pyLitePT `datasets/utils.py:collate_fn`. Point clouds have
variable point counts, so a batch is stored as concatenated tensors plus an
`offset` / `batch` index — the contract expected by `LitePT.forward`.

Works for both the labeled loader (`segment`, `gt_boxes` present) and the
unlabeled loader (those keys simply absent).
"""

import torch

from models.utils import offset2batch


def _grid_size_scalar(gs) -> float:
    if torch.is_tensor(gs):
        return float(gs.reshape(-1)[0].item())
    try:
        return float(gs[0])
    except (TypeError, IndexError):
        return float(gs)


def collate_point_batch(batch):
    """Collate a list of per-sample dicts into one batched dict.

    Produces keys: `coord`, `feat`, `grid_size`, `offset`, `batch`,
    `grid_coord`, plus any of `segment` / `gt_boxes` / `name` that are present.
    """
    assert len(batch) > 0, "empty batch"
    keys = list(batch[0].keys())
    result = {}

    lengths = [d["coord"].shape[0] for d in batch]
    offset = torch.cumsum(torch.tensor(lengths, dtype=torch.long), dim=0)

    for key in keys:
        vals = [d[key] for d in batch]
        if key == "gt_boxes":
            # pad to (B, M_max, C) — detection ground truth
            max_b = max((v.shape[0] for v in vals), default=0)
            dim = vals[0].shape[1] if vals[0].ndim == 2 else 8
            out = torch.zeros(len(vals), max(max_b, 1), dim, dtype=torch.float32)
            for i, v in enumerate(vals):
                if v.shape[0] > 0:
                    out[i, : v.shape[0], :] = v
            result[key] = out
        elif key in ("name", "split"):
            result[key] = list(vals)
        elif key == "grid_size":
            result[key] = vals[0] if torch.is_tensor(vals[0]) \
                else torch.tensor(vals[0], dtype=torch.float32)
        elif torch.is_tensor(vals[0]):
            result[key] = torch.cat(vals, dim=0)
        else:
            result[key] = list(vals)

    result["offset"] = offset
    result["batch"] = offset2batch(offset)

    if "grid_size" not in result:
        result["grid_size"] = torch.tensor([0.02], dtype=torch.float32)

    # voxel coordinates for sparse conv / serialization
    if "grid_coord" not in result and "coord" in result:
        gs = _grid_size_scalar(result["grid_size"])
        coord = result["coord"]
        result["grid_coord"] = torch.div(
            coord - coord.min(0)[0], gs, rounding_mode="trunc").int()

    return result
