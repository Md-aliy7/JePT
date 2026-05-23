"""Scale-adaptive point-cloud grouping for JEPA.

Point-JEPA tokenises a fixed-size cloud with FPS + KNN. That assumes a constant
point count, which breaks for *mixed* scene/object data. Instead JePT defines a
"token" as a coarse voxel group: points are bucketed by a coarse grid of edge
`group_size`. Grid bucketing is scale-adaptive for free — a 100k-point scene
yields many groups, a 5k-point object yields few — and is purely geometric, so
the student and the EMA teacher always agree on the group set.
"""

import torch


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Mean of `src` rows grouped by `index`.

    Args:
        src:      (N, C) features.
        index:    (N,) group id in [0, dim_size) for each row.
        dim_size: number of groups G.
    Returns:
        (G, C) per-group mean. Empty groups are zero.
    """
    out = torch.zeros(dim_size, src.shape[1], dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    count.index_add_(0, index, torch.ones(src.shape[0], dtype=src.dtype, device=src.device))
    return out / count.clamp(min=1.0).unsqueeze(1)


@torch.no_grad()
def compute_groups(coord: torch.Tensor, batch: torch.Tensor, group_size: float):
    """Bucket points into coarse voxel groups.

    Args:
        coord:      (N, 3) float point coordinates.
        batch:      (N,) long batch index per point.
        group_size: coarse voxel edge length (metres).
    Returns:
        group_id:     (N,) long, global group index per point in [0, G).
        group_center: (G, 3) float, mean coordinate of each group.
        group_batch:  (G,) long, batch index of each group.
    """
    gc = torch.floor(coord / group_size).long()
    gc = gc - gc.min(dim=0).values
    # lexicographic key: (batch, gx, gy, gz) -> unique rows give groups
    keys = torch.stack([batch, gc[:, 0], gc[:, 1], gc[:, 2]], dim=1)
    uniq, group_id = torch.unique(keys, dim=0, return_inverse=True)
    num_groups = uniq.shape[0]
    group_center = scatter_mean(coord, group_id, num_groups)
    group_batch = uniq[:, 0].contiguous()  # batch is the leading key column
    return group_id, group_center, group_batch
