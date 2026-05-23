"""Spatial token ordering (Point-JEPA's point sequencer, ragged-aware).

Point-JEPA orders tokens with an iterative nearest-neighbour walk so that
contiguous index ranges correspond to spatially contiguous regions — this is
what lets the context/target sampler carve coherent *blocks* rather than
scattered points. The original implementation assumes a dense `(B, N, C)`
tensor; here we operate per sample because JePT groups are ragged (each scene /
object yields a different group count).
"""

import torch


@torch.no_grad()
def sequence_iterative_nn(centers: torch.Tensor) -> torch.Tensor:
    """Greedy nearest-neighbour ordering of group centers for ONE sample.

    Starts from the center with the smallest coordinate sum (a deterministic
    corner) and repeatedly hops to the nearest unvisited center.

    Args:
        centers: (G, 3) group centers of a single sample.
    Returns:
        (G,) long permutation — `centers[order]` is the spatially-sorted set.
    """
    g = centers.shape[0]
    if g <= 1:
        return torch.arange(g, device=centers.device, dtype=torch.long)

    dist = torch.cdist(centers, centers)          # (G, G)
    start = int(torch.argmin(centers.sum(dim=1)))
    order = [start]
    # mask out visited columns by setting their distance to +inf
    work = dist.clone()
    work[:, start] = float("inf")
    cur = start
    for _ in range(g - 1):
        nxt = int(torch.argmin(work[cur]))
        order.append(nxt)
        work[:, nxt] = float("inf")
        cur = nxt
    return torch.tensor(order, device=centers.device, dtype=torch.long)
