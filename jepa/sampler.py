"""Context / target block sampling (I-JEPA multi-block, Point-JEPA recipe).

For every sample we draw `num_target_blocks` spatially-contiguous *target*
blocks from the sequenced group order. The *context* is the complement — every
group not covered by any target block. Predicting masked target blocks from the
visible context, in latent space, is the JEPA objective.

Block masking (contiguous spans in the sequenced order) — not random masking —
is what makes the latent prediction non-trivial: the model must infer a whole
coherent region it never saw.
"""

import random
from typing import List, Tuple

import torch

from .sequencer import sequence_iterative_nn


def sample_blocks(
    centers: torch.Tensor,
    num_target_blocks: int,
    target_ratio: Tuple[float, float],
    rng: random.Random,
) -> Tuple[List[int], List[List[int]]]:
    """Split one sample's groups into context + target blocks.

    Args:
        centers:           (G, 3) group centers for a single sample (LOCAL idx).
        num_target_blocks: number of target blocks to sample.
        target_ratio:      (min, max) fraction of G per target block.
        rng:               seeded `random.Random` for reproducibility.
    Returns:
        context: list of local group indices forming the context.
        targets: list of `num_target_blocks` lists of local group indices.
    """
    g = centers.shape[0]
    order = sequence_iterative_nn(centers).tolist()   # spatial order of local idx

    covered = set()                                   # positions in `order`
    targets: List[List[int]] = []
    for _ in range(num_target_blocks):
        ratio = rng.uniform(*target_ratio)
        blk = max(1, int(round(ratio * g)))
        blk = min(blk, g)
        start = rng.randint(0, g - blk)
        positions = range(start, start + blk)
        targets.append([order[p] for p in positions])
        covered.update(positions)

    context = [order[p] for p in range(g) if p not in covered]
    return context, targets
