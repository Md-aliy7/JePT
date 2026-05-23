"""Scale-aware point-cloud augmentation.

Used for both unlabeled pretraining and (optionally) labeled fine-tuning.
Augmentations preserve metric scale (no unit-sphere normalisation) so the same
pipeline is valid for *mixed* data — large indoor scenes and small objects
alike. Normalising scenes to a unit sphere would destroy real-world scale and
break detection; see plan risk note.
"""

import numpy as np


def random_rotation_z(rng: np.random.RandomState) -> np.ndarray:
    """A random rotation matrix about the gravity (Z) axis."""
    angle = rng.uniform(-np.pi, np.pi)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32)


def augment(coord: np.ndarray, feat: np.ndarray,
            rng: np.random.RandomState = None,
            jitter_sigma: float = 0.005,
            scale_range=(0.9, 1.1)) -> tuple:
    """Apply flip / Z-rotation / scaling / jitter to a single cloud.

    Args:
        coord: (N, 3) float coordinates.
        feat:  (N, C) features — first 3 channels assumed to be coords.
        rng:   numpy RandomState (a fresh default is created if None).
    Returns:
        (coord, feat) augmented. `feat[:, :3]` is kept in sync with `coord`.
    """
    if rng is None:
        rng = np.random.RandomState()
    coord = coord.astype(np.float32, copy=True)
    feat = feat.astype(np.float32, copy=True)

    # random axis flips
    if rng.rand() > 0.5:
        coord[:, 0] = -coord[:, 0]
    if rng.rand() > 0.5:
        coord[:, 1] = -coord[:, 1]

    # random rotation about Z (scale-preserving)
    coord = coord @ random_rotation_z(rng).T

    # random isotropic scaling
    coord *= rng.uniform(*scale_range)

    # per-point jitter
    coord += rng.normal(0.0, jitter_sigma, size=coord.shape).astype(np.float32)

    # keep the coordinate channels of `feat` consistent
    if feat.shape[1] >= 3:
        feat[:, :3] = coord
    return coord, feat
