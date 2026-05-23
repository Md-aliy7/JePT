"""Unlabeled point-cloud dataset for JEPA pretraining.

No annotations are needed: only `coord` (+ optional `color` / `normal`) is
read. Scenes (NPY folders) and objects (NPY folders or `.ply` files) are mixed
freely in one dataset — JEPA pretraining is scale-agnostic.

Returns per sample: `coord` (N,3), `feat` (N,C), `grid_size`, `name`.
"""

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .labeled_dataset import CustomDataset      # reuse the binary-PLY reader
from .transforms import augment


class UnlabeledPointDataset(Dataset):
    """Dataset of raw, unlabeled point clouds for self-supervised pretraining.

    Args:
        data_root:  folder containing NPY scene folders and/or `.ply` files.
                    Sub-folders `train/` / `val/` are used if present.
        cfg:        config module — reads `GRID_SIZE`, `INPUT_CHANNELS`,
                    `USE_GRID_SAMPLE`, `AUGMENT`.
        split:      'train' (augmented) or 'val' (deterministic).
    """

    def __init__(self, data_root, cfg=None, split="train"):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.cfg = cfg
        self.grid_size = float(getattr(cfg, "GRID_SIZE", 0.02))
        self.target_channels = getattr(cfg, "INPUT_CHANNELS", 6)
        if not isinstance(self.target_channels, int):
            self.target_channels = 6
        self.use_grid_sample = getattr(cfg, "USE_GRID_SAMPLE", True)
        self.augment = getattr(cfg, "AUGMENT", True) and split == "train"

        self.samples = self._discover(data_root, split)
        print(f"UnlabeledPointDataset [{split}]: {len(self.samples)} clouds "
              f"in {data_root}")

    # ------------------------------------------------------------------
    def _discover(self, root, split):
        """Find NPY scene folders and `.ply` files (optionally under split/)."""
        search_dirs = []
        split_dir = os.path.join(root, split)
        search_dirs.append(split_dir if os.path.isdir(split_dir) else root)

        samples = []
        for d in search_dirs:
            samples += sorted(
                folder for folder in glob.glob(os.path.join(d, "*"))
                if os.path.isdir(folder)
                and os.path.exists(os.path.join(folder, "coord.npy")))
            samples += sorted(glob.glob(os.path.join(d, "*.ply")))
        return samples

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------
    def _load(self, path):
        """Return (coord (N,3), feat (N,C)) for an NPY folder or PLY file."""
        if path.endswith(".ply"):
            data = CustomDataset._read_ply_binary(path)
            coord = np.vstack((data["x"], data["y"], data["z"])).T.astype(np.float32)
            if "red" in data.dtype.names:
                rgb = np.vstack((data["red"], data["green"],
                                 data["blue"])).T.astype(np.float32)
                if rgb.max() > 1.0:
                    rgb /= 255.0
            else:
                rgb = None
            feat_extra = [rgb] if rgb is not None else []
        else:
            coord = np.load(os.path.join(path, "coord.npy")).astype(np.float32)
            feat_extra = []
            for fname in ("color", "normal"):
                fp = os.path.join(path, f"{fname}.npy")
                if os.path.exists(fp):
                    arr = np.load(fp).astype(np.float32)
                    if fname == "color" and arr.max() > 1.0:
                        arr = arr / 255.0
                    if arr.ndim == 1:
                        arr = arr[:, None]
                    feat_extra.append(arr)

        feat = np.concatenate([coord] + feat_extra, axis=1)
        feat = self._fit_channels(feat, coord)
        return coord, feat

    def _fit_channels(self, feat, coord):
        """Pad / trim feature channels to `target_channels`."""
        c = feat.shape[1]
        if c == self.target_channels:
            return feat
        if c > self.target_channels:
            return feat[:, : self.target_channels]
        # pad missing channels with a neutral 0.5 (e.g. unknown colour)
        pad = np.full((coord.shape[0], self.target_channels - c), 0.5,
                      dtype=np.float32)
        return np.concatenate([feat, pad], axis=1)

    @staticmethod
    def _fnv_hash(arr):
        """FNV-64-1A hash of integer voxel coordinates (collision-free, any
        scene size). Same hash Pointcept / Sonata use for GridSample."""
        arr = np.ascontiguousarray(arr).astype(np.uint64, copy=True)
        h = np.uint64(14695981039346656037) * np.ones(arr.shape[0], np.uint64)
        for j in range(arr.shape[1]):
            h *= np.uint64(1099511628211)
            h = np.bitwise_xor(h, arr[:, j])
        return h

    def _grid_sample(self, coord, feat):
        """Hash-based voxel down-sampling (one point per voxel)."""
        gc = np.floor(coord / self.grid_size).astype(np.int64)
        gc -= gc.min(0, keepdims=True)
        keys = self._fnv_hash(gc)            # collision-free for any extent
        if self.split == "train":
            perm = np.random.permutation(len(keys))
            _, uidx = np.unique(keys[perm], return_index=True)
            idx = perm[uidx]
        else:
            _, idx = np.unique(keys, return_index=True)
        return coord[idx], feat[idx]

    def __getitem__(self, idx):
        path = self.samples[idx]
        coord, feat = self._load(path)

        if self.augment:
            coord, feat = augment(coord, feat)

        if self.use_grid_sample:
            coord, feat = self._grid_sample(coord, feat)

        return {
            "coord": torch.from_numpy(np.ascontiguousarray(coord)),
            "feat": torch.from_numpy(np.ascontiguousarray(feat)),
            "grid_size": np.array([self.grid_size], dtype=np.float32),
            "name": os.path.basename(path.rstrip(os.sep)),
        }
