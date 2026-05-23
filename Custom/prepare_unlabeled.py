"""Scan a folder for unlabeled point clouds and write a manifest.

Usage:
    python Custom/prepare_unlabeled.py <folder>

Counts NPY scene folders (containing coord.npy) and .ply files and writes
`unlabeled_manifest.json` next to them. The dataset loader discovers files on
its own — this tool is just a convenience inventory / sanity check.
"""

import glob
import json
import os
import sys


def scan(root: str) -> dict:
    npy = sorted(d for d in glob.glob(os.path.join(root, "**", "*"),
                                      recursive=True)
                 if os.path.isdir(d)
                 and os.path.exists(os.path.join(d, "coord.npy")))
    ply = sorted(glob.glob(os.path.join(root, "**", "*.ply"), recursive=True))
    return {"root": os.path.abspath(root),
            "npy_folders": npy, "ply_files": ply,
            "total": len(npy) + len(ply)}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    root = sys.argv[1]
    if not os.path.isdir(root):
        print(f"Not a directory: {root}")
        sys.exit(1)
    manifest = scan(root)
    out = os.path.join(root, "unlabeled_manifest.json")
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Found {manifest['total']} unlabeled clouds "
          f"({len(manifest['npy_folders'])} NPY, {len(manifest['ply_files'])} PLY)")
    print(f"Manifest written to {out}")


if __name__ == "__main__":
    main()
