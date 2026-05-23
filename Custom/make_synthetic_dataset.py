"""Structured synthetic indoor-scene generator for JePT.

Produces *learnable* point-cloud scenes — geometry genuinely determines the
label — so training improvement is real and measurable (unlike pure random
data). Each scene is a small room containing:

  segmentation classes        detection classes
  --------------------        -----------------
  0 floor   1 wall  2 table   0 box  1 sphere  2 cylinder
  3 box     4 sphere  5 cylinder       (seg class 3/4/5 -> det 0/1/2)

Output is the pyLitePT / labelCloud NPY-folder format consumed directly by
JePT's datasets: each scene is a folder with coord.npy, color.npy, segment.npy,
gt_boxes.npy ([x,y,z,dx,dy,dz,heading,label]).

Usage:
    python tools/make_synthetic_dataset.py --out data_demo --scenes 40
"""

import argparse
import os

import numpy as np

SEG_CLASS_NAMES = ["floor", "wall", "table", "box", "sphere", "cylinder"]
DET_CLASS_NAMES = ["box", "sphere", "cylinder"]


def _colorize(n, seg_class, rng):
    """Class-INDEPENDENT random colour for one instance + per-point noise.

    Colour carries NO class signal (`seg_class` is intentionally ignored): a
    given object class is a different colour every scene. This forces the
    segmentation head to learn GEOMETRY instead of trivially reading the colour
    channel as the label — a fixed per-class palette would make any mIoU here a
    colour lookup, not real learning.
    """
    base = rng.uniform(0.15, 0.85, 3).astype(np.float32)
    return np.clip(base + rng.normal(0, 0.05, (n, 3)), 0, 1).astype(np.float32)


def _plane(rng, n, x_rng, y_rng, z, axis="z", jitter=0.01):
    """Sample a jittered planar patch. axis = which coord is fixed."""
    a = rng.uniform(*x_rng, n).astype(np.float32)
    b = rng.uniform(*y_rng, n).astype(np.float32)
    c = np.full(n, z, np.float32) + rng.normal(0, jitter, n).astype(np.float32)
    if axis == "z":
        return np.stack([a, b, c], 1)
    if axis == "x":
        return np.stack([c, a, b], 1)
    return np.stack([a, c, b], 1)            # axis == "y"


def _box_surface(rng, n, center, dims):
    """Points on the surface of an axis-aligned box."""
    pts = (rng.random((n, 3)) - 0.5).astype(np.float32) * dims
    # snap each point to the nearest face
    face = rng.integers(0, 3, n)
    sign = rng.choice([-0.5, 0.5], n).astype(np.float32)
    for ax in range(3):
        m = face == ax
        pts[m, ax] = sign[m] * dims[ax]
    return pts + center


def _sphere_surface(rng, n, center, radius):
    v = rng.normal(0, 1, (n, 3)).astype(np.float32)
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-6)
    return v * radius + center


def _cylinder_surface(rng, n, center, radius, height):
    theta = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
    z = rng.uniform(-0.5, 0.5, n).astype(np.float32) * height
    x = np.cos(theta) * radius
    y = np.sin(theta) * radius
    return np.stack([x, y, z], 1).astype(np.float32) + center


def make_scene(rng, room=6.0, wall_h=3.0):
    """Generate one structured room scene.

    Returns (coord N3, color N3, segment N, gt_boxes M8).
    """
    coords, colors, segs = [], [], []

    def add(pts, seg_class):
        coords.append(pts)
        colors.append(_colorize(pts.shape[0], seg_class, rng))
        segs.append(np.full(pts.shape[0], seg_class, np.int64))

    # --- floor (class 0) ---------------------------------------------
    add(_plane(rng, 1600, (0, room), (0, room), 0.0, "z"), 0)
    # --- two walls (class 1) -----------------------------------------
    add(_plane(rng, 900, (0, wall_h), (0, room), 0.0, "x"), 1)   # x=0 wall
    add(_plane(rng, 900, (0, room), (0, wall_h), 0.0, "y"), 1)   # y=0 wall
    # --- table (class 2): horizontal slab on 4 legs ------------------
    tx, ty = rng.uniform(1.5, room - 2.5), rng.uniform(1.5, room - 2.5)
    top_z = rng.uniform(0.8, 1.1)
    add(_plane(rng, 500, (tx - 0.8, tx + 0.8), (ty - 0.6, ty + 0.6),
               top_z, "z"), 2)

    # --- movable objects (classes 3/4/5) -> detection targets --------
    gt_boxes = []
    n_obj = rng.integers(3, 6)
    for _ in range(n_obj):
        kind = int(rng.integers(0, 3))             # 0 box,1 sphere,2 cyl
        seg_class = kind + 3
        cx, cy = rng.uniform(0.6, room - 0.6, 2)
        if kind == 0:
            dims = rng.uniform(0.35, 0.7, 3).astype(np.float32)
            cz = dims[2] / 2.0
            center = np.array([cx, cy, cz], np.float32)
            add(_box_surface(rng, 320, center, dims), seg_class)
        elif kind == 1:
            r = rng.uniform(0.2, 0.4)
            center = np.array([cx, cy, r], np.float32)
            add(_sphere_surface(rng, 320, center, r), seg_class)
            dims = np.array([2 * r, 2 * r, 2 * r], np.float32)
        else:
            r = rng.uniform(0.18, 0.35)
            h = rng.uniform(0.4, 0.9)
            center = np.array([cx, cy, h / 2.0], np.float32)
            add(_cylinder_surface(rng, 320, center, r, h), seg_class)
            dims = np.array([2 * r, 2 * r, h], np.float32)
        heading = float(rng.uniform(-np.pi, np.pi))
        gt_boxes.append([*center.tolist(), *dims.tolist(), heading, kind])

    coord = np.concatenate(coords, 0).astype(np.float32)
    color = np.concatenate(colors, 0).astype(np.float32)
    segment = np.concatenate(segs, 0).astype(np.int64)
    gt = np.array(gt_boxes, np.float32) if gt_boxes \
        else np.zeros((0, 8), np.float32)
    return coord, color, segment, gt


def write_scene(folder, coord, color, segment=None, gt_boxes=None):
    """Write one scene as an NPY folder (labelCloud / pyLitePT format)."""
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "coord.npy"), coord.astype(np.float32))
    np.save(os.path.join(folder, "color.npy"), color.astype(np.float32))
    if segment is not None:
        np.save(os.path.join(folder, "segment.npy"), segment.astype(np.int64))
    if gt_boxes is not None:
        np.save(os.path.join(folder, "gt_boxes.npy"), gt_boxes.astype(np.float32))


def generate(out_dir, n_scenes, labeled, seed=0, split=None):
    """Generate `n_scenes` scenes into `out_dir` (optionally under `split/`)."""
    rng = np.random.default_rng(seed)
    base = os.path.join(out_dir, split) if split else out_dir
    for i in range(n_scenes):
        coord, color, segment, gt = make_scene(rng)
        write_scene(os.path.join(base, f"scene_{i:04d}"), coord, color,
                    segment if labeled else None,
                    gt if labeled else None)
    print(f"  generated {n_scenes} scenes -> {base} "
          f"({'labeled' if labeled else 'unlabeled'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default=None)
    ap.add_argument("--unlabeled", action="store_true")
    args = ap.parse_args()
    generate(args.out, args.scenes, not args.unlabeled, args.seed, args.split)


if __name__ == "__main__":
    main()
