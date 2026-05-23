"""Real SSL -> supervised demo on the Shapes3D dataset.

Shapes3D (Custom/shapes3d_generator.py — bundled here) generates scenes: a flat
floor plane plus 3-5 floating geometric solids (cube/sphere/cylinder/cone/
pyramid/torus). 7 segmentation classes (6 shapes + floor), 6 detection classes
(the shapes only — the floor plane gets no box). Point COLOUR is randomised per
shape instance and carries NO class signal — only GEOMETRY determines the
label, so the heads cannot cheat by reading the colour channel. Genuine benchmark.

This script demonstrates JePT exactly as intended for real use:
  1. generate a Shapes3D dataset (train / val / test),
  2. JEPA-pretrain the LitePT backbone on the train scenes WITHOUT labels,
  3. fine-tune seg+det heads on only a FEW labeled train scenes — twice
     (JEPA-pretrained vs. from scratch),
  4. evaluate both on the held-out UNSEEN test set,
  5. write a checkpoint (with class-name meta) ready for Custom/visualize.py.

Run from the JePT/ directory:
    python -m Custom.run_shapes3d_demo
    python -m Custom.run_shapes3d_demo --quick      # fast dry run
"""

import os
import shutil
import sys
import time

import torch
from torch.utils.data import DataLoader

_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from hybrid_backend import setup_backends  # noqa: E402

setup_backends(verbose=False)

from data import CustomDataset, collate_point_batch          # noqa: E402
from engine.evaluate import evaluate_detection, evaluate_segmentation  # noqa: E402
from engine.finetune import finetune                         # noqa: E402
from engine.pretrain import pretrain                         # noqa: E402
from Custom.shapes3d_generator import (                      # noqa: E402
    SHAPE_CLASSES, generate_scene)

DATA = os.path.join(_ROOT, "data_shapes3d")
EXP = os.path.join(_ROOT, "exp_shapes3d")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = SHAPE_CLASSES + ["floor"]         # 7 segmentation classes

# scale knobs — edit freely; everything downstream derives from these.
#
# Scientifically correct SSL regime: the UNLABELED pretrain pool (N_TRAIN) is
# large; only a small labelled SUBSET (N_LABELED) is annotated for fine-tuning;
# the model is judged on a disjoint held-out TEST split it never saw in either
# stage. The demo then fine-tunes TWICE — JEPA-pretrained vs from-scratch — on
# the *same* labelled subset, so the test-set gap is a clean measure of what
# the self-supervised pretraining bought.
N_TRAIN, N_VAL, N_TEST = 200, 30, 30
N_LABELED = 80                  # detection is data-hungrier than segmentation;
                                # 80 labelled scenes (40% of N_TRAIN) gives the
                                # per-point box head enough examples per class
                                # to converge without collapsing to majority
PRETRAIN_EPOCHS = 100
FINETUNE_EPOCHS = 100
VARIANT = "nano"
GRID_SIZE = 0.6
# Dual-path unified (pyLitePT's "best performance" recipe):
#   - seg branch = multi-stage encoder-decoder (full-cloud features)
#   - det branch = SINGLE-STAGE encoder (no downsampling) -> high-resolution
#     features that small objects need. The JEPA-pretrained multi-stage
#     backbone transfers fully to the seg branch and partially to the det
#     branch (input embedding only). The det branch trains the rest fresh.
USE_DUAL_PATH = True
POINTS_PER_SHAPE, FLOOR_POINTS = 600, 1200      # smaller -> faster on CPU

if "--quick" in sys.argv:
    N_TRAIN, N_VAL, N_TEST, N_LABELED = 6, 3, 3, 4
    PRETRAIN_EPOCHS = FINETUNE_EPOCHS = 2


def _set(module, **kw):
    for k, v in kw.items():
        setattr(module, k, v)
    return module


# ----------------------------------------------------------------------
def step_generate():
    print("\n[1] generating Shapes3D dataset")
    if os.path.exists(DATA):
        shutil.rmtree(DATA)
    for split, n in (("train", N_TRAIN), ("val", N_VAL), ("test", N_TEST)):
        for i in range(n):
            generate_scene(i, DATA, split, points_per_shape=POINTS_PER_SHAPE,
                           floor_points=FLOOR_POINTS)
        print(f"  {split}: {n} scenes")


def step_pretrain():
    print("\n[2] JEPA pretraining on Shapes3D train scenes (labels ignored)")
    import Custom.pretrain_config as pc
    _set(pc,
         UNLABELED_DATA_PATH=DATA,            # UnlabeledPointDataset reads train/
         RESULTS_DIR=os.path.join(EXP, "pretrain"),
         MODEL_VARIANT=VARIANT, INPUT_CHANNELS=6, GRID_SIZE=GRID_SIZE,
         AUGMENT=True, GROUP_SIZE=8.0, MIN_TOKENS=8, MAX_TOKENS=1024,
         NUM_TARGET_BLOCKS=4, TARGET_RATIO=(0.15, 0.20),
         PREDICTOR_DIM=192, PREDICTOR_DEPTH=4, PREDICTOR_HEADS=6,
         EPOCHS=PRETRAIN_EPOCHS, BATCH_SIZE=4, LR=1.5e-3,
         WARMUP_EPOCHS=max(1, PRETRAIN_EPOCHS // 10),     # derived, not fixed
         EMA_TAU_EPOCHS=PRETRAIN_EPOCHS, NUM_WORKERS=0, USE_AMP=False,
         SEED=0, SAVE_EVERY=max(1, PRETRAIN_EPOCHS // 5))
    t0 = time.time()
    hist = pretrain(pc, verbose=True)
    print(f"    pretraining took {time.time() - t0:.0f}s")
    return os.path.join(pc.RESULTS_DIR, "last.pth"), hist


def step_finetune(pretrained_ckpt, tag):
    print(f"\n[3:{tag}] fine-tuning on {N_LABELED} labeled scenes "
          f"({'JEPA-pretrained' if pretrained_ckpt else 'from scratch'})")
    import Custom.finetune_config as fc
    _set(fc,
         DATA_PATH=DATA, DATA_FORMAT="npy",
         RESULTS_DIR=os.path.join(EXP, f"finetune_{tag}"),
         CLASS_NAMES=CLASS_NAMES,
         NUM_CLASSES_SEG=len(CLASS_NAMES),       # derived: 6 shapes + floor
         NUM_CLASSES_DET=len(SHAPE_CLASSES),     # derived: shapes only (no floor)
         MODEL_VARIANT=VARIANT, USE_DUAL_PATH_UNIFIED=USE_DUAL_PATH,
         INPUT_CHANNELS=6, GRID_SIZE=GRID_SIZE, AUGMENT=True,
         CLASS_WEIGHTS="auto", PRETRAINED_CKPT=pretrained_ckpt,
         FREEZE_MODE="full_finetune", BACKBONE_LR_SCALE=0.3,
         EPOCHS=FINETUNE_EPOCHS, BATCH_SIZE=2, LEARNING_RATE=1e-3,
         NUM_WORKERS=0, USE_AMP=False)
    t0 = time.time()
    model, hist = finetune(fc, verbose=True, max_train_scenes=N_LABELED)
    print(f"    fine-tuning took {time.time() - t0:.0f}s")

    test_ds = CustomDataset(DATA, "test", fc, "npy")
    loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                        collate_fn=collate_point_batch)
    seg = evaluate_segmentation(model, loader, fc.NUM_CLASSES_SEG, DEVICE)
    det = evaluate_detection(model, loader, fc.NUM_CLASSES_DET, DEVICE)
    return hist, seg, det


def _report(tag, seg, det):
    print(f"\n[{tag}]")
    print(f"  Segmentation : acc={seg['acc']:.4f}  mIoU={seg['miou']:.4f}")
    print("    per-class IoU: " + "  ".join(
        f"{n}={v:.2f}" for n, v in zip(CLASS_NAMES, seg["per_class_iou"])))
    print(f"  Detection    : recall={det['recall']:.4f}  "
          f"precision={det['precision']:.4f}  ({det['n_gt']} GT objects)")
    print("    per-class recall: " + "  ".join(
        f"{n}={v:.2f}" for n, v in zip(SHAPE_CLASSES, det["per_class_recall"])))


def main():
    print("=" * 64)
    print("JePT — Shapes3D real SSL -> supervised demo")
    print(f"  device={DEVICE}  variant={VARIANT}  classes={CLASS_NAMES}")
    print(f"  unlabeled pretrain pool={N_TRAIN}  labelled subset={N_LABELED}"
          f"  held-out test={N_TEST}")
    print("=" * 64)

    step_generate()
    ckpt, pre_hist = step_pretrain()

    # controlled comparison: same labelled subset, JEPA-pretrained vs scratch
    _, seg_pre, det_pre = step_finetune(ckpt, "pretrained")
    _, seg_scr, det_scr = step_finetune(None, "scratch")

    print("\n" + "=" * 64)
    print("RESULTS  (Shapes3D, held-out UNSEEN test set)")
    print("=" * 64)
    print(f"JEPA pretraining loss : {pre_hist['loss'][0]:.4f} -> "
          f"{pre_hist['loss'][-1]:.4f}  "
          f"(target_std {pre_hist['target_std'][-1]:.3f} — >0 = no collapse)")
    _report("JEPA-pretrained -> fine-tuned", seg_pre, det_pre)
    _report("from scratch    -> fine-tuned", seg_scr, det_scr)
    print("\n" + "-" * 64)
    print(f"JEPA gain on {N_LABELED} labelled scenes:  "
          f"seg mIoU {seg_scr['miou']:.3f} -> {seg_pre['miou']:.3f} "
          f"({seg_pre['miou'] - seg_scr['miou']:+.3f})   "
          f"det recall {det_scr['recall']:.3f} -> {det_pre['recall']:.3f} "
          f"({det_pre['recall'] - det_scr['recall']:+.3f})")
    print("-" * 64)
    print(f"\nVisualise the result:")
    print(f"  python Custom/visualize.py --checkpoint "
          f"{os.path.join(EXP, 'finetune_pretrained', 'best.pth')}")
    print("=" * 64)


if __name__ == "__main__":
    main()
