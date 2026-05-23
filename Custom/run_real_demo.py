"""Real end-to-end demonstration of the JePT pipeline.

Unlike `smoke_test.py` (which only checks that code runs on random data), this
script proves the pipeline *learns*:

  1. generates structured indoor scenes where geometry truly determines labels
     (Custom/make_synthetic_dataset.py),
  2. JEPA-pretrains the LitePT backbone on UNLABELED scenes,
  3. fine-tunes seg+det heads on a small LABELED set — twice:
        (a) starting from the JEPA-pretrained backbone,
        (b) starting from scratch (ablation baseline),
  4. evaluates BOTH on a held-out UNSEEN test set (mean IoU / accuracy),
  5. prints a report so pretraining's effect is visible and honest.

Run from the JePT/ directory:
    python -m Custom.run_real_demo
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
from engine.evaluate import evaluate_segmentation            # noqa: E402
from engine.finetune import finetune                         # noqa: E402
from engine.pretrain import pretrain                         # noqa: E402
from Custom.make_synthetic_dataset import (                         # noqa: E402
    SEG_CLASS_NAMES, DET_CLASS_NAMES, generate)

DEMO = os.path.join(_ROOT, "data_demo")
EXP = os.path.join(_ROOT, "exp_demo")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# scale knobs (kept modest so the demo finishes on CPU)
N_UNLABELED = 24
N_TRAIN, N_VAL, N_TEST = 24, 8, 8
PRETRAIN_EPOCHS = 18
FINETUNE_EPOCHS = 30
VARIANT = "nano"

# `--quick`: tiny dry run to validate the full code path fast
if "--quick" in sys.argv:
    N_UNLABELED = 6
    N_TRAIN, N_VAL, N_TEST = 6, 3, 3
    PRETRAIN_EPOCHS = 2
    FINETUNE_EPOCHS = 2


def _set(module, **kw):
    for k, v in kw.items():
        setattr(module, k, v)
    return module


# ----------------------------------------------------------------------
def step_generate():
    print("\n[1] generating structured synthetic data")
    if os.path.exists(DEMO):
        shutil.rmtree(DEMO)
    generate(os.path.join(DEMO, "unlabeled"), N_UNLABELED, labeled=False, seed=1)
    generate(os.path.join(DEMO, "labeled"), N_TRAIN, labeled=True, seed=100,
             split="train")
    generate(os.path.join(DEMO, "labeled"), N_VAL, labeled=True, seed=200,
             split="val")
    generate(os.path.join(DEMO, "labeled"), N_TEST, labeled=True, seed=300,
             split="test")


def step_pretrain():
    print("\n[2] JEPA pretraining on unlabeled scenes")
    import Custom.pretrain_config as pc
    _set(pc,
         UNLABELED_DATA_PATH=os.path.join(DEMO, "unlabeled"),
         RESULTS_DIR=os.path.join(EXP, "pretrain"),
         MODEL_VARIANT=VARIANT, INPUT_CHANNELS=6, GRID_SIZE=0.05,
         AUGMENT=True, GROUP_SIZE=0.3, MIN_TOKENS=8, MAX_TOKENS=1024,
         NUM_TARGET_BLOCKS=4, TARGET_RATIO=(0.15, 0.20),
         PREDICTOR_DIM=192, PREDICTOR_DEPTH=4, PREDICTOR_HEADS=6,
         EPOCHS=PRETRAIN_EPOCHS, BATCH_SIZE=4, LR=1.5e-3, WARMUP_EPOCHS=2,
         EMA_TAU_EPOCHS=PRETRAIN_EPOCHS, NUM_WORKERS=0, USE_AMP=False,
         SEED=0, SAVE_EVERY=PRETRAIN_EPOCHS)
    t0 = time.time()
    hist = pretrain(pc, verbose=True)
    print(f"    pretraining took {time.time() - t0:.0f}s")
    return os.path.join(pc.RESULTS_DIR, "last.pth"), hist


def step_finetune(pretrained_ckpt, tag):
    print(f"\n[3:{tag}] fine-tuning "
          f"({'JEPA-pretrained' if pretrained_ckpt else 'from scratch'})")
    import Custom.finetune_config as fc
    _set(fc,
         DATA_PATH=os.path.join(DEMO, "labeled"), DATA_FORMAT="npy",
         RESULTS_DIR=os.path.join(EXP, f"finetune_{tag}"),
         CLASS_NAMES=SEG_CLASS_NAMES,
         NUM_CLASSES_SEG=len(SEG_CLASS_NAMES),   # derived from the name lists
         NUM_CLASSES_DET=len(DET_CLASS_NAMES),
         MODEL_VARIANT=VARIANT, USE_DUAL_PATH_UNIFIED=False,
         INPUT_CHANNELS=6, GRID_SIZE=0.05, AUGMENT=True,
         CLASS_WEIGHTS="auto", PRETRAINED_CKPT=pretrained_ckpt,
         # full_finetune for BOTH runs: a clean ablation — identical training
         # regime, only the backbone initialisation differs.
         FREEZE_MODE="full_finetune", FREEZE_EPOCHS=5, BACKBONE_LR_SCALE=0.3,
         EPOCHS=FINETUNE_EPOCHS, BATCH_SIZE=2, LEARNING_RATE=1e-3,
         NUM_WORKERS=0, USE_AMP=False)
    t0 = time.time()
    model, hist = finetune(fc, verbose=True)
    print(f"    fine-tuning took {time.time() - t0:.0f}s")

    test_ds = CustomDataset(fc.DATA_PATH, "test", fc, "npy")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             collate_fn=collate_point_batch)
    metrics = evaluate_segmentation(model, test_loader, fc.NUM_CLASSES_SEG,
                                    DEVICE)
    return hist, metrics


def main():
    print("=" * 64)
    print("JePT — REAL end-to-end demonstration")
    print(f"  device={DEVICE}  variant={VARIANT}")
    print("=" * 64)

    step_generate()
    ckpt, pre_hist = step_pretrain()

    hist_pre, test_pre = step_finetune(ckpt, "pretrained")
    hist_scr, test_scr = step_finetune(None, "scratch")

    # ---- report -----------------------------------------------------
    print("\n" + "=" * 64)
    print("RESULTS")
    print("=" * 64)
    print(f"JEPA pretraining loss : {pre_hist['loss'][0]:.4f} "
          f"-> {pre_hist['loss'][-1]:.4f}  "
          f"(target_std {pre_hist['target_std'][-1]:.3f})")
    print(f"  {'pretraining genuinely reduced the loss' if pre_hist['loss'][-1] < pre_hist['loss'][0] else 'WARNING: loss did not decrease'}")

    print("\nFine-tuning val segmentation accuracy (per epoch):")
    print(f"  pretrained: {[round(x, 3) for x in hist_pre['val_seg_acc']]}")
    print(f"  scratch   : {[round(x, 3) for x in hist_scr['val_seg_acc']]}")

    print("\nHeld-out UNSEEN test set (segmentation):")
    print(f"  {'init':<22s} {'accuracy':>10s} {'mIoU':>10s}")
    print(f"  {'JEPA-pretrained':<22s} {test_pre['acc']:>10.4f} "
          f"{test_pre['miou']:>10.4f}")
    print(f"  {'from scratch':<22s} {test_scr['acc']:>10.4f} "
          f"{test_scr['miou']:>10.4f}")
    chance = 1.0 / len(SEG_CLASS_NAMES)
    print(f"  (random-guess accuracy ~= {chance:.3f})")

    delta = test_pre["miou"] - test_scr["miou"]
    print("\nCONCLUSION:")
    if test_pre["miou"] > 0.5:
        print(f"  - Pipeline learns: pretrained model reaches "
              f"{test_pre['miou']:.3f} mIoU on unseen data (>> chance).")
    else:
        print(f"  - Pretrained model mIoU={test_pre['miou']:.3f} — needs more "
              f"epochs/data for this run.")
    if delta > 0.01:
        print(f"  - JEPA pretraining HELPS: +{delta:.3f} mIoU over scratch.")
    elif delta < -0.01:
        print(f"  - On this small/easy demo, scratch matched/beat pretrain "
              f"({delta:+.3f} mIoU) — expected; SSL gains grow with data scale.")
    else:
        print(f"  - Pretrained and scratch are within {abs(delta):.3f} mIoU "
              f"on this easy demo.")
    print("=" * 64)


if __name__ == "__main__":
    main()
