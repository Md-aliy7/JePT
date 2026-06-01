"""Downstream fine-tuning configuration (Stage 2).

Mirrors pyLitePT's `Custom/config.py` and adds the JEPA-specific knobs:
`PRETRAINED_CKPT` and the freeze policy. Edit, then run:
    python -m Custom.run_finetune
"""

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# ---------------------------------------------------------------- data
# Folder of LABELED data (NPY folders with coord/color/segment/gt_boxes,
# or .ply files with sidecar *_gt_boxes.npy). Only ~1-10% need labels.
DATA_PATH = os.path.join(_ROOT, "data_labeled")
RESULTS_DIR = os.path.join(_ROOT, "exp", "jepa_finetune")
DATA_FORMAT = "npy"             # 'npy' or 'ply'

# ---------------------------------------------------------------- classes
CLASS_NAMES = ["cart", "trial", "wall", "table", "book", "chair"]
NUM_CLASSES_SEG = 6
NUM_CLASSES_DET = 6
LABEL_MAPPING = {}
IGNORED_LABELS = [-1]

# ---------------------------------------------------------------- model
MODEL_VARIANT = "nano"          # MUST match the pretraining variant
USE_DUAL_PATH_UNIFIED = False   # False -> single shared backbone (full transfer)
INPUT_CHANNELS = 6
GRID_SIZE = 0.02
USE_GRID_SAMPLE = True
AUGMENT = True

# =====================================================================
# JEPA TRANSFER SETTINGS
# =====================================================================
# Path to a Stage-1 JEPA pretraining checkpoint. Set to None to train the
# backbone from scratch (ablation baseline).
PRETRAINED_CKPT = os.path.join(_ROOT, "exp", "jepa_pretrain", "last.pth")

# Freeze policy for the pretrained backbone:
#   'linear_probe'    - backbone frozen; only heads train. Diagnostic of
#                       representation quality; lowest accuracy ceiling.
#   'full_finetune'   - backbone trainable from epoch 0 at a reduced LR.
#                       Usually the best final accuracy.
#   'staged_unfreeze' - backbone frozen for FREEZE_EPOCHS, then unfrozen at a
#                       reduced LR. Stable and accurate (Point-JEPA recipe).
# DEFAULT = staged_unfreeze.
FREEZE_MODE = "staged_unfreeze"
FREEZE_EPOCHS = 20              # staged_unfreeze: epochs frozen before unfreeze
BACKBONE_LR_SCALE = 0.1         # backbone LR = LEARNING_RATE * this (when unfrozen)

# =====================================================================
# TRAINING HYPERPARAMETERS
# =====================================================================
EPOCHS = 200
BATCH_SIZE = 2
LEARNING_RATE = 1e-3            # head LR
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
CLASS_WEIGHTS = "auto"          # 'auto' or an explicit list of length NUM_CLASSES_SEG
NUM_WORKERS = 0
USE_AMP = True                  # mixed precision (GPU only; ignored on CPU)
SEED = 0
RESUME = False                  # True -> resume from <RESULTS_DIR>/last.pth

# ---------------------------------------------------------------- detection
DETECTION_CONFIG = {
    "POINT_CLOUD_RANGE": [-1000, -1000, -1000, 1000, 1000, 1000],
    "MEAN_SIZE": "auto",
    "GT_EXTRA_WIDTH": [0.2, 0.2, 0.2],
    "LOSS_CONFIG": {
        "LOSS_REG": "weighted-smooth-l1",
        "LOSS_WEIGHTS": {
            "point_cls_weight": 1.0,
            "point_box_weight": 1.0,
            "code_weights": [1.0] * 8,
        },
    },
}
# Multi-task balancing: 'uncertainty' (Kendall 2018) | 'static'
LOSS_BALANCING_METHOD = "uncertainty"
DETECTION_LOSS_WEIGHT = 2.0     # used only when LOSS_BALANCING_METHOD == 'static'


def validate():
    """Validate critical fields; raise on hard errors, warn on risk."""
    errors = []
    if FREEZE_MODE not in ("linear_probe", "full_finetune", "staged_unfreeze"):
        errors.append(f"invalid FREEZE_MODE '{FREEZE_MODE}'")
    if MODEL_VARIANT not in ("nano", "micro", "tiny", "small", "base", "large"):
        errors.append(f"invalid MODEL_VARIANT '{MODEL_VARIANT}'")
    if NUM_CLASSES_SEG <= 0 and NUM_CLASSES_DET <= 0:
        errors.append("at least one of NUM_CLASSES_SEG / NUM_CLASSES_DET must be > 0")
    if NUM_CLASSES_SEG > 0 and len(CLASS_NAMES) != NUM_CLASSES_SEG:
        print(f"[finetune_config] WARNING: CLASS_NAMES has {len(CLASS_NAMES)} "
              f"entries but NUM_CLASSES_SEG={NUM_CLASSES_SEG} — generic names "
              f"will be used in visualisation.")
    if not isinstance(INPUT_CHANNELS, int) or INPUT_CHANNELS < 3:
        errors.append(f"INPUT_CHANNELS must be an int >= 3, got {INPUT_CHANNELS}")
    for name in ("EPOCHS", "BATCH_SIZE", "LEARNING_RATE", "GRID_SIZE"):
        if globals()[name] <= 0:
            errors.append(f"{name} must be > 0")
    if FREEZE_MODE == "staged_unfreeze" and FREEZE_EPOCHS >= EPOCHS:
        print(f"[finetune_config] WARNING: FREEZE_EPOCHS ({FREEZE_EPOCHS}) >= "
              f"EPOCHS ({EPOCHS}) — backbone never unfreezes (= linear_probe).")
    if PRETRAINED_CKPT and not os.path.exists(PRETRAINED_CKPT):
        print(f"[finetune_config] WARNING: PRETRAINED_CKPT not found "
              f"({PRETRAINED_CKPT}) — backbone will train from scratch.")
    if errors:
        raise ValueError("finetune_config errors: " + "; ".join(errors))


validate()
