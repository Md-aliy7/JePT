"""JEPA pretraining configuration (Stage 1).

Plain Python module — edit the values below, then run:
    python -m Custom.run_pretrain

This file is part of the editable `Custom/` layer: change it per dataset; the
core (jepa/, engine/, models/) is never touched.
"""

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# ---------------------------------------------------------------- data
# Folder of UNLABELED point clouds (NPY scene folders and/or .ply files).
# May contain train/ and val/ sub-folders.
UNLABELED_DATA_PATH = os.path.join(_ROOT, "data_unlabeled")
RESULTS_DIR = os.path.join(_ROOT, "exp", "jepa_pretrain")

# ---------------------------------------------------------------- model
MODEL_VARIANT = "nano"          # nano|micro|tiny|small|base|large
INPUT_CHANNELS = 6              # coord(3) + colour(3)
GRID_SIZE = 0.02                # voxel size for input down-sampling
USE_GRID_SAMPLE = True
AUGMENT = True

# ---------------------------------------------------------------- JEPA
# GROUP_SIZE = coarse voxel edge -> one JEPA token per group.
# 'auto' calibrates it from the data's spatial extent — robust for any scale
# (small objects or large scenes). Set a float to override.
GROUP_SIZE = "auto"
MIN_TOKENS = 8                  # skip clouds with fewer groups than this
MAX_TOKENS = 2048               # cap groups/sample (bounds predictor cost)
NUM_TARGET_BLOCKS = 4           # masked target blocks per cloud
TARGET_RATIO = (0.15, 0.20)     # fraction of groups per target block
PREDICTOR_DIM = 192             # narrow predictor transformer width
PREDICTOR_DEPTH = 6
PREDICTOR_HEADS = 6
LOSS_BETA = 2.0                 # smooth-L1 beta (Point-JEPA default)

# ------------------------------------------------------------- optimiser
EPOCHS = 300
BATCH_SIZE = 8
LR = 1.5e-3
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 30
ETA_MIN = 1e-6                  # final cosine LR
GRAD_CLIP = 1.0
USE_AMP = True                  # mixed precision (GPU only; ignored on CPU)
NUM_WORKERS = 0
SEED = 0

# ------------------------------------------------------------------ EMA
EMA_TAU_MIN = 0.9998
EMA_TAU_MAX = 0.99999
EMA_TAU_EPOCHS = 200            # epochs over which tau ramps min -> max

# ------------------------------------------------------------ checkpoint
SAVE_EVERY = 10                 # save a checkpoint every N epochs
RESUME = False                  # True -> resume from <RESULTS_DIR>/last.pth
                                # if it exists (set True after an interruption)


def validate():
    """Sanity-check the configuration; raise on hard errors, warn on risk."""
    errors = []
    if MODEL_VARIANT not in ("nano", "micro", "tiny", "small", "base", "large"):
        errors.append(f"invalid MODEL_VARIANT '{MODEL_VARIANT}'")
    if not isinstance(INPUT_CHANNELS, int) or INPUT_CHANNELS < 3:
        errors.append(f"INPUT_CHANNELS must be an int >= 3, got {INPUT_CHANNELS}")
    if PREDICTOR_DIM % PREDICTOR_HEADS != 0:
        errors.append(f"PREDICTOR_DIM ({PREDICTOR_DIM}) must be divisible by "
                      f"PREDICTOR_HEADS ({PREDICTOR_HEADS})")
    if not (GROUP_SIZE == "auto" or
            (isinstance(GROUP_SIZE, (int, float)) and GROUP_SIZE > 0)):
        errors.append("GROUP_SIZE must be 'auto' or a positive number")
    for name in ("EPOCHS", "BATCH_SIZE", "LR"):
        if globals()[name] <= 0:
            errors.append(f"{name} must be > 0")
    if NUM_TARGET_BLOCKS < 1:
        errors.append("NUM_TARGET_BLOCKS must be >= 1")
    if not os.path.isdir(UNLABELED_DATA_PATH):
        print(f"[pretrain_config] WARNING: UNLABELED_DATA_PATH does not exist "
              f"({UNLABELED_DATA_PATH}) — set it before running.")
    if errors:
        raise ValueError("pretrain_config errors: " + "; ".join(errors))


validate()
