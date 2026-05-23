"""Entry point — Stage 1 JEPA pretraining.

    python -m Custom.run_pretrain        (run from the JePT/ directory)
    python Custom/run_pretrain.py

Edit Custom/pretrain_config.py to point at your unlabeled data.
"""

import os
import sys

# make the JePT package root importable regardless of how this is launched
_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# backends must be configured BEFORE any LitePT import (spconv / flash-attn)
from hybrid_backend import setup_backends

setup_backends(verbose=True)

from engine.pretrain import pretrain  # noqa: E402
import Custom.pretrain_config as cfg  # noqa: E402


def main():
    print("=" * 60)
    print("JePT — JEPA pretraining")
    print(f"  variant={cfg.MODEL_VARIANT}  data={cfg.UNLABELED_DATA_PATH}")
    print("=" * 60)
    pretrain(cfg)
    print(f"Done. Checkpoints in {cfg.RESULTS_DIR}")


if __name__ == "__main__":
    main()
