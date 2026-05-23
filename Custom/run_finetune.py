"""Entry point — Stage 2 downstream fine-tuning (segmentation + detection).

    python -m Custom.run_finetune        (run from the JePT/ directory)
    python Custom/run_finetune.py

Edit Custom/finetune_config.py — set PRETRAINED_CKPT and FREEZE_MODE.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from hybrid_backend import setup_backends

setup_backends(verbose=True)

from engine.finetune import finetune  # noqa: E402
import Custom.finetune_config as cfg  # noqa: E402


def main():
    print("=" * 60)
    print("JePT — downstream fine-tuning")
    print(f"  variant={cfg.MODEL_VARIANT}  freeze_mode={cfg.FREEZE_MODE}")
    print(f"  pretrained={cfg.PRETRAINED_CKPT}")
    print("=" * 60)
    finetune(cfg)
    print(f"Done. Checkpoints in {cfg.RESULTS_DIR}")


if __name__ == "__main__":
    main()
