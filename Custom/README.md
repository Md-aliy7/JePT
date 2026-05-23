# `Custom/` — the JePT user layer

Everything you edit or run lives here. The **core** (`jepa/`, `engine/`,
`models/`, `data/`) is never touched for customization — mirrors how
pyLitePT keeps user code in its own `Custom/` folder.

## Files

| File | Purpose |
|------|---------|
| `pretrain_config.py`        | Stage-1 JEPA pretraining settings — **edit this** |
| `finetune_config.py`        | Stage-2 fine-tuning settings (`PRETRAINED_CKPT`, `FREEZE_MODE`) — **edit this** |
| `run_pretrain.py`           | Run Stage 1 — JEPA pretraining on unlabeled data |
| `run_finetune.py`           | Run Stage 2 — seg + det fine-tuning on labeled data |
| `run_shapes3d_demo.py`      | **Recommended real demo** — Shapes3D dataset: pretrain (no labels) → fine-tune on few labels → evaluate on unseen test |
| `run_real_demo.py`          | Full demo on procedural indoor scenes (pretrained vs scratch → unseen test) |
| `run_lowlabel_ablation.py`  | Low-label ablation — pretrained vs scratch as the labeled-scene budget shrinks (JePT's core use case) |
| `visualize.py`              | Interactive PySide6 + VisPy GUI — GT vs prediction, scene navigator, detection controls, per-class IoU legend (`--save` for headless PLY export) |
| `postprocess.py`            | Detection post-processing — confidence filter + per-class BEV NMS (shared by `visualize.py` and evaluation) |
| `smoke_test.py`             | Fast end-to-end correctness check on tiny synthetic data |
| `shapes3d_generator.py`     | Shapes3D generator — 6 solids + floor plane, NPY-folder format; class-independent random colours |
| `make_synthetic_dataset.py` | Generate structured indoor scenes (labelCloud / pyLitePT NPY format) |
| `prepare_unlabeled.py`      | Inventory a folder of unlabeled clouds |

## Typical workflow

```bash
# run everything from the JePT/ directory

# (A) verify the install
python -m Custom.smoke_test

# (B) full learning demo — Shapes3D (recommended): SSL pretrain -> fine-tune
python -m Custom.run_shapes3d_demo

# (C) your own data
#  1. edit Custom/pretrain_config.py  -> UNLABELED_DATA_PATH
python -m Custom.run_pretrain
#  2. edit Custom/finetune_config.py  -> DATA_PATH, PRETRAINED_CKPT, FREEZE_MODE
python -m Custom.run_finetune
```

## Annotating your own data — labelCloud

`JePT/labelCloud/` is the included 3D annotation GUI. Export with
`data_format = npy_folder` (its `config.ini` default) — it writes exactly the
`coord.npy / color.npy / segment.npy / gt_boxes.npy` layout that
`finetune_config.py` consumes. Annotate only ~1–10 % of your scenes; the rest
stay unlabeled for Stage-1 pretraining.

> Only `Custom/` files need editing. To change model architecture or the JEPA
> algorithm itself, see the core packages — but that is not "customization".
