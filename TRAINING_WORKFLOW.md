# JePT — Training Workflow

A practical, step-by-step guide to training JePT on your own data. For the
design rationale see [ARCHITECTURE.md](ARCHITECTURE.md).

> **Run everything from the `JePT/` directory.** Only files in `Custom/` are
> meant to be edited — the core (`jepa/`, `engine/`, `models/`, `data/`) stays
> untouched from dataset to dataset.

---

## 0. Install

```bash
pip install -r requirements.txt
```
Core: `torch`, `numpy`, `timm`, `addict`, `scipy`. GUI: `PySide6`, `vispy`,
`open3d`. CUDA libraries (`spconv`, `flash-attn`) are optional — JePT falls
back to pure-Python CPU implementations automatically.

**Verify the install** (≈1 min, CPU, synthetic data):
```bash
python -m Custom.smoke_test
```
Expected: `ALL SMOKE TESTS PASSED`.

---

## 1. Prepare data

JePT uses the **labelCloud / pyLitePT NPY-folder format**. One scene = one
folder:

```
my_data/
├── train/
│   ├── scene_0000/
│   │   ├── coord.npy     # (N,3) float32  — REQUIRED
│   │   ├── color.npy     # (N,3) float32 RGB in [0,1] or [0,255]  — optional
│   │   ├── segment.npy   # (N,)  int      — per-point label (labeled data only)
│   │   └── gt_boxes.npy  # (M,8) float32  [x,y,z,dx,dy,dz,heading,label]
│   └── ...
├── val/   ...
└── test/  ...
```

- **Stage 1 (pretraining)** reads only `coord.npy` (+ optional `color.npy`) —
  **no labels needed**. Point it at *all* your scans.
- **Stage 2 (fine-tuning)** needs `segment.npy` / `gt_boxes.npy` — only on the
  small labeled subset.

**Annotating** — the bundled `labelCloud/` GUI exports this format directly
(set `data_format = npy_folder` in its `config.ini`). Annotate only ~1–10 % of
your scenes.

**No data yet?** Generate a realistic dataset:
```bash
python Custom/shapes3d_generator.py --output data_shapes3d --train 200 --val 30 --test 30
```

---

## 2. Stage 1 — JEPA pretraining (no labels)

**Edit `Custom/pretrain_config.py`:**

| field | meaning |
|---|---|
| `UNLABELED_DATA_PATH` | folder of raw clouds (NPY folders / PLY) |
| `MODEL_VARIANT` | `nano`/`micro`/`tiny`/`small`/`base`/`large` — **must match Stage 2** |
| `INPUT_CHANNELS` | 3 (coord) or 6 (coord+colour) |
| `GRID_SIZE` | input voxel size — set to your data's scale |
| `GROUP_SIZE` | JEPA token scale; `"auto"` calibrates from data extent |
| `EPOCHS`, `BATCH_SIZE`, `LR` | optimiser settings |
| `RESUME` | `True` resumes from `<RESULTS_DIR>/last.pth` after an interruption |

**Run:**
```bash
python -m Custom.run_pretrain
```

**What to watch** (printed per epoch):
- `loss` — should **decrease** and plateau.
- `target_std` — must stay clearly **> 0** (collapse = it vanishes toward 0).
- `pred_std` — likewise non-zero.

Output: `RESULTS_DIR/last.pth` (+ periodic `epoch_*.pth`). The `student`
state-dict inside is the pretrained backbone.

---

## 3. Stage 2 — supervised fine-tuning (few labels)

**Edit `Custom/finetune_config.py`:**

| field | meaning |
|---|---|
| `DATA_PATH` | labeled dataset root (`train/`, `val/`, `test/`) |
| `CLASS_NAMES`, `NUM_CLASSES_SEG`, `NUM_CLASSES_DET` | your classes |
| `MODEL_VARIANT` | **must equal the pretraining variant** |
| `PRETRAINED_CKPT` | path to the Stage-1 checkpoint (`None` = train from scratch) |
| `FREEZE_MODE` | `linear_probe` / `full_finetune` / `staged_unfreeze` (default) |
| `FREEZE_EPOCHS` | staged mode: epochs frozen before unfreezing |
| `BACKBONE_LR_SCALE` | backbone LR = `LEARNING_RATE × this` |
| `USE_DUAL_PATH_UNIFIED` | `False` = single multi-stage backbone shared by seg+det (FULL JEPA transfer to both heads). `True` = pyLitePT's best-perf recipe: multi-stage for seg, single-stage for det (better small-object detection; det branch only gets PARTIAL transfer — its input embedding — and trains the rest fresh). |
| `SEED` | Python / NumPy / torch (CPU+CUDA) seed for reproducible runs. cuDNN set deterministic. |
| `RESUME` | `True` resumes model weights + epoch from `<RESULTS_DIR>/last.pth` |

**Freeze-mode guidance:**
- very few labels / quick check → `linear_probe`
- best accuracy, detection → `full_finetune`
- balanced default → `staged_unfreeze`

**Run:**
```bash
python -m Custom.run_finetune
```

**What to watch**:
- `seg_loss` / `det_loss` decrease per epoch.
- `val_seg_acc` increases (segmentation accuracy on the val split).
- `val_det_mAP50` increases (**real 3D-IoU mAP@0.5** — the same metric pyLitePT
  uses, so JePT detection numbers are directly comparable).

Output: `RESULTS_DIR/best.pth` (best **combined `val_seg_acc + val_det_mAP50`**
— so a checkpoint is not frozen while detection is still converging) and
`last.pth`. Checkpoints store `meta` (variant, class names, channels, grid
size, dual-path flag, data path) so downstream tools rebuild the exact model
without manual config editing.

---

## 4. Visualise predictions

Interactive GUI (PySide6 + VisPy — dual Ground-Truth ┃ Prediction canvases,
scene navigator, detection controls, per-class IoU legend):
```bash
python Custom/visualize.py                              # auto-discovers ckpt + data
python Custom/visualize.py --checkpoint <best.pth> --split test
python Custom/visualize.py --save                       # headless: writes coloured PLYs
```

---

## 5. The bundled demos

| command | what it does |
|---|---|
| `python -m Custom.smoke_test` | end-to-end correctness check on tiny synthetic data (~1 min) |
| `python -m Custom.run_shapes3d_demo` | **recommended** — real SSL → supervised run on Shapes3D (200 unlabeled / 80 labelled / 30 unseen test, dual-path), prints pretrained-vs-scratch test mIoU + mAP@{0.25, 0.5, 0.75} |
| `python -m Custom.run_real_demo` | same idea on procedural indoor scenes |
| `python -m Custom.run_lowlabel_ablation` | pretrained vs scratch as the labeled-scene budget shrinks |

Append `--quick` to `run_shapes3d_demo` / `run_real_demo` for a fast dry run.

---

## 6. Tips for your own data

- **Scale.** Set `GRID_SIZE` to your data's units (≈ 1–3 % of object size).
  Leave `GROUP_SIZE = "auto"`.
- **Channels.** `INPUT_CHANNELS = 3` if you only have XYZ, `6` with RGB. It must
  match the data; `finetune()` errors clearly if it does not.
- **Stage consistency.** `MODEL_VARIANT`, `GRID_SIZE` and `INPUT_CHANNELS`
  **must be identical** in `pretrain_config.py` and `finetune_config.py` — the
  pretrained backbone is only valid on the same voxel grid and channel count it
  was trained on. `load_pretrained_backbone` prints the checkpoint's pretraining
  values so a mismatch is caught.
- **Label budget.** Detection transfers less readily than segmentation — budget
  ~5–10 % labeled scenes for good detection, less for segmentation only.
- **Pretrain on everything.** Stage 1 ignores labels, so point it at your full
  scan collection — more unlabeled data → better representations.
- **Single-path vs dual-path.** Set `USE_DUAL_PATH_UNIFIED=True` (pyLitePT's
  best-perf recipe) for tasks where small-object detection matters — the
  detection branch is single-stage and keeps full spatial resolution. Trade-off:
  the JEPA-pretrained multi-stage backbone only **partially** transfers to the
  det branch (input embedding only). Set `False` for full JEPA transfer to both
  heads (one shared multi-stage backbone) at the cost of downsampled detection
  features.
- **Reproducibility.** `SEED` (default 0) seeds Python/NumPy/torch and sets
  cuDNN deterministic — two runs with the same seed and data should match
  bitwise on CPU and modulo non-determinism on GPU.

---

## 7. Troubleshooting

| symptom | fix |
|---|---|
| `No unlabeled clouds found` | check `UNLABELED_DATA_PATH`; folders need `coord.npy` |
| `data provides K channels but INPUT_CHANNELS=…` | set `INPUT_CHANNELS = K` |
| pretraining `target_std` → 0 | representation collapse — lower `LR`, check `GROUP_SIZE` |
| `load_pretrained_backbone matched 0 tensors` | `MODEL_VARIANT` differs between configs |
| weight-shape mismatch on visualise | checkpoint trained with different class counts — meta handles new ckpts; retrain old ones |
| GUI will not open | runs headless? use `python Custom/visualize.py --save` |
| detection conf looks binary (0 or 1) | expected — the head is a per-point objectness classifier; a converged sigmoid saturates. Use the **NMS** slider, not conf, as the main control |
| boxes missing for some objects | raise the **NMS** IoU slider; lower **conf**. NMS is per-class, so cross-class overlap no longer drops boxes — if still missing, the object needs more labeled training scenes |

Validation runs automatically when a config is imported and reports problems
with clear messages — read the first error printed.
