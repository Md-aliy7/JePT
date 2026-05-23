# JePT — JEPA-pretrained LitePT for point-cloud segmentation + detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)

**JePT** = **JE**PA + Lite**PT**.

A pure-PyTorch point-cloud **segmentation + 3D detection** pipeline that mirrors
[`pyLitePT`](../pyLitePT-main) but obtains its backbone through **JEPA
self-supervised pretraining** instead of supervised training.

> **Why.** In pyLitePT every scene must be fully annotated (per-point labels +
> 3D boxes) before training can start — annotation is the cost bottleneck. JePT
> pretrains the LitePT backbone on **unlabeled** point clouds, so you only need
> to annotate a small fraction (~1–10 %) of your data to fine-tune the task
> heads.

The SSL recipe follows **Point-JEPA** / **I-JEPA**; the backbone is the SOTA
realtime **LitePT** model, reused unchanged. See [REFERENCES.md](REFERENCES.md)
for the scientific sources behind every design decision.

---

## Documentation

| document | contents |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | how JePT works — JEPA design, LitePT backbone, group-pooling, predictor, freeze policy, data flow |
| [TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md) | step-by-step guide — data prep, pretraining, fine-tuning, visualisation, configs, troubleshooting |
| [Custom/README.md](Custom/README.md) | the user layer — every editable script and config |
| [REFERENCES.md](REFERENCES.md) | scientific sources behind every design decision |

## Two-stage workflow

```
            STAGE 1  (no labels)                STAGE 2  (1-10% labels)
  ┌──────────────────────────────┐   ┌────────────────────────────────────┐
  │ unlabeled point clouds        │   │ small labeled set                  │
  │   → group into JEPA tokens    │   │   → LitePTUnifiedCustom            │
  │   → student LitePT (context)  │   │     (backbone + seg + det heads)   │
  │   → EMA teacher  (targets)    │   │   → load pretrained backbone       │
  │   → predictor, smooth-L1 loss │   │   → freeze policy (3 modes)        │
  │   ⇒  pretrained backbone      │──▶│   → fine-tune heads                │
  └──────────────────────────────┘   └────────────────────────────────────┘
     Custom/run_pretrain.py             Custom/run_finetune.py
```

### Stage 1 — JEPA pretraining (`engine/pretrain.py`)
1. Points are bucketed into coarse voxel **groups** — scale-adaptive tokens that
   work for both 100k-point scenes and small objects ([jepa/tokenizer.py](jepa/tokenizer.py)).
2. Groups are spatially ordered ([jepa/sequencer.py](jepa/sequencer.py)) and
   split into contiguous **context** / **target** blocks ([jepa/sampler.py](jepa/sampler.py)).
3. The **EMA teacher** (a frozen moving-average LitePT) encodes the *full* cloud →
   per-group **target latents**.
4. The **student** LitePT encodes only the **context** points → context latents.
5. A small **predictor** transformer predicts the target latents from context
   latents + target 3D positions; **smooth-L1** loss in latent space.

Design note: JePT runs JEPA over *group-pooled tokens* with LitePT used as a
**black box** (`enc_mode=False`, full encoder+decoder). This refines the
approved plan, which proposed slicing LitePT's internal voxel stages — that
proved fragile (it entangles GridPooling, sparse-conv indice keys and
re-serialization). The black-box approach pretrains the **entire** backbone and
the resulting encoder state-dict drops into the downstream model with **zero
key remapping** (verified: 136/136 tensors, exact key match). Downstream
transfer uses the **EMA teacher** (target encoder), per the I-JEPA / DINO
convention — a smoother, better downstream initialisation than the student.

### Stage 2 — downstream fine-tuning (`engine/finetune.py`)
The JEPA-pretrained backbone is loaded into pyLitePT's `LitePTUnifiedCustom` /
`LitePTDualPathUnified` and fine-tuned with the original segmentation
(`nn.Linear`) and detection (`PointHeadBox`) heads. Three **freeze policies** are
supported (see [engine/freeze.py](engine/freeze.py) and REFERENCES.md):

| `FREEZE_MODE`     | Backbone           | When to use |
|-------------------|--------------------|-------------|
| `linear_probe`    | frozen             | diagnose representation quality; tiniest label budgets |
| `full_finetune`   | trainable, low LR  | best final accuracy; usually needed for detection |
| `staged_unfreeze` | frozen N epochs, then trainable | **default** — stable compromise (Point-JEPA recipe) |

---

## Layout

```
JePT/
├── Custom/      USER LAYER — configs, run scripts, data tools, demo  (edit here)
│   ├── pretrain_config.py / finetune_config.py
│   ├── run_pretrain.py / run_finetune.py / smoke_test.py
│   ├── run_shapes3d_demo.py / run_real_demo.py / run_lowlabel_ablation.py
│   ├── shapes3d_generator.py / make_synthetic_dataset.py / prepare_unlabeled.py
│   └── postprocess.py / visualize.py / README.md
│
├── jepa/        CORE — JEPA SSL: tokenizer, sequencer, sampler, predictor, EMA, model, loss
├── engine/      CORE — pretrain / finetune / evaluate loops, freeze controller, checkpointing
├── data/        CORE — unlabeled + labeled datasets, collation, scale-aware augmentation
├── models/      LitePT backbone (from pyLitePT) + unified.py (heads + weight loader)
├── pcdet_lite/  detection head (PointHeadBox) — from pyLitePT
├── labelCloud/  3D annotation GUI — from pyLitePT (exports JePT-compatible NPY)
├── backend_cpu/, libs/, metrics/, utils/, hybrid_backend.py   (from pyLitePT)
├── REFERENCES.md
└── README.md
```

**`Custom/` is the only folder you edit** — it groups every user-facing
process (configs, entry points, data tools), exactly like pyLitePT's `Custom/`,
so the core (`jepa/`, `engine/`, `models/`, `data/`) is never touched for
customization. See [Custom/README.md](Custom/README.md).

`models/`, `pcdet_lite/`, `libs/`, `backend_cpu/`, `metrics/`, `utils/`,
`labelCloud/` and `hybrid_backend.py` are reused **verbatim** from pyLitePT.
`models/unified.py` is pyLitePT's `Custom/core.py` plus `load_pretrained_backbone()`.

## Usage

```bash
# from the JePT/ directory

# 0. (optional) inventory your unlabeled data
python Custom/prepare_unlabeled.py /path/to/clouds

# 1. pretrain the backbone on unlabeled data
#    edit Custom/pretrain_config.py: UNLABELED_DATA_PATH, MODEL_VARIANT
python -m Custom.run_pretrain

# 2. fine-tune on your small labeled set
#    edit Custom/finetune_config.py: DATA_PATH, PRETRAINED_CKPT, FREEZE_MODE
python -m Custom.run_finetune

# end-to-end correctness check on tiny synthetic data (CPU, ~1 min)
python -m Custom.smoke_test

# recommended demo: generate Shapes3D, JEPA-pretrain, fine-tune (pretrained vs
# scratch) on a small labelled subset, evaluate on a held-out unseen test split
python -m Custom.run_shapes3d_demo
```

### Data format
Both stages accept pyLitePT's format. **Unlabeled** (Stage 1): NPY folders with
`coord.npy` (+ optional `color.npy` / `normal.npy`) and/or `.ply` files — no
labels. **Labeled** (Stage 2): NPY folders additionally with `segment.npy` and
`gt_boxes.npy` (`[x,y,z,dx,dy,dz,heading,label]`), under `train/` and `val/`.

## Status & results

`python -m Custom.smoke_test` exercises the whole pipeline (tokenisation, JEPA
pretraining, weight loading, all three freeze modes, seg+det inference) — passes.

**`python -m Custom.run_shapes3d_demo`** — the recommended end-to-end demo.
Shapes3D = a flat floor plane plus 3–5 floating solids (cube/sphere/cylinder/
cone/pyramid/torus); 7 segmentation classes (6 shapes + floor), 6 detection
classes (the shapes only — the floor plane gets no box). Point **colour is
randomised per shape instance and carries no class signal**, so only *geometry*
determines the label — the heads cannot cheat by reading the colour channel.

The demo runs the scientifically correct SSL regime:

- JEPA-pretrain the backbone on a **large unlabeled pool** (`N_TRAIN`, default 200);
- fine-tune the seg+det heads on a **labelled subset** (`N_LABELED`, default 80)
  — **twice**: JEPA-pretrained vs. from-scratch;
- evaluate both on a **disjoint held-out test split** never seen in either stage.
- the model uses **dual-path unified** (`USE_DUAL_PATH_UNIFIED=True`) — pyLitePT's
  best-performance recipe: multi-stage backbone for segmentation, single-stage
  (no downsampling) for detection so small objects are not downsampled away.

The test-set gap between the two arms is a clean, controlled measurement of what
self-supervised pretraining bought. The script prints per-class seg IoU,
detection recall/precision, and the JEPA gain (Δ). `target_std` is reported to
confirm no representation collapse during pretraining.

`python -m Custom.run_lowlabel_ablation` — sweeps the labelled-scene budget and
reports JEPA-pretrained vs. from-scratch unseen-test mIoU at each point: the
JEPA gain is largest when labels are scarcest.

> Result numbers are intentionally not pinned here — every demo regenerates its
> dataset and prints fresh metrics. Run the demo to see current numbers for your
> machine. The scientific claim being demonstrated is the **pretrained > scratch
> gap on the held-out test set**, not any single absolute figure.

## Validity & scope — what these results do and do NOT show

Being explicit so nothing here is over-read:

- **Verified:** the pipeline is mechanically correct — JEPA pretraining runs,
  the loss decreases, representations do not collapse (`target_std` stays
  ≈0.5), pretrained weights transfer (136/136 tensors), and the pretrained
  backbone beats from-scratch training in the low-label regime. All numbers
  above are from real runs in this repo, not estimates.
- **Not yet verified:** representation quality on *real-world semantic*
  benchmarks. Every demo here uses **synthetic, geometry-defined data**
  (Shapes3D, procedural rooms) where the label is determined by shape alone.
  Such tasks are solvable from geometry, so they confirm *transfer and
  low-label benefit* but do **not** prove JePT learns semantics beyond
  geometry.
- **Geometric shortcut (known limitation).** Sonata (CVPR 2025, arXiv:2503.16429)
  shows point-cloud SSL can exploit raw geometry instead of learning semantics
  — because coordinates are fed as input features. JePT follows the published
  **Point-JEPA** recipe (latent-space prediction, EMA teacher, block masking),
  which mitigates this to a workable degree, but JePT does **not** add Sonata's
  stronger mitigations and is not validated against the shortcut. For
  real-scan deployment, validate on real annotated data.

JePT's reused Pointcept components (`Point` structure, z-order / Hilbert
serialization, `PointModule`) were diff-checked against Sonata's released code
and are **identical** to the canonical implementation.

Inspect predictions (interactive PySide6 + VisPy GUI — GT vs prediction,
scene navigator, detection toggle, per-class IoU legend):
`python Custom/visualize.py`  (add `--save` for headless PLY export).

## Known limitations (v1)
- **Synthetic-only validation / geometric shortcut** — see *Validity & scope*
  above. Real-scan semantic validation is future work.
- **Detection — single-path vs dual-path trade-off** (pick consciously):
  - `USE_DUAL_PATH_UNIFIED=False` — one multi-stage backbone shared by seg + det;
    the JEPA-pretrained backbone transfers **fully** (136/136 tensors) to both
    tasks. Detection sees downsampled features (worse for small objects).
  - `USE_DUAL_PATH_UNIFIED=True` (pyLitePT's "best performance" recipe — used
    by `run_shapes3d_demo`) — multi-stage backbone for seg, **single-stage**
    backbone for det (no downsampling, high-resolution features → better small-
    object detection). The multi-stage pretrain only **partially** transfers to
    the det branch (input embedding only, reported as `1/N PARTIAL`); the rest
    of the det branch is trained fresh. A separate single-stage JEPA pretrain
    would close the gap — future work.
  The detection head itself (`PointHeadBox`) is always trained from scratch.
- A realistic labeled fraction for detection is ~5–10 %, not 1 % — detection
  transfers less readily than segmentation/classification.
- **`MODEL_VARIANT`, `GRID_SIZE` and `INPUT_CHANNELS` must be identical in the
  pretrain and finetune configs.** The backbone's sparse-conv + serialization
  depend on the voxel grid, so a stage mismatch silently degrades transfer.
  `load_pretrained_backbone` prints what the checkpoint was pretrained with so a
  mismatch is visible.

---

## License & attribution

JePT is released under the [MIT License](LICENSE).

The repository bundles and builds on several open-source projects, all under
permissive licenses (MIT). Full credit:

| Project | Role in JePT | Authors / copyright |
|---|---|---|
| [**LitePT**](https://github.com/prs-eth/LitePT) ([paper](https://arxiv.org/abs/2512.13689)) | Backbone — reused verbatim in `models/litept/` | Yuanwen Yue, Damien Robert, Jianyuan Wang, Sunghwan Hong, Jan Dirk Wegner, Christian Rupprecht, Konrad Schindler (ETH Zurich / Oxford / UZH). MIT, © Photogrammetry and Remote Sensing Lab. |
| **pyLitePT** | Detection head, data loader, post-processing, CPU backend — reused in `pcdet_lite/`, `models/detection.py`, `models/modules.py`, `backend_cpu/`, `libs/`, `metrics/`, `utils/`, `hybrid_backend.py` | MIT |
| **Point-JEPA** ([paper](https://arxiv.org/abs/2404.16432)) | Self-supervised recipe for point clouds; `jepa/ema.py` adapted from the reference implementation | Project authors (paper): Ayumu Saito, Prachi Kudeshia, Jiju Poovvancheri. Reference-implementation code MIT. |

Scientific references (algorithm design): [I-JEPA](https://arxiv.org/abs/2301.08243),
[DINOv3](https://ai.meta.com/dinov3), [Sonata](https://arxiv.org/abs/2503.16429),
[Concerto](https://arxiv.org/abs/2510.23607). See [REFERENCES.md](REFERENCES.md).
