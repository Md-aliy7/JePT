# JePT — Architecture Guide

A technical deep-dive into how JePT works. For a practical step-by-step guide
see [TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md); for the science behind each
choice see [REFERENCES.md](REFERENCES.md).

---

## 1. What JePT is

JePT = **JE**PA + Lite**PT**. It is the [pyLitePT](../pyLitePT-main) point-cloud
**segmentation + 3D detection** pipeline, with one change: the LitePT backbone
is obtained by **JEPA self-supervised pretraining** on *unlabeled* point clouds
instead of supervised training.

**The problem.** In pyLitePT every scene must be fully annotated (per-point
labels + 3D boxes) before training can start. Annotation is the cost
bottleneck.

**The solution.** JEPA (Joint-Embedding Predictive Architecture) learns useful
representations with **no labels**. JePT pretrains the backbone on all your raw
scans, then fine-tunes the lightweight task heads on a small labeled fraction
(~1–10 %). Empirically the pretrained backbone reaches far higher accuracy than
training from scratch when labels are scarce.

```
 STAGE 1  — self-supervised, NO labels        STAGE 2 — supervised, FEW labels
 ┌───────────────────────────────────┐        ┌────────────────────────────────┐
 │ raw point clouds                  │        │ small labeled set               │
 │  → group into JEPA tokens         │        │  → LitePTUnifiedCustom          │
 │  → student LitePT  (context)      │        │     (backbone + seg + det heads)│
 │  → EMA teacher LitePT (targets)   │  ───▶  │  → load pretrained backbone     │
 │  → predictor, smooth-L1 latent loss│       │  → freeze policy (3 modes)      │
 │  ⇒ pretrained backbone weights    │        │  → train heads (+ backbone)     │
 └───────────────────────────────────┘        └────────────────────────────────┘
       engine/pretrain.py                            engine/finetune.py
```

---

## 2. The LitePT backbone (reused, unchanged)

LitePT is a realtime hierarchical point transformer (Point-Transformer-V3
lineage). JePT reuses it **verbatim** from pyLitePT — `models/litept/litept.py`.

- **Input**: a `Point` structure — `coord (N,3)`, `feat (N,C)`, per-point
  `batch` index, `grid_size`.
- **Encoder**: 5 stages. Stages 0–2 use sparse convolution (`SubMConv3d`);
  stages 3–4 use serialized rotary attention (`PointROPEAttention`). `GridPooling`
  downsamples between stages.
- **Decoder**: `GridUnpooling` upsamples back to the input point resolution.
- **Output** (`enc_mode=False`): a per-input-point feature tensor `(N, feat_dim)`.

JePT treats LitePT as a **black box**: it never edits the architecture. Both the
JEPA student/teacher and the downstream model are ordinary `LitePT` instances,
so a pretrained backbone's `state_dict` drops into the downstream model with
zero key remapping.

---

## 3. Stage 1 — JEPA pretraining

`jepa/` + `engine/pretrain.py`. Orchestrated by `jepa/jepa_model.py:JePTModel`.

### 3.1 Tokenisation — group pooling (`jepa/tokenizer.py`)

Point-JEPA tokenises a *fixed-size* cloud with FPS + KNN. That breaks for mixed
object/scene scale. JePT instead defines a **token = a coarse voxel group**:
points are bucketed by a coarse grid of edge `GROUP_SIZE`. Grid bucketing is
scale-adaptive and purely geometric, so the student and the EMA teacher always
agree on the token set.

`GROUP_SIZE` is the one data-dependent knob; with `GROUP_SIZE = "auto"` it is
calibrated from the median scene extent (`engine/pretrain.py:calibrate_group_size`).

### 3.2 Sequencing + context/target split (`sequencer.py`, `sampler.py`)

- **Sequencer** — orders the group tokens with a greedy nearest-neighbour walk
  so contiguous index ranges = spatially contiguous regions (Point-JEPA recipe).
- **Sampler** — draws `NUM_TARGET_BLOCKS` contiguous *target* blocks; the
  *context* is every group not in a target block. Block masking (not random)
  forces the model to infer whole coherent regions.

### 3.3 Student / Teacher / Predictor

| component | what it is | trained? |
|---|---|---|
| **student** | a `LitePT(enc_mode=False)` — the backbone we keep | yes (gradient) |
| **teacher** | EMA copy of the student (`jepa/ema.py`) | no — EMA only |
| **predictor** | a small transformer (`jepa/predictor.py`) | yes (gradient) |

Forward pass (`JePTModel.forward`):

1. **Teacher** encodes the **full** cloud → per-point features → pooled per
   group → `target` token latents (layer-normalised, à la data2vec).
2. **Student** encodes only the **context** points → pooled per group →
   `context` token latents.
3. **Predictor** takes the context tokens + the 3D positions of the target
   groups and predicts the target latents.
4. **Loss** = smooth-L1 between predicted and teacher target latents — a pure
   latent-space objective (no point/colour reconstruction).

The EMA teacher tracks the student with momentum `tau` ramped 0.9998→0.99999,
updated after every optimiser step. This is what prevents representation
collapse (BYOL / data2vec).

### 3.4 Design note — group pooling vs. internal slicing

The original plan proposed slicing LitePT's internal voxel stages for JEPA.
That entangles `GridPooling`, sparse-conv indice keys and re-serialisation and
is fragile. The group-pooling design treats LitePT as a black box: it pretrains
the **entire** encoder + decoder and transfers with zero key remapping
(verified: 136/136 tensors).

### 3.5 Optimisation

AdamW over student + predictor (teacher excluded), linear-warmup → cosine LR
(plain `LambdaLR`, no PyTorch-Lightning), gradient clipping, optional AMP.
Weight decay is excluded from 1-D parameters (LayerNorm / bias) — standard
transformer-SSL practice (I-JEPA / DINO / Point-JEPA). Collapse monitors
(`target_std`, `pred_std`) are logged each epoch.

### 3.6 Known limitation — the geometric shortcut

Point coordinates are part of the input features, and the predictor receives
the target groups' 3D positions. As Sonata (CVPR 2025) shows, this lets
point-cloud SSL partly *shortcut* — predicting from raw geometry instead of
learning semantics. JePT follows the published **Point-JEPA** recipe
(latent-space prediction against an EMA teacher, contiguous block masking),
which mitigates this to a workable degree, but it does not add Sonata's
stronger mitigations. Consequently the synthetic, geometry-defined demos in
this repo verify *pipeline correctness, transfer and low-label benefit* — not
semantic representation quality on real scans. See README → *Validity & scope*.

---

## 4. Stage 2 — downstream fine-tuning

`engine/finetune.py` + `models/unified.py`.

### 4.1 Weight transfer (`models/unified.py:load_pretrained_backbone`)

The pretraining checkpoint stores both encoders (`student` and the EMA
`teacher`). Downstream transfer uses the **EMA teacher** (target encoder) — the
I-JEPA / DINO / data2vec convention: the moving-average encoder is a
weight-space ensemble, smoother and a consistently better downstream
initialisation than the gradient-trained student. Its keys (`embedding.*`,
`enc.*`, `dec.*`) match the `LitePT` inside the downstream model exactly (136/136
tensors, verified). Loading is shape-checked and reported (`N/N tensors loaded
[FULL]`), so a silent no-op is impossible; the checkpoint's pretraining
`variant`/`grid_size`/`input_channels` are printed so a stage mismatch is
caught. The seg/det heads stay randomly initialised and are trained fresh.

### 4.2 Freeze policy (`engine/freeze.py:FreezeController`)

Three modes — the literature does not endorse one, so all are supported:

| `FREEZE_MODE` | backbone | use when |
|---|---|---|
| `linear_probe` | frozen (`.eval()`, BN stats frozen) | diagnostic; tiniest label budgets |
| `full_finetune` | trainable, reduced LR | best final accuracy; needed for detection |
| `staged_unfreeze` *(default)* | frozen `FREEZE_EPOCHS` epochs, then unfrozen | stable compromise (Point-JEPA recipe) |

The controller owns optimiser (re)construction because the trainable parameter
set changes when the backbone is (un)frozen; backbone parameters always get a
reduced LR (`base_lr * BACKBONE_LR_SCALE`), and weight decay is excluded from
1-D params. Fine-tuning also applies a warm-up → cosine LR schedule as a
stateless per-epoch factor, so it survives the optimiser being rebuilt at the
unfreeze epoch (a stateful torch scheduler would not).

### 4.3 Heads + multi-task loss

Reused from pyLitePT, **no new head code**:
- **segmentation** — `nn.Linear` → per-point logits; weighted cross-entropy.
- **detection** — `pcdet_lite/PointHeadBox` → per-point box regression + class.

The two losses are fused with learnable uncertainty weighting (Kendall et al.
2018, `model.log_vars`) or a static weight.

`PointHeadBox` is a **per-point** detector: every point regresses a box and a
class. Box regression is a residual from a per-class **mean-size anchor**
(`calculate_mean_sizes`, indexed by the 0-based class id). The classification
score is `sigmoid(max class logit)` — a point-objectness score that saturates
toward 0/1 once trained, which is expected for this head.

### 4.4 Detection inference — post-processing (`Custom/postprocess.py`)

The per-point head emits one box per point, so inference collapses them:
confidence filter → **per-class** BEV NMS (`torchvision.batched_nms`, so a box
of one class never suppresses an overlapping box of a different class) →
optional top-K. `Custom/visualize.py` exposes the confidence and NMS-IoU
thresholds as live sliders.

### 4.5 Checkpoint selection

`best.pth` is chosen on a **combined** metric — validation segmentation
accuracy **plus** detection **mAP@0.5** — so a checkpoint is not frozen the
moment segmentation saturates while detection is still converging (the two
tasks converge at very different rates). The detection score is real 3D-IoU
mAP, computed by `metrics.detection_metrics.DetectionMetrics` — the same
metric pyLitePT uses, so JePT detection numbers are directly comparable.

---

## 5. Data layer (`data/`)

| file | role |
|---|---|
| `unlabeled_dataset.py` | `UnlabeledPointDataset` — reads `coord` (+ optional `color`/`normal`); pads/trims channels; mixes scenes + objects |
| `labeled_dataset.py` | `CustomDataset` — pyLitePT's loader; NPY folders or PLY; reads `segment` + `gt_boxes` |
| `collate.py` | `collate_point_batch` — ragged clouds → flat tensors + `offset`/`batch` |
| `transforms.py` | scale-aware augmentation (no unit-sphere — preserves metric scale) |

Both stages accept the **labelCloud / pyLitePT NPY-folder format**
(`coord.npy`, `color.npy`, `segment.npy`, `gt_boxes.npy`), so the bundled
`labelCloud/` annotation GUI feeds JePT directly.

---

## 6. Folder map

```
JePT/
├── Custom/      USER LAYER — configs, run scripts, data tools, GUI  (edit here)
├── jepa/        CORE — tokenizer, sequencer, sampler, predictor, ema, jepa_model, losses
├── engine/      CORE — pretrain / finetune / evaluate loops, freeze controller, checkpoint
├── data/        CORE — datasets, collation, augmentation
├── models/      LitePT backbone (from pyLitePT) + unified.py (heads + weight loader)
├── pcdet_lite/  detection head (PointHeadBox) — from pyLitePT
├── labelCloud/  3D annotation GUI — from pyLitePT
├── backend_cpu/, libs/, metrics/, utils/, hybrid_backend.py   (from pyLitePT)
├── ARCHITECTURE.md / TRAINING_WORKFLOW.md / REFERENCES.md / README.md
```

**Core vs. Custom.** `Custom/` holds everything you edit per dataset (configs,
run scripts, data tools). The core packages (`jepa/`, `engine/`, `models/`,
`data/`) implement the algorithm and are never touched for customization — this
mirrors pyLitePT's `Custom/` split.

---

## 7. End-to-end data flow

```
unlabeled clouds ─▶ UnlabeledPointDataset ─▶ collate ─▶ JePTModel
                                                          │
                            student(context) ◀────────────┤
                            teacher(full, EMA) ◀───────────┤
                            predictor ─▶ smooth-L1 loss ◀──┘
                                                          │
                                      EMA teacher state_dict (backbone)
                                                          │
labeled clouds ─▶ CustomDataset ─▶ collate ─▶ create_unified_model
                                              load_pretrained_backbone ◀┘
                                                          │
                            backbone ─▶ seg head ─▶ CE loss
                                     └▶ det head ─▶ box loss
                                                          │
                                              FreezeController + AdamW
                                                          │
                                          fine-tuned checkpoint ─▶ visualize.py
```
