# Scientific references

Every non-trivial design choice in JePT is grounded in published work. This
file lists the primary sources and states exactly which JePT component each one
justifies.

## Self-supervised learning — the JEPA recipe

1. **I-JEPA** — Assran, Duval, Misra, Bojanowski, Vincent, Rabbat, LeCun,
   Ballas. *Self-Supervised Learning from Images with a Joint-Embedding
   Predictive Architecture.* CVPR 2023. arXiv:2301.08243.
   → Justifies: context/target split, latent-space prediction with a dedicated
   predictor, EMA target encoder, smooth-L1 loss, multi-block masking.
   Used in: [jepa/jepa_model.py](jepa/jepa_model.py),
   [jepa/predictor.py](jepa/predictor.py), [jepa/sampler.py](jepa/sampler.py).

2. **Point-JEPA** — Saito, Poovvancheri. *Point-JEPA: A Joint Embedding
   Predictive Architecture for Self-Supervised Learning on Point Cloud.*
   arXiv:2404.16432 (2024).
   → Justifies: JEPA adapted to point clouds, the iterative-nearest-neighbour
   token *sequencer*, contiguous-block context/target sampling, EMA `tau`
   ramp (0.9998→0.99999), smooth-L1 `beta=2`, staged encoder unfreezing for
   downstream fine-tuning.
   Used in: [jepa/sequencer.py](jepa/sequencer.py), [jepa/ema.py](jepa/ema.py),
   [engine/freeze.py](engine/freeze.py).

3. **data2vec** — Baevski, Hsu, Xu, Babu, Gu, Auli. *data2vec: A General
   Framework for Self-Supervised Learning in Speech, Vision and Language.*
   ICML 2022. arXiv:2202.03555.
   → Justifies: predicting *latent* (contextualised) target representations
   rather than raw inputs; normalising targets before the regression loss.
   Used in: target `layer_norm` in [jepa/jepa_model.py](jepa/jepa_model.py).

4. **BYOL** — Grill et al. *Bootstrap Your Own Latent.* NeurIPS 2020.
   arXiv:2006.07733.
   → Justifies: an EMA "teacher" of the online network as a stable, non-collapsing
   prediction target; copying buffers (BatchNorm stats) into the EMA model.
   Used in: [jepa/ema.py](jepa/ema.py).

5. **DINOv3** — Siméoni et al. *DINOv3.* arXiv:2508.10104 (2025).
   → Justifies: the *linear-probe / frozen-backbone* downstream mode — DINOv3
   freezes the backbone for dense tasks (segmentation, depth) and trains only
   lightweight heads.
   Used in: `linear_probe` mode in [engine/freeze.py](engine/freeze.py).

## Backbone

6. **pyLitePT / LitePT** — the realtime point-cloud transformer backbone JePT
   reuses unchanged (`models/litept/litept.py`). LitePT builds on the Point
   Transformer V3 line of work: Wu et al. *Point Transformer V3: Simpler,
   Faster, Stronger.* CVPR 2024. arXiv:2312.10035 (serialized attention,
   grid pooling, the `Point` data structure).

## Cross-validation references (point-cloud SSL on PTv3)

8. **Sonata** — Wu et al. *Sonata: Self-Supervised Learning of Reliable Point
   Representations.* CVPR 2025 (Highlight). arXiv:2503.16429.
   → JePT's reused Pointcept components (`Point` structure, z-order / Hilbert
   serialization, `PointModule`/`PointSequential`) were diff-checked against
   Sonata's released code and confirmed **identical** to the canonical
   implementation. Sonata pretrains PTv3 by multi-crop self-distillation — a
   different SSL paradigm from JePT's JEPA, included here as a correctness
   reference for the shared backbone primitives.

9. **Concerto** — Pointcept. *Concerto: Joint 2D-3D Self-Supervised Learning
   Emerges Spatial Representations.* arXiv:2510.23607.
   → Joint 2D-3D SSL pretraining of the same PTv3 backbone (built on Sonata).
   Reviewed; its 3D code path is the same canonical PTv3.

## Multi-task learning

7. **Uncertainty weighting** — Kendall, Gal, Cipolla. *Multi-Task Learning Using
   Uncertainty to Weigh Losses for Scene Geometry and Semantics.* CVPR 2018.
   arXiv:1705.07115.
   → Justifies: the learnable `log_vars` that balance the segmentation and
   detection losses during fine-tuning.
   Used in: `_combine_losses` in [engine/finetune.py](engine/finetune.py).

## Fine-tuning policy (freeze vs. unfreeze) — summary of the evidence

The literature does **not** endorse a single policy; JePT therefore supports
all three and defaults to staged unfreezing:

* Frozen backbone + light heads — DINOv3 [5] for dense tasks.
* Staged unfreeze — Point-JEPA [2] (encoder frozen for N epochs, then unfrozen).
* Full fine-tune at reduced LR — standard discriminative-transfer practice;
  generally the best final accuracy and typically required for detection,
  which transfers less readily than classification.
