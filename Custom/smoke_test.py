"""End-to-end smoke test for the JePT pipeline (CPU, tiny synthetic data).

Verifies, on a few synthetic clouds:
  1. tokenisation handles mixed object/scene scale,
  2. JEPA pretraining runs, loss is finite, no representation collapse,
  3. pretrained weights load into the downstream model,
  4. all three freeze modes behave correctly,
  5. a fine-tuned model produces valid seg + det outputs.

Run from the JePT/ directory:
    python -m Custom.smoke_test
    python Custom/smoke_test.py
"""

import os
import shutil
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from hybrid_backend import setup_backends  # noqa: E402

setup_backends(verbose=False)

_SMOKE = os.path.join(_ROOT, "_smoke")


# ----------------------------------------------------------------------
# synthetic data
# ----------------------------------------------------------------------
def _make_object(rng, n=1500):
    pts = rng.standard_normal((n, 3)).astype(np.float32)
    pts /= (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-6)
    pts *= rng.uniform(0.2, 1.0, size=(n, 1)).astype(np.float32)  # filled ball
    color = rng.random((n, 3)).astype(np.float32)
    return pts, color


def _make_scene(rng, n=8000):
    pts = (rng.random((n, 3)) * np.array([10.0, 10.0, 3.0])).astype(np.float32)
    color = rng.random((n, 3)).astype(np.float32)
    return pts, color


def _write_npy_folder(path, coord, color, labeled, rng):
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "coord.npy"), coord)
    np.save(os.path.join(path, "color.npy"), color)
    if labeled:
        seg = rng.integers(0, 6, size=coord.shape[0]).astype(np.int64)
        np.save(os.path.join(path, "segment.npy"), seg)
        lo, hi = coord.min(0), coord.max(0)
        m = 3
        boxes = np.zeros((m, 8), dtype=np.float32)
        for i in range(m):
            center = rng.uniform(lo, hi)
            dims = rng.uniform(0.4, 1.2, size=3)
            heading = rng.uniform(-np.pi, np.pi)
            label = rng.integers(0, 6)
            boxes[i] = [*center, *dims, heading, label]
        np.save(os.path.join(path, "gt_boxes.npy"), boxes)


def build_data():
    if os.path.exists(_SMOKE):
        shutil.rmtree(_SMOKE)
    rng = np.random.default_rng(0)
    unlab = os.path.join(_SMOKE, "data_unlabeled")
    # mixed scale: 3 objects + 3 scenes, unlabeled
    for i in range(3):
        c, col = _make_object(rng)
        _write_npy_folder(os.path.join(unlab, f"obj_{i}"), c, col, False, rng)
    for i in range(3):
        c, col = _make_scene(rng)
        _write_npy_folder(os.path.join(unlab, f"scene_{i}"), c, col, False, rng)
    # labeled train/val
    for split, count in (("train", 4), ("val", 2)):
        for i in range(count):
            if i % 2 == 0:
                c, col = _make_scene(rng, n=6000)
            else:
                c, col = _make_object(rng, n=1500)
            _write_npy_folder(os.path.join(_SMOKE, "data_labeled", split,
                                           f"{split}_{i}"), c, col, True, rng)
    return unlab, os.path.join(_SMOKE, "data_labeled")


# ----------------------------------------------------------------------
# config helpers
# ----------------------------------------------------------------------
def _override(module, **kw):
    for k, v in kw.items():
        setattr(module, k, v)
    return module


def _check(name, ok):
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}")
    if not ok:
        raise AssertionError(f"smoke test failed: {name}")


# ----------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------
def test_tokenizer(unlab):
    print("\n[2] tokenizer — mixed scale")
    from jepa.tokenizer import compute_groups
    from data import UnlabeledPointDataset, collate_point_batch
    import Custom.pretrain_config as pc
    _override(pc, GRID_SIZE=0.05, INPUT_CHANNELS=6, AUGMENT=False)

    ds = UnlabeledPointDataset(unlab, pc, split="train")
    obj = next(s for s in (ds[i] for i in range(len(ds)))
               if s["coord"].shape[0] < 3000)
    scene = next(s for s in (ds[i] for i in range(len(ds)))
                 if s["coord"].shape[0] >= 3000)
    b_obj = collate_point_batch([obj])
    b_scene = collate_point_batch([scene])
    gid_o, _, _ = compute_groups(b_obj["coord"], b_obj["batch"], 0.4)
    gid_s, _, _ = compute_groups(b_scene["coord"], b_scene["batch"], 0.4)
    n_obj = int(gid_o.max()) + 1
    n_scene = int(gid_s.max()) + 1
    print(f"      object groups={n_obj}  scene groups={n_scene}")
    _check("object yields >= 4 groups", n_obj >= 4)
    _check("scene yields more groups than object", n_scene > n_obj)


def test_pretrain(unlab):
    print("\n[3] JEPA pretraining")
    import Custom.pretrain_config as pc
    _override(pc,
              UNLABELED_DATA_PATH=unlab,
              RESULTS_DIR=os.path.join(_SMOKE, "exp_pretrain"),
              MODEL_VARIANT="nano", INPUT_CHANNELS=6, GRID_SIZE=0.05,
              AUGMENT=False, GROUP_SIZE=0.4, MIN_TOKENS=4, MAX_TOKENS=128,
              NUM_TARGET_BLOCKS=4, TARGET_RATIO=(0.15, 0.20),
              PREDICTOR_DIM=96, PREDICTOR_DEPTH=2, PREDICTOR_HEADS=6,
              EPOCHS=2, BATCH_SIZE=2, LR=1.5e-3, WARMUP_EPOCHS=1,
              EMA_TAU_EPOCHS=2, NUM_WORKERS=0, USE_AMP=False, SEED=0,
              SAVE_EVERY=1)
    from engine.pretrain import pretrain
    hist = pretrain(pc, verbose=True)

    losses = hist["loss"]
    _check("loss is finite", all(np.isfinite(losses)))
    # collapse = representation variance vanishing toward 0; targets are
    # layer-normalised so a healthy across-token std is O(0.1), well above 0.
    _check("target_std > 0.02 (no collapse)", hist["target_std"][-1] > 0.02)
    _check("pred_std > 0.002 (no collapse)", hist["pred_std"][-1] > 0.002)
    _check("loss decreased over training", losses[-1] < losses[0])
    ckpt = os.path.join(pc.RESULTS_DIR, "last.pth")
    _check("checkpoint written", os.path.exists(ckpt))

    sd = torch.load(ckpt, map_location="cpu")
    teacher_p = sd["teacher"]
    student_p = sd["student"]
    tkey = next(k for k in teacher_p if k.startswith("ema_model.")
                and teacher_p[k].dtype.is_floating_point)
    skey = tkey[len("ema_model."):]
    diff = (teacher_p[tkey] - student_p[skey]).abs().sum().item()
    _check("teacher differs from student (EMA updated)", diff > 0)
    return ckpt


def _transferred_encoder_sd(ckpt):
    """The encoder state-dict that load_pretrained_backbone transfers — the EMA
    teacher (ema_model.-prefix stripped), falling back to the student."""
    sd = torch.load(ckpt, map_location="cpu")
    teacher = sd.get("teacher")
    if isinstance(teacher, dict) and any(
            k.startswith("ema_model.") for k in teacher):
        return {k[len("ema_model."):]: v for k, v in teacher.items()
                if k.startswith("ema_model.")}
    return sd["student"]


def test_weight_load(ckpt):
    print("\n[4] pretrained weight loading")
    import Custom.finetune_config as fc
    _override(fc, MODEL_VARIANT="nano", USE_DUAL_PATH_UNIFIED=False,
              INPUT_CHANNELS=6, NUM_CLASSES_SEG=6, NUM_CLASSES_DET=6,
              PRETRAINED_CKPT=None)
    from models.unified import create_unified_model, load_pretrained_backbone

    det_cfg = {"MEAN_SIZE": [[1.0, 1.0, 1.0]] * 6}
    model = create_unified_model(fc, 6, det_cfg, device="cpu")

    enc_sd = _transferred_encoder_sd(ckpt)
    wkey = next(k for k in enc_sd
                if "embedding" in k and enc_sd[k].dtype.is_floating_point)
    before = model.state_dict()[f"backbone.{wkey}"].clone()

    report = load_pretrained_backbone(model, ckpt, verbose=True)
    after = model.state_dict()[f"backbone.{wkey}"]
    matched = report["backbone"][0]
    _check("matched > 0 backbone tensors", matched > 0)
    _check("backbone embedding weights changed after load",
           not torch.allclose(before, after))
    _check("loaded weights equal pretrained encoder (EMA teacher)",
           torch.allclose(after, enc_sd[wkey]))


def test_finetune_modes(labeled, ckpt):
    print("\n[5] fine-tuning — all three freeze modes")
    import Custom.finetune_config as fc
    from engine.finetune import finetune

    enc_sd = _transferred_encoder_sd(ckpt)
    wkey = next(k for k in enc_sd
                if "embedding" in k and enc_sd[k].dtype.is_floating_point)

    results = {}
    for mode in ("linear_probe", "full_finetune", "staged_unfreeze"):
        print(f"  --- mode: {mode} ---")
        _override(fc,
                  DATA_PATH=labeled, DATA_FORMAT="npy",
                  RESULTS_DIR=os.path.join(_SMOKE, f"exp_ft_{mode}"),
                  MODEL_VARIANT="nano", USE_DUAL_PATH_UNIFIED=False,
                  INPUT_CHANNELS=6, GRID_SIZE=0.05, AUGMENT=False,
                  NUM_CLASSES_SEG=6, NUM_CLASSES_DET=6,
                  CLASS_WEIGHTS=[1.0] * 6, PRETRAINED_CKPT=ckpt,
                  FREEZE_MODE=mode, FREEZE_EPOCHS=1, BACKBONE_LR_SCALE=0.1,
                  EPOCHS=3, BATCH_SIZE=2, LEARNING_RATE=1e-3,
                  NUM_WORKERS=0, USE_AMP=False)
        model, hist = finetune(fc, verbose=True)
        results[mode] = (model, hist)

        seg = hist["train_seg_loss"]
        det = hist["train_det_loss"]
        _check(f"{mode}: seg loss finite", all(np.isfinite(seg)))
        _check(f"{mode}: det loss finite", all(np.isfinite(det)))

        after = model.state_dict()[f"backbone.{wkey}"]
        changed = not torch.allclose(after, enc_sd[wkey])
        if mode == "linear_probe":
            _check("linear_probe: backbone UNCHANGED", not changed)
        else:
            _check(f"{mode}: backbone CHANGED (was fine-tuned)", changed)

    # one decreasing-loss check on the most-trainable mode
    total = results["full_finetune"][1]["train_total"]
    _check("full_finetune: total loss did not diverge",
           total[-1] <= total[0] * 2.0)
    return results["staged_unfreeze"][0], fc


def test_inference(model, fc, labeled):
    print("\n[6] inference — seg + det outputs")
    from data import CustomDataset, collate_point_batch
    ds = CustomDataset(labeled, "val", fc, "npy")
    batch = collate_point_batch([ds[0]])
    model.eval()
    with torch.no_grad():
        out = model(batch)
    seg = out["seg_logits"]
    _check("seg_logits shape (N, NUM_CLASSES_SEG)",
           seg.ndim == 2 and seg.shape[1] == fc.NUM_CLASSES_SEG)
    det_keys = {"batch_cls_preds", "point_cls_scores", "batch_box_preds"}
    _check("detection outputs present",
           len(det_keys & set(out.keys())) > 0)


def main():
    print("=" * 60)
    print("JePT smoke test")
    print("=" * 60)
    print("\n[1] building synthetic data")
    unlab, labeled = build_data()
    print(f"      unlabeled: {unlab}")
    print(f"      labeled:   {labeled}")

    test_tokenizer(unlab)
    ckpt = test_pretrain(unlab)
    test_weight_load(ckpt)
    model, fc = test_finetune_modes(labeled, ckpt)
    test_inference(model, fc, labeled)

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
