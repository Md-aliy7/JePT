"""JePT prediction visualizer — interactive PySide6 + VisPy GUI.

Mirrors pyLitePT's `Custom/visualize.py`: a dual-canvas window (Ground Truth |
Prediction) with a scene navigator, detection controls and a per-class legend.
Adapted to the JePT data/model layer and robust to checkpoints with or without
stored meta.

Usage (run from the JePT/ directory):
    python Custom/visualize.py                       # auto-discover ckpt + data
    python Custom/visualize.py --checkpoint exp_demo/finetune_pretrained/best.pth
    python Custom/visualize.py --split test --save   # headless: write PLYs, no GUI

Arguments:
    --checkpoint  fine-tuning checkpoint (default: auto-discover newest)
    --split       train | val | test            (default: test)
    --data        dataset root                   (default: auto-discover)
    --save        headless mode — write coloured PLYs instead of opening a window
"""

import argparse
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from hybrid_backend import setup_backends  # noqa: E402

setup_backends(verbose=False)

import Custom.finetune_config as cfg                       # noqa: E402
from data import CustomDataset, collate_point_batch        # noqa: E402
from engine.common import to_device                        # noqa: E402
from models.unified import create_unified_model            # noqa: E402
from pcdet_lite.box_utils import boxes_to_corners_3d        # noqa: E402
from Custom.postprocess import filter_and_nms               # noqa: E402

# 12 edges of a 3D box (pcdet corner ordering)
_BOX_EDGES = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                       [4, 5], [5, 6], [6, 7], [7, 4],
                       [0, 4], [1, 5], [2, 6], [3, 7]])


# ======================================================================
# colours
# ======================================================================
def make_colors(num_classes):
    """Distinct RGB per class via evenly-spaced HSV hues."""
    if num_classes <= 0:
        return np.zeros((0, 3), np.float32)
    colors = np.ones((num_classes, 3), np.float32)
    hues = np.arange(num_classes) / max(num_classes, 1)
    s, v = 0.8, 0.9
    i_h = (hues * 6).astype(int) % 6
    f = hues * 6 - np.floor(hues * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    for i in range(num_classes):
        ih = i_h[i]
        colors[i] = ([v, t[i], p], [q[i], v, p], [p, v, t[i]],
                     [p, q[i], v], [t[i], p, v], [v, p, q[i]])[ih]
    return colors


# ======================================================================
# checkpoint -> model
# ======================================================================
def infer_classes_from_state(state):
    """Infer (num_seg, num_det) from a state-dict (fallback for no-meta ckpts)."""
    num_seg = num_det = None
    if "seg_head.weight" in state:
        num_seg = int(state["seg_head.weight"].shape[0])
    cls = {}
    for k, v in state.items():
        if ".cls_layers." in k and k.endswith(".weight") and v.ndim == 2:
            idx = int(k.split(".cls_layers.")[1].split(".")[0])
            cls[idx] = int(v.shape[0])
    if cls:
        num_det = cls[max(cls)]
    return num_seg, num_det


def apply_checkpoint_meta(ckpt):
    """Make finetune_config match the trained checkpoint (meta, else inference)."""
    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
    if meta:
        cfg.MODEL_VARIANT = meta.get("variant", cfg.MODEL_VARIANT)
        cfg.INPUT_CHANNELS = meta.get("input_channels", cfg.INPUT_CHANNELS)
        cfg.NUM_CLASSES_SEG = meta.get("num_classes_seg", cfg.NUM_CLASSES_SEG)
        cfg.NUM_CLASSES_DET = meta.get("num_classes_det", cfg.NUM_CLASSES_DET)
        cfg.USE_DUAL_PATH_UNIFIED = meta.get(
            "use_dual_path", getattr(cfg, "USE_DUAL_PATH_UNIFIED", False))
        if meta.get("class_names"):
            cfg.CLASS_NAMES = list(meta["class_names"])
        if meta.get("grid_size"):
            cfg.GRID_SIZE = meta["grid_size"]
        print(f"[visualize] checkpoint meta applied: {meta}")
    else:
        state = ckpt.get("model_state_dict", ckpt)
        ns, nd = infer_classes_from_state(state)
        if ns is not None:
            cfg.NUM_CLASSES_SEG = ns
        if nd is not None:
            cfg.NUM_CLASSES_DET = nd
        print(f"[visualize] no meta — inferred class counts from weights: "
              f"seg={cfg.NUM_CLASSES_SEG} det={cfg.NUM_CLASSES_DET} "
              f"(variant={cfg.MODEL_VARIANT} from config)")
    # ensure class-name list length matches the segmentation class count
    names = list(getattr(cfg, "CLASS_NAMES", []) or [])
    if len(names) != cfg.NUM_CLASSES_SEG:
        cfg.CLASS_NAMES = [f"class {i}" for i in range(cfg.NUM_CLASSES_SEG)]


def build_model(ckpt, dataset, device):
    """Build the unified model and load fine-tuned weights (shape-filtered)."""
    det_config = dict(getattr(cfg, "DETECTION_CONFIG", {}) or {})
    if cfg.NUM_CLASSES_DET > 0 and det_config.get("MEAN_SIZE", "auto") == "auto":
        det_config["MEAN_SIZE"] = dataset.calculate_mean_sizes(cfg.NUM_CLASSES_DET)

    prev = getattr(cfg, "PRETRAINED_CKPT", None)
    cfg.PRETRAINED_CKPT = None      # load the full checkpoint, not a backbone
    model = create_unified_model(cfg, cfg.INPUT_CHANNELS, det_config, device)
    cfg.PRETRAINED_CKPT = prev

    state = ckpt.get("model_state_dict", ckpt)
    model_state = model.state_dict()
    filtered, skipped = {}, []
    for k, v in state.items():
        if k in model_state and v.shape == model_state[k].shape:
            filtered[k] = v
        elif k in model_state:
            skipped.append(k)
    if skipped:
        print(f"[visualize] WARNING: {len(skipped)} tensors skipped "
              f"(shape mismatch) — check variant/class counts")
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model


@torch.no_grad()
def infer(model, dataset, idx, device):
    """Run the model on scene `idx`; return a result dict."""
    data = dataset[idx]
    batch = to_device(collate_point_batch([data]), device)
    out = model(batch)
    coord = batch["coord"].cpu().numpy()
    pred_seg = out["seg_logits"].argmax(dim=1).cpu().numpy() \
        if "seg_logits" in out else np.zeros(len(coord), np.int64)
    gt_seg = data["segment"].numpy() if "segment" in data else None
    gt_boxes = data["gt_boxes"].numpy() if "gt_boxes" in data else None

    # raw detection output — kept as tensors; conf + NMS applied at draw time
    boxes = scores = cls_logits = None
    if "batch_box_preds" in out and out["batch_box_preds"].numel() > 0:
        boxes = out["batch_box_preds"].detach().cpu()[:, :7]
        if "point_cls_scores" in out:
            scores = out["point_cls_scores"].detach().cpu().reshape(-1)
        if "batch_cls_preds" in out:
            cls_logits = out["batch_cls_preds"].detach().cpu()
    return dict(coord=coord, pred_seg=pred_seg, gt_seg=gt_seg,
                gt_boxes=gt_boxes, boxes=boxes, scores=scores,
                cls_logits=cls_logits, name=str(data.get("name", idx)))


def box_lines(boxes):
    """(N,7) boxes -> (N*24, 3) line-segment vertices for VisPy/Open3D."""
    corners = boxes_to_corners_3d(torch.from_numpy(boxes).float()).numpy()
    start = corners[:, _BOX_EDGES[:, 0], :]
    end = corners[:, _BOX_EDGES[:, 1], :]
    return np.stack([start, end], axis=2).reshape(-1, 3)


# ======================================================================
# headless save mode (no display)
# ======================================================================
def save_mode(checkpoint, split, data_path, scene_index=0):
    import open3d as o3d
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device)
    apply_checkpoint_meta(ckpt)
    dataset = CustomDataset(data_path, split, cfg, getattr(cfg, "DATA_FORMAT", "npy"))
    if len(dataset) == 0:
        raise RuntimeError(f"no scenes in split '{split}' of {data_path}")
    model = build_model(ckpt, dataset, device)
    r = infer(model, dataset, min(scene_index, len(dataset) - 1), device)

    palette = make_colors(cfg.NUM_CLASSES_SEG)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(checkpoint)), "viz")
    os.makedirs(out_dir, exist_ok=True)

    def _ply(path, coord, labels):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(coord.astype(np.float64))
        col = palette[np.clip(labels, 0, cfg.NUM_CLASSES_SEG - 1)]
        pc.colors = o3d.utility.Vector3dVector(col.astype(np.float64))
        o3d.io.write_point_cloud(path, pc)

    _ply(os.path.join(out_dir, "pred_seg.ply"), r["coord"], r["pred_seg"])
    if r["gt_seg"] is not None:
        x = float(np.ptp(r["coord"][:, 0])) + 1.0
        _ply(os.path.join(out_dir, "gt_seg.ply"),
             r["coord"] + np.array([x, 0, 0], np.float32), r["gt_seg"])
        valid = (r["gt_seg"] >= 0) & (r["gt_seg"] < cfg.NUM_CLASSES_SEG)
        if valid.any():
            acc = float((r["pred_seg"][valid] == r["gt_seg"][valid]).mean())
            print(f"[visualize] scene '{r['name']}' point accuracy = {acc:.4f}")
    if r["boxes"] is not None and r["scores"] is not None:
        res = filter_and_nms(r["boxes"], r["scores"], conf_thresh=0.3,
                             iou_thresh=0.2, cls_preds=r["cls_logits"])
        np.save(os.path.join(out_dir, "pred_boxes.npy"), res["boxes"])
        print(f"[visualize] {len(res['boxes'])} predicted boxes "
              f"(after conf>=0.3 + NMS)")
    print(f"[visualize] scene '{r['name']}' — artefacts written to {out_dir}")


# ======================================================================
# VisPy canvas
# ======================================================================
def _make_canvas():
    from vispy import scene
    canvas = scene.SceneCanvas(keys="interactive", show=False, bgcolor="white")
    canvas.unfreeze()
    canvas.view = canvas.central_widget.add_view()
    canvas.view.camera = scene.ArcballCamera(fov=45, distance=5)
    canvas.scatter = scene.visuals.Markers(parent=canvas.view.scene)
    canvas.scatter.set_gl_state("translucent", depth_test=True)
    canvas.boxes = scene.visuals.Line(method="gl", width=2,
                                      parent=canvas.view.scene)
    scene.visuals.XYZAxis(parent=canvas.view.scene)
    canvas.freeze()
    return canvas


def _set_points(canvas, coord, labels, palette):
    n = len(coord)
    col = np.full((n, 3), 0.5, np.float32)
    if labels is not None and len(palette) > 0:
        valid = labels >= 0
        col[valid] = palette[np.clip(labels[valid], 0, len(palette) - 1)]
    canvas.scatter.set_data(pos=coord, face_color=col, size=5, edge_width=0)
    c = coord.mean(0)
    canvas.view.camera.center = tuple(c)
    canvas.view.camera.distance = float(np.max(coord.max(0) - coord.min(0))) * 1.5


def _set_boxes(canvas, boxes, color):
    if boxes is None or len(boxes) == 0:
        canvas.boxes.set_data(pos=np.zeros((0, 3)), connect="segments")
        return
    verts = box_lines(boxes)
    if not isinstance(color, str) and len(color) == len(boxes):
        color = np.repeat(color, 24, axis=0)
    canvas.boxes.set_data(pos=verts, color=color, connect="segments")


# ======================================================================
# GUI
# ======================================================================
def run_gui(checkpoint, split, data_path, test_mode=False):
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QComboBox, QLabel, QPushButton, QSplitter, QCheckBox, QSlider,
        QFrame, QGroupBox, QScrollArea)
    from PySide6.QtCore import Qt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device)
    apply_checkpoint_meta(ckpt)
    dataset = CustomDataset(data_path, split, cfg,
                            getattr(cfg, "DATA_FORMAT", "npy"))
    if len(dataset) == 0:
        raise RuntimeError(f"no scenes in split '{split}' of {data_path}")
    model = build_model(ckpt, dataset, device)
    class_names = list(getattr(cfg, "CLASS_NAMES",
                               [f"class {i}" for i in range(cfg.NUM_CLASSES_SEG)]))
    palette = make_colors(max(cfg.NUM_CLASSES_SEG, len(class_names)))

    class Window(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("JePT Visualizer")
            self.setGeometry(80, 80, 1600, 880)
            self.idx = 0
            self.show_boxes = True
            self.conf = 0.3            # detection confidence threshold
            self.nms_iou = 0.20        # NMS IoU threshold
            self.result = None
            self._build()
            self.load(0)

        # ---- UI ----------------------------------------------------
        def _build(self):
            central = QWidget()
            self.setCentralWidget(central)
            layout = QHBoxLayout(central)

            side = QFrame()
            side.setFixedWidth(300)
            sl = QVBoxLayout(side)

            info = QGroupBox("Model")
            il = QVBoxLayout(info)
            il.addWidget(QLabel(f"device: {device}"))
            il.addWidget(QLabel(f"variant: {cfg.MODEL_VARIANT}"))
            il.addWidget(QLabel(f"seg classes: {cfg.NUM_CLASSES_SEG}"))
            il.addWidget(QLabel(f"det classes: {cfg.NUM_CLASSES_DET}"))
            il.addWidget(QLabel(f"scenes: {len(dataset)}"))
            sl.addWidget(info)

            navg = QGroupBox("Scene")
            nv = QVBoxLayout(navg)
            self.combo = QComboBox()
            for s in dataset.scenes:
                self.combo.addItem(os.path.basename(str(s)))
            self.combo.currentIndexChanged.connect(self.load)
            nv.addWidget(self.combo)
            row = QHBoxLayout()
            b_prev = QPushButton("◀ Prev")
            b_prev.clicked.connect(lambda: self.combo.setCurrentIndex(
                max(0, self.idx - 1)))
            b_next = QPushButton("Next ▶")
            b_next.clicked.connect(lambda: self.combo.setCurrentIndex(
                min(len(dataset) - 1, self.idx + 1)))
            row.addWidget(b_prev)
            row.addWidget(b_next)
            nv.addLayout(row)
            sl.addWidget(navg)

            if cfg.NUM_CLASSES_DET > 0:
                detg = QGroupBox("Detection")
                dv = QVBoxLayout(detg)
                chk = QCheckBox("show boxes")
                chk.setChecked(True)
                chk.stateChanged.connect(self._toggle_boxes)
                dv.addWidget(chk)
                cr = QHBoxLayout()
                cr.addWidget(QLabel("conf"))
                self.slider = QSlider(Qt.Horizontal)
                self.slider.setRange(0, 100)
                self.slider.setValue(int(self.conf * 100))
                self.slider.valueChanged.connect(self._conf)
                cr.addWidget(self.slider)
                self.conf_lbl = QLabel(f"{self.conf:.2f}")
                cr.addWidget(self.conf_lbl)
                dv.addLayout(cr)
                # NMS IoU threshold — collapses overlapping per-point boxes
                nr = QHBoxLayout()
                nr.addWidget(QLabel("NMS"))
                self.nms_slider = QSlider(Qt.Horizontal)
                self.nms_slider.setRange(1, 100)
                self.nms_slider.setValue(int(self.nms_iou * 100))
                self.nms_slider.valueChanged.connect(self._nms)
                nr.addWidget(self.nms_slider)
                self.nms_lbl = QLabel(f"{self.nms_iou:.2f}")
                nr.addWidget(self.nms_lbl)
                dv.addLayout(nr)
                sl.addWidget(detg)

            legg = QGroupBox("Class legend (per-scene IoU)")
            lgl = QVBoxLayout(legg)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            sw = QWidget()
            self.leg = QVBoxLayout(sw)
            self.leg_labels = []
            for i, name in enumerate(class_names):
                r = QHBoxLayout()
                sw_c = QLabel()
                sw_c.setFixedSize(18, 18)
                rgb = (palette[i] * 255).astype(int) if i < len(palette) \
                    else (127, 127, 127)
                sw_c.setStyleSheet(
                    f"background-color: rgb({rgb[0]},{rgb[1]},{rgb[2]});")
                lab = QLabel(f"{name}: --")
                self.leg_labels.append(lab)
                r.addWidget(sw_c)
                r.addWidget(lab)
                self.leg.addLayout(r)
            scroll.setWidget(sw)
            lgl.addWidget(scroll)
            sl.addWidget(legg, stretch=1)
            layout.addWidget(side)

            split_w = QSplitter(Qt.Horizontal)
            for title, attr in (("Ground Truth", "cv_gt"),
                                ("Prediction", "cv_pred")):
                cont = QWidget()
                cl = QVBoxLayout(cont)
                lab = QLabel(title)
                lab.setAlignment(Qt.AlignCenter)
                lab.setStyleSheet("font-weight: bold;")
                cl.addWidget(lab)
                cv = _make_canvas()
                setattr(self, attr, cv)
                cl.addWidget(cv.native)
                split_w.addWidget(cont)
            layout.addWidget(split_w, stretch=1)
            self.cv_pred.view.camera = self.cv_gt.view.camera   # linked

        # ---- callbacks --------------------------------------------
        def _toggle_boxes(self, state):
            self.show_boxes = bool(state)
            self.draw()

        def _conf(self, value):
            self.conf = value / 100.0
            self.conf_lbl.setText(f"{self.conf:.2f}")
            self.draw()

        def _nms(self, value):
            self.nms_iou = value / 100.0
            self.nms_lbl.setText(f"{self.nms_iou:.2f}")
            self.draw()

        def load(self, idx):
            self.idx = int(idx)
            self.result = infer(model, dataset, self.idx, device)
            self.draw()

        # ---- drawing ----------------------------------------------
        def draw(self):
            r = self.result
            if r is None:
                return
            _set_points(self.cv_gt, r["coord"], r["gt_seg"], palette)
            _set_points(self.cv_pred, r["coord"], r["pred_seg"], palette)

            if self.show_boxes and r["gt_boxes"] is not None \
                    and len(r["gt_boxes"]) > 0:
                gl = r["gt_boxes"][:, 7].astype(int) \
                    if r["gt_boxes"].shape[1] > 7 else None
                gc = palette[np.clip(gl, 0, len(palette) - 1)] \
                    if gl is not None else "green"
                _set_boxes(self.cv_gt, r["gt_boxes"][:, :7], gc)
            else:
                _set_boxes(self.cv_gt, None, "green")

            n_box = 0
            if self.show_boxes and r["boxes"] is not None \
                    and r["scores"] is not None:
                # confidence filter + NMS — collapses the ~100 raw per-point
                # boxes into one box per detected object
                res = filter_and_nms(r["boxes"], r["scores"],
                                     conf_thresh=self.conf,
                                     iou_thresh=self.nms_iou,
                                     cls_preds=r["cls_logits"],
                                     topk_fallback=20)
                fb, fl = res["boxes"], res["labels"]
                if len(fb) > 0:
                    bc = palette[np.clip(fl, 0, len(palette) - 1)]
                    _set_boxes(self.cv_pred, fb, bc)
                    n_box = len(fb)
                else:
                    _set_boxes(self.cv_pred, None, "red")
            else:
                _set_boxes(self.cv_pred, None, "red")

            self._update_legend(r)
            self.setWindowTitle(
                f"JePT Visualizer — scene {self.idx} '{r['name']}' — "
                f"{n_box} boxes (conf≥{self.conf:.2f}, NMS={self.nms_iou:.2f})")

        def _update_legend(self, r):
            gt, pred = r["gt_seg"], r["pred_seg"]
            if gt is None:
                return
            for i, lab in enumerate(self.leg_labels):
                inter = int(((gt == i) & (pred == i)).sum())
                union = int((gt == i).sum() + (pred == i).sum() - inter)
                iou = inter / union if union > 0 else 0.0
                lab.setText(f"{class_names[i]}: IoU={iou * 100:.0f}%")
                lab.setStyleSheet("color: green;" if iou > 0.5 else "color: #b00;")

    app = QApplication.instance() or QApplication(sys.argv)
    win = Window()
    win.show()
    if test_mode:
        from PySide6.QtCore import QTimer
        QTimer.singleShot(1000, app.quit)        # build, render, auto-close
        print("[visualize] --test-gui: window built OK, auto-closing.")
    else:
        print("[visualize] GUI open — close the window to exit.")
    app.exec()


# ======================================================================
# discovery + entry point
# ======================================================================
def _has_scenes(path):
    if not path or not os.path.isdir(path):
        return False
    import glob
    return bool(glob.glob(os.path.join(path, "**", "coord.npy"), recursive=True)
                or glob.glob(os.path.join(path, "**", "*.ply"), recursive=True))


def find_checkpoint():
    """Locate a fine-tuning checkpoint, preferring best.pth and pretrained runs."""
    import glob
    default = os.path.join(cfg.RESULTS_DIR, "best.pth")
    if os.path.exists(default):
        return default
    cands = []
    for name in ("best.pth", "last.pth"):
        cands += glob.glob(os.path.join(_ROOT, "exp*", "**", name),
                           recursive=True)
    # exclude JEPA *pretraining* output dirs (named exactly 'pretrain' /
    # 'jepa_pretrain') — those checkpoints have no seg/det heads. A precise
    # match so 'finetune_pretrained' is NOT excluded.
    cands = [c for c in cands if os.path.isfile(c)
             and os.path.basename(os.path.dirname(c)).lower()
             not in ("pretrain", "jepa_pretrain")]
    if not cands:
        return None

    def score(path):
        dirname = os.path.basename(os.path.dirname(path)).lower()
        return (os.path.basename(path) == "best.pth",   # best over last
                "pretrained" in dirname,                # pretrained over scratch
                os.path.getmtime(path))                 # newest
    return max(cands, key=score)


def find_data(meta_data_path=None):
    """Locate a dataset: the checkpoint's own training data if it still exists,
    else the most recently generated dataset folder."""
    if meta_data_path and _has_scenes(meta_data_path):
        return meta_data_path
    cands = [getattr(cfg, "DATA_PATH", None),
             os.path.join(_ROOT, "data_shapes3d"),
             os.path.join(_ROOT, "data_demo", "labeled"),
             os.path.join(_ROOT, "data_labeled")]
    existing = [p for p in cands if _has_scenes(p)]
    if not existing:
        return None
    # newest folder — usually the dataset most recently trained on
    return max(existing, key=os.path.getmtime)


def main():
    ap = argparse.ArgumentParser(description="JePT prediction visualizer")
    ap.add_argument("--checkpoint", default=None,
                    help="fine-tuning checkpoint (default: auto-discover)")
    ap.add_argument("--split", default="test",
                    choices=["train", "val", "test"])
    ap.add_argument("--data", default=None,
                    help="dataset root (default: auto-discover)")
    ap.add_argument("--scene", type=int, default=0,
                    help="scene index (save mode only)")
    ap.add_argument("--save", action="store_true",
                    help="headless: write coloured PLYs instead of a GUI window")
    ap.add_argument("--test-gui", action="store_true",
                    help="build the GUI window then auto-close (self-test)")
    args = ap.parse_args()

    checkpoint = args.checkpoint or find_checkpoint()
    if not checkpoint or not os.path.exists(checkpoint):
        print("ERROR: no fine-tuning checkpoint found.")
        print("  train first:  python -m Custom.run_finetune")
        print("  or the demo:  python -m Custom.run_shapes3d_demo")
        print("  or pass:      --checkpoint <path/to/best.pth>")
        sys.exit(1)

    # probe the checkpoint: reject JEPA pretraining ckpts, read its data path
    probe = torch.load(checkpoint, map_location="cpu")
    if isinstance(probe, dict) and "model_state_dict" not in probe \
            and "student" in probe:
        print(f"ERROR: {checkpoint} is a JEPA *pretraining* checkpoint "
              f"(no seg/det heads). Use a fine-tuning checkpoint.")
        sys.exit(1)
    meta_data = probe.get("meta", {}).get("data_path") \
        if isinstance(probe, dict) else None
    del probe

    # the checkpoint's own training data is the correct dataset to visualise
    data_path = args.data or find_data(meta_data)
    if not data_path:
        print("ERROR: no dataset found. Pass --data <folder>.")
        sys.exit(1)

    print(f"[visualize] checkpoint = {checkpoint}")
    print(f"[visualize] data       = {data_path}  (split={args.split})")

    if args.save:
        save_mode(checkpoint, args.split, data_path, args.scene)
        return
    try:
        run_gui(checkpoint, args.split, data_path, test_mode=args.test_gui)
    except Exception as exc:                       # no display / Qt failure
        print(f"[visualize] GUI unavailable ({exc}) — falling back to "
              f"--save mode.")
        save_mode(checkpoint, args.split, data_path, args.scene)


if __name__ == "__main__":
    main()
