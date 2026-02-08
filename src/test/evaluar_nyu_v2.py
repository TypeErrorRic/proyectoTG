"""
Evalua segmentacion de suelo y muros sobre NYU Depth V2 (MATLAB v7.3).

Este script usa AlgoritmosSegmentacion en modo "prueba" y compara las
mascaras resultantes con el ground-truth del dataset.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Tuple

import numpy as np
import cv2

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    import h5py
except Exception as exc:
    raise SystemExit(
        "Falta h5py para leer archivos .mat v7.3. "
        "Instala con: pip install h5py"
    ) from exc

# Ensure repo root is in sys.path so "src.*" imports work.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utilities import segmentar

WALL_IDS = [21]
FLOOR_IDS = [11]


def _find_dataset(h5file: h5py.File, names) -> h5py.Dataset:
    keys = list(h5file.keys())
    lower_map = {k.lower(): k for k in keys}
    for name in names:
        if name in h5file:
            return h5file[name]
        lname = name.lower()
        if lname in lower_map:
            return h5file[lower_map[lname]]
    # Fallback: partial match
    for k in keys:
        kl = k.lower()
        if any(n.lower() in kl for n in names):
            return h5file[k]
    raise KeyError(f"No se encontro dataset con nombres {names}. Keys: {keys}")


def _infer_frame_axis(shape) -> int:
    if not shape:
        raise ValueError("Shape invalida para dataset")
    return int(np.argmax(shape))


def _slice_on_axis(ds: h5py.Dataset, axis: int, idx: int) -> np.ndarray:
    sl = [slice(None)] * ds.ndim
    sl[axis] = idx
    return ds[tuple(sl)]


def _normalize_rgb(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 3:
        arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"RGB invalido, shape={arr.shape}")
    ch_axes = [i for i, s in enumerate(arr.shape) if s == 3]
    if not ch_axes:
        raise ValueError(f"No se encontro canal RGB, shape={arr.shape}")
    c = ch_axes[0]
    if c != 2:
        arr = np.moveaxis(arr, c, -1)
    h, w = arr.shape[:2]
    if h > w:
        arr = arr.transpose(1, 0, 2)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _normalize_hw(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Mapa invalido, shape={arr.shape}")
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    return arr


class NYUV2Mat:
    def __init__(self, path: str):
        self.path = path
        self.h5 = h5py.File(path, "r")
        self.ds_images = _find_dataset(self.h5, ["images", "image"])
        self.ds_depths = _find_dataset(self.h5, ["depths", "depth"])
        self.ds_labels = _find_dataset(self.h5, ["labels", "label"])

        self.axis_images = _infer_frame_axis(self.ds_images.shape)
        self.axis_depths = _infer_frame_axis(self.ds_depths.shape)
        self.axis_labels = _infer_frame_axis(self.ds_labels.shape)

        self.num_frames = min(
            self.ds_images.shape[self.axis_images],
            self.ds_depths.shape[self.axis_depths],
            self.ds_labels.shape[self.axis_labels],
        )

    def get_frame(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = int(index) % self.num_frames
        rgb = _slice_on_axis(self.ds_images, self.axis_images, idx)
        depth = _slice_on_axis(self.ds_depths, self.axis_depths, idx)
        labels = _slice_on_axis(self.ds_labels, self.axis_labels, idx)

        rgb = _normalize_rgb(rgb)
        depth = _normalize_hw(depth).astype(np.float32)
        labels = _normalize_hw(labels).astype(np.int32)

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr, depth, labels

    def close(self) -> None:
        try:
            self.h5.close()
        except Exception:
            pass


class FrameCache:
    def __init__(self, reader: NYUV2Mat):
        self.reader = reader
        self.last_index = None
        self.last_bgr = None
        self.last_depth = None
        self.last_labels = None

    def load(self, index: int):
        if self.last_index != index:
            bgr, depth, labels = self.reader.get_frame(index)
            self.last_index = index
            self.last_bgr = bgr
            self.last_depth = depth
            self.last_labels = labels
        return self.last_bgr, self.last_depth, self.last_labels


def make_dataset_loader(cache: FrameCache):
    def _loader(index=None):
        idx = 0 if index is None else int(index)
        bgr, depth, _ = cache.load(idx)
        return bgr, depth

    return _loader


def _drain_results():
    while True:
        if segmentar.obtener_resultado() is None:
            break


def _as_bool_mask(mask: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[:, :, 0]
    if m.shape[:2] != shape_hw:
        m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return m > 0


def _metrics_from_counts(counts: Dict[str, int]) -> Dict[str, float]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    denom_iou = tp + fp + fn
    denom_p = tp + fp
    denom_r = tp + fn
    iou = (tp / denom_iou) if denom_iou else 0.0
    prec = (tp / denom_p) if denom_p else 0.0
    rec = (tp / denom_r) if denom_r else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"iou": iou, "precision": prec, "recall": rec, "f1": f1}


def _iou_from_masks(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    return (inter / union) if union else 0.0


def _update_counts(counts: Dict[str, int], pred: np.ndarray, gt: np.ndarray) -> None:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    counts["tp"] += int(np.logical_and(pred, gt).sum())
    counts["fp"] += int(np.logical_and(pred, ~gt).sum())
    counts["fn"] += int(np.logical_and(~pred, gt).sum())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mat",
        default=os.path.join("src", "test", "data", "nyu_depth_v2_labeled.mat"),
        help="Ruta al archivo .mat del NYU V2",
    )
    parser.add_argument("--index", type=int, default=None, help="Indice unico a evaluar")
    parser.add_argument("--start", type=int, default=0, help="Indice inicial")
    parser.add_argument("--count", type=int, default=5, help="Cantidad de frames")
    parser.add_argument("--step", type=int, default=1, help="Paso entre indices")
    parser.add_argument("--timeout", type=float, default=8.0, help="Timeout por frame (s)")
    parser.add_argument("--verbose", action="store_true", help="Log por frame")
    parser.add_argument(
        "--save_iou",
        type=float,
        default=0.80,
        help="Umbral de IoU para guardar overlay (0-1)",
    )
    parser.add_argument(
        "--save_dir",
        default=os.path.join("data", "nyu_eval_iou80"),
        help="Carpeta base para guardar overlays",
    )
    args = parser.parse_args()

    mat_path = args.mat
    if not os.path.isfile(mat_path):
        print(f"No existe: {mat_path}")
        return 1

    reader = NYUV2Mat(mat_path)
    cache = FrameCache(reader)

    # Monkeypatch dataset loader used by AlgoritmosSegmentacion
    segmentar.load_dataset_frame = make_dataset_loader(cache)

    try:
        if args.index is not None:
            indices = [args.index]
        else:
            indices = list(range(args.start, args.start + args.count * args.step, args.step))

        totals = {
            "floor": {"tp": 0, "fp": 0, "fn": 0},
            "wall": {"tp": 0, "fp": 0, "fn": 0},
        }

        save_thresh = float(args.save_iou)
        save_base = args.save_dir
        floor_dir = os.path.join(save_base, "floor")
        wall_dir = os.path.join(save_base, "wall")
        both_dir = os.path.join(save_base, "both")
        os.makedirs(floor_dir, exist_ok=True)
        os.makedirs(wall_dir, exist_ok=True)
        os.makedirs(both_dir, exist_ok=True)

        time_sum = 0.0
        processed = 0
        iterator = indices
        if tqdm is not None:
            iterator = tqdm(indices, desc="Evaluando", unit="frame")

        for idx in iterator:
            _drain_results()
            t0 = time.time()

            bgr, depth, labels = cache.load(idx)
            h, w = labels.shape[:2]

            gt_floor = np.isin(labels, FLOOR_IDS)
            if int(gt_floor.sum()) <= 1200:
                if args.verbose:
                    print(f"[{idx}] Skip: piso <= 1200 pixeles")
                continue

            # Schedules segmentation
            segmentar.AlgoritmosSegmentacion(
                mode="prueba",
                dataset_index=idx,
            )

            # Wait for background result (forces completion)
            overlay = segmentar.obtener_resultado(bloqueante=True, timeout=args.timeout)
            masks = segmentar.obtener_mascaras(copy=True)

            pred_floor = _as_bool_mask(masks.get("ground"), (h, w))
            pred_wall = _as_bool_mask(masks.get("wall"), (h, w))
            if pred_floor is None or pred_wall is None:
                print(f"[{idx}] No se obtuvieron mascaras")
                continue

            gt_wall = np.isin(labels, WALL_IDS)

            _update_counts(totals["floor"], pred_floor, gt_floor)
            _update_counts(totals["wall"], pred_wall, gt_wall)

            iou_floor = _iou_from_masks(pred_floor, gt_floor)
            iou_wall = _iou_from_masks(pred_wall, gt_wall)

            if overlay is not None and (iou_floor >= save_thresh or iou_wall >= save_thresh):
                if not isinstance(overlay, np.ndarray):
                    overlay = np.asarray(overlay)
                fname = f"idx_{idx:05d}_floor_{iou_floor:.3f}_wall_{iou_wall:.3f}.png"
                if iou_floor >= save_thresh:
                    cv2.imwrite(os.path.join(floor_dir, fname), overlay)
                if iou_wall >= save_thresh:
                    cv2.imwrite(os.path.join(wall_dir, fname), overlay)
                if iou_floor >= save_thresh and iou_wall >= save_thresh:
                    cv2.imwrite(os.path.join(both_dir, fname), overlay)

            if args.verbose:
                dt = time.time() - t0
                mf = _metrics_from_counts({"tp": int(np.logical_and(pred_floor, gt_floor).sum()),
                                           "fp": int(np.logical_and(pred_floor, ~gt_floor).sum()),
                                           "fn": int(np.logical_and(~pred_floor, gt_floor).sum())})
                mw = _metrics_from_counts({"tp": int(np.logical_and(pred_wall, gt_wall).sum()),
                                           "fp": int(np.logical_and(pred_wall, ~gt_wall).sum()),
                                           "fn": int(np.logical_and(~pred_wall, gt_wall).sum())})
                print(
                    f"[{idx}] floor IoU={mf['iou']:.3f} wall IoU={mw['iou']:.3f} "
                    f"({dt:.2f}s)"
                )

            time_sum += (time.time() - t0)
            processed += 1

        floor_metrics = _metrics_from_counts(totals["floor"])
        wall_metrics = _metrics_from_counts(totals["wall"])

        print("Resultados globales")
        print(
            f"Floor: IoU={floor_metrics['iou']:.3f} "
            f"Precision={floor_metrics['precision']:.3f} "
            f"Recall={floor_metrics['recall']:.3f} "
            f"F1={floor_metrics['f1']:.3f}"
        )
        print(
            f"Wall:  IoU={wall_metrics['iou']:.3f} "
            f"Precision={wall_metrics['precision']:.3f} "
            f"Recall={wall_metrics['recall']:.3f} "
            f"F1={wall_metrics['f1']:.3f}"
        )
        if processed > 0:
            avg = time_sum / processed
            print(f"Tiempo promedio por frame: {avg:.2f}s ({processed} frames)")
        else:
            print("Tiempo promedio por frame: N/A (0 frames procesados)")
    finally:
        reader.close()
        segmentar.liberar_recursos()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
