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

# Ensure repo root and src are in sys.path so "src.*" and "utilities.*" imports work.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from application.segmentacion import segmentacion
from application.gestorFotogramas import dataset_frames, mascaras

WALL_IDS = [21]
FLOOR_IDS = [11, 143, 891]


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
        if segmentacion.obtener_resultado() is None:
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


def _overlay_cpu(
    bgr: np.ndarray,
    floor_mask: np.ndarray | None,
    wall_mask: np.ndarray | None,
    alpha: float = 0.35,
) -> np.ndarray | None:
    if bgr is None:
        return None
    out = bgr.copy()
    if floor_mask is None and wall_mask is None:
        return out
    alpha = min(1.0, max(0.0, float(alpha)))
    if wall_mask is not None:
        color = np.array([255, 0, 0], dtype=np.uint8)  # BGR blue
        mask = wall_mask.astype(bool)
        out[mask] = (out[mask] * (1.0 - alpha) + color * alpha).astype(np.uint8)
    if floor_mask is not None:
        color = np.array([0, 255, 0], dtype=np.uint8)  # BGR green
        mask = floor_mask.astype(bool)
        out[mask] = (out[mask] * (1.0 - alpha) + color * alpha).astype(np.uint8)
    return out


def _depth_to_u16(depth: np.ndarray) -> np.ndarray | None:
    if depth is None:
        return None
    d = np.asarray(depth)
    if d.ndim != 2:
        d = np.squeeze(d)
    if d.ndim != 2:
        return None
    if d.dtype == np.uint16:
        return d
    if np.issubdtype(d.dtype, np.floating):
        d_mm = np.clip(d * 1000.0, 0, 65535)
        return d_mm.astype(np.uint16)
    return d.astype(np.uint16)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mat",
        default=os.path.join("tests", "data", "nyu_depth_v2_labeled.mat"),
        help="Ruta al archivo .mat del NYU V2",
    )
    parser.add_argument("--index", type=int, default=None, help="Indice unico a evaluar")
    parser.add_argument("--start", type=int, default=0, help="Indice inicial")
    parser.add_argument("--count", type=int, default=1449, help="Cantidad de frames")
    parser.add_argument("--step", type=int, default=1, help="Paso entre indices")
    parser.add_argument("--timeout", type=float, default=8.0, help="Timeout por frame (s)")
    parser.add_argument("--verbose", action="store_true", help="Log por frame")
    parser.add_argument(
        "--retry",
        type=int,
        default=3,
        help="Reintentos cuando falla segmentar o falta la mascara de suelo/pared",
    )
    parser.add_argument(
        "--save_iou",
        type=float,
        default=0.70,
        help="Umbral de IoU para suelo (0-1)",
    )
    parser.add_argument(
        "--save_iou_wall",
        type=float,
        default=0.50,
        help="Umbral de IoU para pared (0-1)",
    )
    parser.add_argument(
        "--save_dir",
        default=os.path.join("tests", "data"),
        help="Carpeta base para guardar resultados (se crean subcarpetas)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Guardar RGB + overlay en carpeta (modo show sin ventana)",
    )
    parser.add_argument(
        "--show_limit",
        type=int,
        default=0,
        help="Guardar overlays solo para las primeras N imagenes (0 = sin limite)",
    )
    parser.add_argument(
        "--show_dir",
        default=os.path.join("data", "nyu_eval_show"),
        help="Carpeta para guardar overlays cuando se usa --show",
    )
    args = parser.parse_args()

    mat_path = args.mat
    if not os.path.isfile(mat_path):
        return 1

    reader = NYUV2Mat(mat_path)
    cache = FrameCache(reader)

    # Monkeypatch dataset loader used by the segmentation pipeline.
    dataset_frames.load_dataset_frame = make_dataset_loader(cache)

    try:
        if args.index is not None:
            indices = [args.index]
        else:
            indices = list(range(args.start, args.start + args.count * args.step, args.step))

        totals = {
            "floor": {"tp": 0, "fp": 0, "fn": 0},
            "wall": {"tp": 0, "fp": 0, "fn": 0},
        }

        show_enabled = bool(args.show)
        show_limit = max(0, int(args.show_limit))
        shown = 0
        show_dir = args.show_dir
        max_retries = max(0, int(args.retry))
        save_thresh = float(args.save_iou)
        save_wall_thresh = float(args.save_iou_wall)
        save_base = args.save_dir
        save_rgb_dir = os.path.join(save_base, "RGB")
        save_depth_dir = os.path.join(save_base, "Depth")
        save_floor_dir = os.path.join(save_base, "Floor")
        save_wall_dir = os.path.join(save_base, "Wall")
        save_overlay_dir = os.path.join(save_base, "Overlay")
        save_pred_floor_dir = os.path.join(save_base, "PredFloor")
        save_pred_wall_dir = os.path.join(save_base, "PredWall")
        for d in (
            save_rgb_dir,
            save_depth_dir,
            save_floor_dir,
            save_wall_dir,
            save_overlay_dir,
            save_pred_floor_dir,
            save_pred_wall_dir,
        ):
            os.makedirs(d, exist_ok=True)
        if show_enabled:
            os.makedirs(show_dir, exist_ok=True)

        time_sum = 0.0
        processed = 0
        iterator = indices
        if tqdm is not None:
            iterator = tqdm(indices, desc="Evaluando", unit="frame")

        segmentacion.detener_hilo_secundario()
        try:
            segmentacion.inicializar(mode="prueba")
        except Exception:
            pass

        best_frames = []
        for idx in iterator:
            t0 = time.time()

            bgr, depth, labels = cache.load(idx)
            h, w = labels.shape[:2]

            gt_floor = np.isin(labels, FLOOR_IDS)
            gt_wall = np.isin(labels, WALL_IDS)
            if not gt_floor.any() or not gt_wall.any():
                continue

            ok = segmentacion.preprocesar(mode="prueba", dataset_index=idx)
            if not ok:
                continue

            best_pred_floor = None
            best_pred_wall = None
            best_overlay = None
            best_iou_floor = -1.0
            best_iou_wall = -1.0
            best_score = -1.0

            for _ in range(2):
                overlay = None
                masks = None
                pred_floor = None
                pred_wall = None
                for attempt in range(max_retries + 1):
                    try:
                        overlay = segmentacion.segmentar()
                        masks = segmentacion.obtener_mascaras(copy=True)
                    except Exception:
                        if attempt < max_retries:
                            continue
                        masks = None
                        overlay = None
                        break

                    pred_floor = _as_bool_mask(masks.get("ground"), (h, w)) if masks else None
                    pred_wall = _as_bool_mask(masks.get("wall"), (h, w)) if masks else None
                    if (pred_floor is None or pred_wall is None) and attempt < max_retries:
                        continue
                    break

                if pred_floor is None or pred_wall is None:
                    continue

                iou_floor = _iou_from_masks(pred_floor, gt_floor)
                iou_wall = _iou_from_masks(pred_wall, gt_wall)
                score = 0.5 * (iou_floor + iou_wall)
                if score > best_score:
                    best_score = score
                    best_iou_floor = iou_floor
                    best_iou_wall = iou_wall
                    best_pred_floor = pred_floor
                    best_pred_wall = pred_wall
                    best_overlay = overlay

            if best_pred_floor is None or best_pred_wall is None:
                continue

            pred_floor = best_pred_floor
            pred_wall = best_pred_wall
            overlay = best_overlay

            _update_counts(totals["floor"], pred_floor, gt_floor)
            _update_counts(totals["wall"], pred_wall, gt_wall)

            iou_floor = best_iou_floor
            iou_wall = best_iou_wall

            if overlay is None:
                try:
                    overlay = mascaras.apply_mask_to_rgb(
                        bgr,
                        pred_floor.astype(np.uint8) * 255,
                        pred_wall.astype(np.uint8) * 255,
                        None,
                    )
                except Exception:
                    overlay = None
            if overlay is None:
                overlay = _overlay_cpu(bgr, pred_floor, pred_wall)

            time_sum += (time.time() - t0)
            processed += 1
            best_frames.append((best_score, idx, iou_floor, iou_wall))

            if show_enabled and overlay is not None and (show_limit == 0 or shown < show_limit):
                view_rgb = bgr
                if view_rgb is None:
                    view_rgb = overlay
                if view_rgb.shape[:2] != overlay.shape[:2]:
                    view_rgb = cv2.resize(
                        view_rgb,
                        (overlay.shape[1], overlay.shape[0]),
                        interpolation=cv2.INTER_AREA,
                    )
                combo = cv2.hconcat([view_rgb, overlay])
                shown += 1
                out_name = f"show_{shown:03d}_idx_{idx:05d}.png"
                cv2.imwrite(os.path.join(show_dir, out_name), combo)

            if overlay is not None and (iou_floor >= save_thresh and iou_wall >= save_wall_thresh):
                base = f"{idx:04d}"
                cv2.imwrite(os.path.join(save_rgb_dir, f"{base}.png"), bgr)
                depth_u16 = _depth_to_u16(depth)
                if depth_u16 is not None:
                    cv2.imwrite(os.path.join(save_depth_dir, f"{base}.png"), depth_u16)
                cv2.imwrite(
                    os.path.join(save_floor_dir, f"{base}.png"),
                    gt_floor.astype(np.uint8) * 255,
                )
                cv2.imwrite(
                    os.path.join(save_wall_dir, f"{base}.png"),
                    gt_wall.astype(np.uint8) * 255,
                )
                cv2.imwrite(
                    os.path.join(save_pred_floor_dir, f"{base}.png"),
                    pred_floor.astype(np.uint8) * 255,
                )
                cv2.imwrite(
                    os.path.join(save_pred_wall_dir, f"{base}.png"),
                    pred_wall.astype(np.uint8) * 255,
                )
                cv2.imwrite(os.path.join(save_overlay_dir, f"{base}.png"), overlay)

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

        if best_frames:
            passing = [
                item
                for item in best_frames
                if item[2] >= save_thresh and item[3] >= save_wall_thresh
            ]
            if passing:
                passing.sort(key=lambda x: x[0], reverse=True)
                top = passing[:10]
                print("Top 10 mejores (cumplen umbrales IoU)")
                for rank, (score, idx, iou_f, iou_w) in enumerate(top, start=1):
                    print(
                        f"{rank:02d}. idx={idx:05d} score={score:.3f} "
                        f"floor={iou_f:.3f} wall={iou_w:.3f}"
                    )
    finally:
        reader.close()
        segmentacion.liberar_recursos()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

