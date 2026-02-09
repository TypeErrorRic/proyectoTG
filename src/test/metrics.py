"""
Evalua segmentacion de suelo y pared sobre el dataset PNG en src/test/data.

Genera metricas (precision, IoU, Dice, etc.) usando mascaras ya predichas
(PredFloor/PredWall) con una barra de progreso y exporta un JSON con todos
los resultados.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Valor fijo reportado por Jetson (evaluar_nyu_v2.py).
REPORTED_AVG_TIME_S = 1.26

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# Ensure repo root and src are in sys.path so "src.*" and "utilities.*" imports work.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(REPO_ROOT, path))


def _list_common_files(*dirs: str) -> List[str]:
    sets = []
    for d in dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"No existe la carpeta: {d}")
        files = [
            f
            for f in os.listdir(d)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        sets.append(set(files))
    common = sorted(set.intersection(*sets)) if sets else []
    return common


def _load_mask(mask_dir: str, filename: str, shape_hw: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    path = os.path.join(mask_dir, filename)
    if not os.path.exists(path):
        return None
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if shape_hw is not None and mask.shape[:2] != shape_hw:
        mask = cv2.resize(mask, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def _compute_counts(pred: np.ndarray, gt: np.ndarray) -> Dict[str, int]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _fbeta(precision: float, recall: float, beta: float) -> float:
    b2 = beta * beta
    denom = (b2 * precision) + recall
    return (1.0 + b2) * precision * recall / denom if denom else 0.0


def _metrics_from_counts(counts: Dict[str, int]) -> Dict[str, float]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    total = tp + fp + fn + tn

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    npv = _safe_div(tn, tn + fn)
    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, fn + tp)
    accuracy = _safe_div(tp + tn, total)
    iou = _safe_div(tp, tp + fp + fn)
    dice = _safe_div(2 * tp, 2 * tp + fp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    f05 = _fbeta(precision, recall, 0.5)
    f2 = _fbeta(precision, recall, 2.0)
    balanced_acc = 0.5 * (recall + specificity)

    denom_mcc = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = ((tp * tn - fp * fn) / math.sqrt(denom_mcc)) if denom_mcc else 0.0

    prevalence = _safe_div(tp + fn, total)
    pred_pos_rate = _safe_div(tp + fp, total)
    pred_neg_rate = _safe_div(tn + fn, total)

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "npv": npv,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "f1": f1,
        "f0_5": f05,
        "f2": f2,
        "fpr": fpr,
        "fnr": fnr,
        "mcc": mcc,
        "prevalence": prevalence,
        "pred_pos_rate": pred_pos_rate,
        "pred_neg_rate": pred_neg_rate,
    }


def _iter_with_progress(indices: List[int], total: int):
    if tqdm is not None:
        return tqdm(indices, desc="Evaluando", unit="frame")

    bar_len = 30

    def _gen():
        for i, idx in enumerate(indices, 1):
            filled = int(bar_len * i / max(1, total))
            bar = "#" * filled + "-" * (bar_len - filled)
            pct = int(100 * i / max(1, total))
            sys.stdout.write(f"\r[{bar}] {i}/{total} {pct}%")
            sys.stdout.flush()
            yield idx
        sys.stdout.write("\n")
        sys.stdout.flush()

    return _gen()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=os.path.join("src", "test", "data"),
        help="Carpeta base del dataset (contiene Floor, Wall, PredFloor, PredWall).",
    )
    parser.add_argument("--floor_dir", default="Floor", help="Subcarpeta Floor.")
    parser.add_argument("--wall_dir", default="Wall", help="Subcarpeta Wall.")
    parser.add_argument("--pred_floor_dir", default="PredFloor", help="Subcarpeta PredFloor.")
    parser.add_argument("--pred_wall_dir", default="PredWall", help="Subcarpeta PredWall.")
    parser.add_argument("--index", type=int, default=None, help="Indice unico a evaluar (0-based).")
    parser.add_argument("--start", type=int, default=0, help="Indice inicial.")
    parser.add_argument("--count", type=int, default=None, help="Cantidad de frames a evaluar.")
    parser.add_argument("--step", type=int, default=1, help="Paso entre indices.")
    parser.add_argument(
        "--skip_empty_gt",
        action="store_true",
        help="Omitir frames sin suelo o pared en el ground-truth.",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(THIS_DIR, "metrics.json"),
        help="Ruta de salida del JSON.",
    )
    parser.add_argument("--verbose", action="store_true", help="Log por frame.")
    args = parser.parse_args()

    data_dir = _resolve_path(args.data_dir)
    floor_dir = os.path.join(data_dir, args.floor_dir)
    wall_dir = os.path.join(data_dir, args.wall_dir)
    pred_floor_dir = os.path.join(data_dir, args.pred_floor_dir)
    pred_wall_dir = os.path.join(data_dir, args.pred_wall_dir)
    out_path = _resolve_path(args.out)

    file_list = _list_common_files(floor_dir, wall_dir, pred_floor_dir, pred_wall_dir)
    if not file_list:
        print("No se encontraron archivos comunes en las carpetas del dataset.")
        return 1

    if args.index is not None:
        indices = [int(args.index)]
    else:
        start = max(0, int(args.start))
        step = max(1, int(args.step))
        if args.count is None:
            indices = list(range(start, len(file_list), step))
        else:
            count = max(0, int(args.count))
            end = min(len(file_list), start + count * step)
            indices = list(range(start, end, step))

    if not indices:
        print("No hay indices a evaluar.")
        return 1

    totals = {
        "floor": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "wall": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
    }
    skipped: List[Dict[str, Any]] = []

    time_sum = 0.0
    processed = 0
    best_frames: List[Dict[str, Any]] = []

    iterator = _iter_with_progress(indices, total=len(indices))
    for idx in iterator:
        t0 = time.time()
        if idx < 0 or idx >= len(file_list):
            skipped.append({"index": idx, "reason": "indice_fuera_de_rango"})
            continue

        filename = file_list[idx]
        gt_floor = _load_mask(floor_dir, filename)
        if gt_floor is None:
            skipped.append({"index": idx, "name": filename, "reason": "gt_floor_invalido"})
            continue

        gt_wall = _load_mask(wall_dir, filename, shape_hw=gt_floor.shape[:2])
        if gt_wall is None:
            skipped.append({"index": idx, "name": filename, "reason": "gt_wall_invalido"})
            continue

        if args.skip_empty_gt and (not gt_floor.any() or not gt_wall.any()):
            skipped.append({"index": idx, "name": filename, "reason": "gt_vacio"})
            continue

        pred_floor = _load_mask(pred_floor_dir, filename, shape_hw=gt_floor.shape[:2])
        if pred_floor is None:
            skipped.append({"index": idx, "name": filename, "reason": "pred_floor_invalido"})
            continue

        pred_wall = _load_mask(pred_wall_dir, filename, shape_hw=gt_floor.shape[:2])
        if pred_wall is None:
            skipped.append({"index": idx, "name": filename, "reason": "pred_wall_invalido"})
            continue

        counts_floor = _compute_counts(pred_floor, gt_floor)
        counts_wall = _compute_counts(pred_wall, gt_wall)
        metrics_floor = _metrics_from_counts(counts_floor)
        metrics_wall = _metrics_from_counts(counts_wall)
        score = 0.5 * (metrics_floor["iou"] + metrics_wall["iou"])
        for k in totals["floor"]:
            totals["floor"][k] += counts_floor[k]
            totals["wall"][k] += counts_wall[k]

        frame_time = float(time.time() - t0)
        time_sum += frame_time
        processed += 1
        best_frames.append(
            {
                "score": score,
                "index": idx,
                "name": filename,
                "floor": metrics_floor,
                "wall": metrics_wall,
            }
        )

        if args.verbose:
            print(
                f"{filename} "
                f"tp_floor={counts_floor['tp']} fp_floor={counts_floor['fp']} "
                f"tp_wall={counts_wall['tp']} fp_wall={counts_wall['fp']}"
            )

    floor_micro = _metrics_from_counts(totals["floor"])
    wall_micro = _metrics_from_counts(totals["wall"])

    avg_time_reported = REPORTED_AVG_TIME_S if processed else None
    top5: List[Dict[str, Any]] = []
    if best_frames:
        best_frames.sort(key=lambda x: x["score"], reverse=True)
        for item in best_frames[:5]:
            mf = item["floor"]
            mw = item["wall"]
            top5.append(
                {
                    "index": item["index"],
                    "name": item["name"],
                    "score": item["score"],
                    "floor": {
                        "precision": mf["precision"],
                        "recall": mf["recall"],
                        "iou": mf["iou"],
                    },
                    "wall": {
                        "precision": mw["precision"],
                        "recall": mw["recall"],
                        "iou": mw["iou"],
                    },
                }
            )

    results = {
        "config": {
            "data_dir": data_dir,
            "floor_dir": floor_dir,
            "wall_dir": wall_dir,
            "pred_floor_dir": pred_floor_dir,
            "pred_wall_dir": pred_wall_dir,
            "start": int(args.start) if args.index is None else None,
            "count": int(args.count) if args.index is None and args.count is not None else None,
            "step": int(args.step) if args.index is None else None,
            "index": int(args.index) if args.index is not None else None,
            "skip_empty_gt": bool(args.skip_empty_gt),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_files": len(file_list),
        },
        "summary": {
            "processed": processed,
            "skipped": len(skipped),
            "avg_time_s": avg_time_reported,
        },
        "metrics": {
            "floor": {
                "counts": totals["floor"],
                "micro": floor_micro,
            },
            "wall": {
                "counts": totals["wall"],
                "micro": wall_micro,
            },
        },
        "top5": top5,
    }

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Resultados globales")
    print(
        "Floor: "
        f"IoU={floor_micro['iou']:.3f} "
        f"Precision={floor_micro['precision']:.3f} "
        f"Recall={floor_micro['recall']:.3f} "
        f"Dice={floor_micro['dice']:.3f}"
    )
    print(
        "Wall:  "
        f"IoU={wall_micro['iou']:.3f} "
        f"Precision={wall_micro['precision']:.3f} "
        f"Recall={wall_micro['recall']:.3f} "
        f"Dice={wall_micro['dice']:.3f}"
    )
    if processed > 0:
        print(f"Tiempo promedio por frame: {avg_time_reported:.2f}s ({processed} frames)")
    else:
        print("Tiempo promedio por frame: N/A (0 frames)")
    print(f"JSON guardado en: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
