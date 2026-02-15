"""
Visor simple RGB + Depth en tiempo real usando utilidades de src/utilities/viewCamera.py.

Controles:
- q / ESC: salir
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np


# Permite ejecutar este archivo directamente: `python src/data/RGB_DEPTH.py`
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.utilities.viewCamera import (  # noqa: E402
    extract_depth_meters,
    extract_rgb,
    init_camera,
    make_depth_to_color_aligner,
)


def depth_to_colormap(depth_m: np.ndarray) -> np.ndarray:
    """Convierte depth (m, float32) a imagen BGR para visualizacion."""
    if depth_m is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    valid = depth_m > 0.0
    if not np.any(valid):
        return np.zeros((depth_m.shape[0], depth_m.shape[1], 3), dtype=np.uint8)

    valid_vals = depth_m[valid]
    vmin = float(np.percentile(valid_vals, 2))
    vmax = float(np.percentile(valid_vals, 98))
    vmax = max(vmax, vmin + 1e-6)

    depth_norm = np.clip((depth_m - vmin) / (vmax - vmin), 0.0, 1.0)
    depth_8u = (depth_norm * 255.0).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_TURBO)
    depth_color[~valid] = 0
    return depth_color


def main() -> int:
    pipeline = None
    try:
        pipeline, _ = init_camera(
            color_width=640,
            color_height=480,
            depth_width=640,
            depth_height=480,
            fps=30,
        )
        align_depth_to_color = make_depth_to_color_aligner(pipeline)

        cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)

        print("Mostrando RGB + Depth. Presiona 'q' o ESC para salir.")

        while True:
            frames = pipeline.wait_for_frames()
            rgb = extract_rgb(frames, copy=False)
            depth_m = (
                align_depth_to_color(frames)
                if align_depth_to_color is not None
                else extract_depth_meters(frames, pipeline=pipeline)
            )

            if rgb is None or depth_m is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                continue

            if depth_m.shape[:2] != rgb.shape[:2]:
                depth_m = cv2.resize(
                    depth_m,
                    (rgb.shape[1], rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            depth_vis = depth_to_colormap(depth_m)
            cv2.imshow("RGB", rgb)
            cv2.imshow("Depth", depth_vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        if pipeline is not None:
            pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
