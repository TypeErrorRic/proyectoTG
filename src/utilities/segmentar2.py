"""
Dummy segmentation shims used only for GUI debugging.

These functions mirror the signatures the GUI expects but return placeholder
values so the interface can run without the real segmentation stack.
"""

from typing import Any, Dict
import time

import cv2
import numpy as np

# Minimal parameter store so the config panel can read/write values.
_ground_params: Dict[str, Any] = {
    "subsample_stride": 2,
    "dist_thresh": 0.03,
    "max_iters": 500,
    "min_inliers": 400,
    "max_angle_deg": 60.0,
    "score_subset": 2048,
    "time_budget_ms": 100,
    "early_stop_ratio": 0.92,
    "batch_size": 256,
}


def _make_placeholder_frame(width: int, height: int, mode: str) -> np.ndarray:
    return None


def AlgoritmosSegmentacion(
    color_width: int = 640,
    color_height: int = 480,
    depth_width: int = 640,
    depth_height: int = 480,
    fps: int = 30,
    stride: int = 2,
    mode: str = "camera",
) -> Any:
    """
    Return a placeholder BGR frame; ignores all parameters.
    """
    _ = depth_width, depth_height, fps, stride  # unused, kept for parity
    return _make_placeholder_frame(color_width, color_height, mode)


def liberar_recursos() -> None:
    """
    Placeholder cleanup for API compatibility.
    """
    return None


def actualizar_parametros_ground(nuevos_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update dummy parameter store and return the merged dict.
    """
    if nuevos_params:
        for key, val in nuevos_params.items():
            if val is None:
                continue
            _ground_params[key] = val
    return obtener_parametros_ground()


def obtener_parametros_ground() -> Dict[str, Any]:
    """
    Return a copy of the current dummy ground parameters.
    """
    return dict(_ground_params)
