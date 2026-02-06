"""
Dummy segmentation shims used only for GUI debugging.

These functions mirror the signatures the GUI expects but return placeholder
values so the interface can run without the real segmentation stack.
"""

from typing import Any, Dict, Optional
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
    "wall_subsample_stride": 2,
    "wall_dist_thresh": 0.03,
    "wall_max_iters": 300,
    "wall_min_inliers": 400,
    "wall_max_angle_deg": 20.0,
    "wall_score_subset": 2048,
    "wall_early_stop_ratio": 0.90,
    "wall_batch_size": 512,
    "wall_refine_dist_mult": 1.6,
    "max_up_dot": 0.35,
    "ground_perp_deg": 20.0,
    "wall_ortho_deg": 20.0,
    "wall_parallel_deg": 10.0,
    "wall_parallel_distance_m": 0.60,
    "wall_mask_refine": True,
    "door_hue_tol": 18,
    "door_min_s": 30,
    "door_min_v": 20,
    "door_glare_s_max": 35,
    "door_glare_v_min": 210,
    "door_glare_v_clip": 200,
    "door_ground_parallel_deg": 15.0,
    "door_plane_inlier_ratio": 0.70,
}

_metrics: Dict[str, Any] = {"last_ransac_ms": None}


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
    ground_params: Optional[Dict[str, Any]] = None,
    dataset_index: Optional[int] = None,
    **_: Any,
) -> Any:
    """
    Return a fixed placeholder value; ignores all parameters.

    Additional parameters (ground_params, dataset_index, **kwargs) are accepted
    for API parity with the real segmentation module.
    """
    # Accept parameters to keep GUI compatibility, but avoid any processing.
    _ = (
        color_width,
        color_height,
        depth_width,
        depth_height,
        fps,
        stride,
        mode,
        ground_params,
        dataset_index,
    )
    _metrics["last_ransac_ms"] = None
    return None


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


def obtener_metricas(copy: bool = True) -> Dict[str, Any]:
    """
    Dummy metrics accessor to match the real segmentation API.
    """
    return dict(_metrics) if copy else _metrics
