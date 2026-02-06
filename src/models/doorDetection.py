#!/usr/bin/env python3
"""
Door Detection using a TensorRT model.
"""
import os
from typing import Optional, Dict, Any

import cv2
import numpy as np

from .trt_inference import TRTInference
from .door_hsv import refine_door_mask_hsv


IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)


# Centralized state for lazy initialization
_runtime: Dict[str, Any] = {
    "model": None,
    "engine_path": None,
    "input_size": (256, 256),  # (width, height) - must match TensorRT engine
    "min_area": 300,  # Minimum connected-component area (pixels) to keep
}


def _lazy_init(engine_path: Optional[str] = None) -> bool:
    """
    Lazy initialization of the TensorRT model.

    Returns True if model is ready, False otherwise.
    """
    if _runtime["model"] is not None:
        return True

    if engine_path is None:
        engine_path = _runtime.get("engine_path")

    if engine_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        engine_path = os.path.join(script_dir, "doors", "bisenetv2.engine")

    if not os.path.exists(engine_path):
        print(f"[doorDetection] Engine file not found: {engine_path}")
        return False

    try:
        print(f"[doorDetection] Loading TensorRT engine from: {engine_path}")
        _runtime["model"] = TRTInference(engine_path)
        _runtime["engine_path"] = engine_path
        print("[doorDetection] Model loaded successfully")
        return True
    except Exception as exc:
        print(f"[doorDetection] Failed to load model: {exc}")
        return False


def _preprocess_inputs(rgb_image: np.ndarray) -> np.ndarray:
    """
    Preprocess RGB input for the door model.

    Args:
        rgb_image: BGR image (H, W, 3) as returned by OpenCV/RealSense.

    Returns:
        Input tensor of shape (1, 3, 256, 256), float32 in [0, 1].
    """
    input_size = _runtime["input_size"]

    if rgb_image.ndim == 2:
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2BGR)
    elif rgb_image.ndim == 3 and rgb_image.shape[2] > 3:
        rgb_image = rgb_image[:, :, :3]

    # Match training pipeline: PIL loads RGB, so convert BGR -> RGB here.
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_image, input_size, interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - np.array(IMG_MEAN, dtype=np.float32)) / np.array(
        IMG_STD, dtype=np.float32
    )

    # HWC -> CHW and add batch dimension
    input_tensor = normalized.transpose(2, 0, 1)[None, ...]
    return np.ascontiguousarray(input_tensor, dtype=np.float32)


def _postprocess_outputs(output: np.ndarray, original_size) -> np.ndarray:
    """
    Postprocess model outputs into a binary door mask.
    """
    output = np.asarray(output)

    if output.ndim == 4:
        # (N, C, H, W)
        output = output[0]
        if output.shape[0] == 1:
            mask_raw = output[0]
        else:
            mask_raw = np.argmax(output, axis=0)
    elif output.ndim == 3:
        # (C, H, W) or (1, H, W)
        if output.shape[0] == 1:
            mask_raw = output[0]
        else:
            mask_raw = np.argmax(output, axis=0)
    elif output.ndim == 2:
        mask_raw = output
    else:
        raise ValueError(f"Unexpected output shape: {output.shape}")

    if mask_raw.dtype.kind in ("f", "b"):
        max_val = float(np.nanmax(mask_raw)) if mask_raw.size else 0.0
        min_val = float(np.nanmin(mask_raw)) if mask_raw.size else 0.0
        if 0.0 <= min_val and max_val <= 1.0:
            mask = mask_raw > 0.5
        else:
            mask = mask_raw > 0.0
    else:
        mask = mask_raw > 0

    door_mask = mask.astype(np.uint8) * 255
    door_mask = cv2.resize(
        door_mask,
        (original_size[1], original_size[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return door_mask


def doorDetection(
    rgb_image: np.ndarray,
    min_area: Optional[int] = None,
    use_roi: bool = True,
    reduce_glare: bool = True,
    hue_tol: Optional[int] = None,
    min_s: Optional[int] = None,
    min_v: Optional[int] = None,
    glare_s_max: Optional[int] = None,
    glare_v_min: Optional[int] = None,
    glare_v_clip: Optional[int] = None,
) -> np.ndarray:
    """
    Detect doors from an RGB image using TensorRT.

    Args:
        rgb_image: BGR image (H, W, 3), values 0-255.
        min_area: Minimum area (pixels) to keep in the output mask.
            Use 0 to disable filtering. If None, uses runtime default.
        use_roi: If True, use the largest ROI to compute the dominant color
            inside the segmented component (HSV) and return a mask of ROI
            pixels that match that color.
        reduce_glare: If True, clip very bright low-saturation pixels before
            HSV-based color selection.
        hue_tol: Hue tolerance around the dominant hue (0-179).
        min_s: Minimum saturation threshold for HSV selection (0-255).
        min_v: Minimum value threshold for HSV selection (0-255).
        glare_s_max: Maximum saturation to classify glare (0-255).
        glare_v_min: Minimum value to classify glare (0-255).
        glare_v_clip: Value used to clip glare pixels (0-255).

    Returns:
        door_mask: Binary mask (H, W) with doors marked as 255.
    """
    if not _lazy_init():
        raise RuntimeError("Model not initialized. Cannot run doorDetection.")

    original_size = rgb_image.shape[:2]
    input_tensor = _preprocess_inputs(rgb_image)
    output = _runtime["model"].infer(input_tensor)
    door_mask = _postprocess_outputs(output, original_size)
    if min_area is None:
        min_area = int(_runtime.get("min_area", 0))
    else:
        min_area = int(min_area)
    return refine_door_mask_hsv(
        rgb_image,
        door_mask,
        min_area=min_area,
        use_roi=use_roi,
        reduce_glare=reduce_glare,
        hue_tol=hue_tol,
        min_s=min_s,
        min_v=min_v,
        glare_s_max=glare_s_max,
        glare_v_min=glare_v_min,
        glare_v_clip=glare_v_clip,
    )


# Backwards-compatible alias (if any old code still references wallDetection)
wallDetection = doorDetection
