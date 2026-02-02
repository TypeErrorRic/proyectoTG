#!/usr/bin/env python3
"""
Door Detection using a TensorRT model.
"""
import os
from typing import Optional, Dict, Any

import cv2
import numpy as np

from .trt_inference import TRTInference


IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)
_HUE_TOL = 10  # Hue tolerance (0-179)
_MIN_S = 40  # Minimum saturation for color selection
_MIN_V = 40  # Minimum value for color selection


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


def _filter_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """
    Remove connected components smaller than min_area from a binary mask.
    """
    if min_area <= 0:
        return mask

    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if num_labels <= 1:
        return mask

    keep = np.zeros_like(binary)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            keep[labels == label] = 1

    return (keep * 255).astype(np.uint8)


def _largest_component(mask: np.ndarray, min_area: int):
    """
    Return labels, largest component label, and its bounding box.
    """
    if mask.size == 0:
        return None, None, None

    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary):
        return None, None, None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if num_labels <= 1:
        return None, None, None

    areas = stats[1:, cv2.CC_STAT_AREA]
    if min_area > 0:
        valid = np.where(areas >= min_area)[0]
        if valid.size == 0:
            return None, None, None
        largest_idx = int(valid[np.argmax(areas[valid])])
    else:
        largest_idx = int(np.argmax(areas))

    label = largest_idx + 1
    x = int(stats[label, cv2.CC_STAT_LEFT])
    y = int(stats[label, cv2.CC_STAT_TOP])
    w = int(stats[label, cv2.CC_STAT_WIDTH])
    h = int(stats[label, cv2.CC_STAT_HEIGHT])
    return labels, label, (x, y, w, h)


def _dominant_hsv_mask(
    bgr_image: np.ndarray,
    labels: np.ndarray,
    label: int,
    bbox,
) -> np.ndarray:
    """
    Build a mask of ROI pixels that match the dominant HSV color within the component.
    """
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return np.zeros(bgr_image.shape[:2], dtype=np.uint8)

    roi_img = bgr_image[y : y + h, x : x + w]
    roi_labels = labels[y : y + h, x : x + w]
    roi_component = roi_labels == label
    if not np.any(roi_component):
        return np.zeros(bgr_image.shape[:2], dtype=np.uint8)

    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    valid = roi_component & (s_channel >= _MIN_S) & (v_channel >= _MIN_V)
    if not np.any(valid):
        valid = roi_component
    h_vals = h_channel[valid]
    if h_vals.size == 0:
        return np.zeros(bgr_image.shape[:2], dtype=np.uint8)

    hist = np.bincount(h_vals, minlength=180)
    dominant_h = int(hist.argmax())

    lower = dominant_h - _HUE_TOL
    upper = dominant_h + _HUE_TOL
    if lower < 0:
        hue_mask = (h_channel >= lower + 180) | (h_channel <= upper)
    elif upper >= 180:
        hue_mask = (h_channel >= lower) | (h_channel <= upper - 180)
    else:
        hue_mask = (h_channel >= lower) & (h_channel <= upper)

    match = hue_mask & (s_channel >= _MIN_S) & (v_channel >= _MIN_V)
    out = np.zeros(bgr_image.shape[:2], dtype=np.uint8)
    out[y : y + h, x : x + w] = (match * 255).astype(np.uint8)
    return out


def doorDetection(
    rgb_image: np.ndarray,
    min_area: Optional[int] = None,
    use_roi: bool = True,
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
    if use_roi:
        labels, label, bbox = _largest_component(door_mask, min_area)
        if bbox is None:
            return _filter_small_components(door_mask, min_area)
        return _dominant_hsv_mask(rgb_image, labels, label, bbox)

    return _filter_small_components(door_mask, min_area)


# Backwards-compatible alias (if any old code still references wallDetection)
wallDetection = doorDetection
