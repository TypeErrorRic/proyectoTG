"""
HSV-based refinement utilities for door segmentation masks.
"""
from typing import Optional, Dict

import cv2
import numpy as np


_HUE_TOL = 18  # Hue tolerance (0-179)
_MIN_S = 30  # Minimum saturation for color selection
_MIN_V = 20  # Minimum value for color selection
_GLARE_S_MAX = 35  # Max saturation to consider as glare
_GLARE_V_MIN = 210  # Min value to consider as glare
_GLARE_V_CLIP = 200  # Clip glare value to this level


def _clamp_int(value: Optional[int], default: int, low: int, high: int) -> int:
    """
    Safely parse and clamp integer HSV parameters.
    """
    if value is None:
        return default
    try:
        parsed = int(value)
    except Exception:
        return default
    return max(low, min(high, parsed))


def _resolve_hsv_params(
    hue_tol: Optional[int],
    min_s: Optional[int],
    min_v: Optional[int],
    glare_s_max: Optional[int],
    glare_v_min: Optional[int],
    glare_v_clip: Optional[int],
) -> Dict[str, int]:
    """
    Resolve HSV parameters with defaults and bounds.
    """
    return {
        "hue_tol": _clamp_int(hue_tol, _HUE_TOL, 0, 179),
        "min_s": _clamp_int(min_s, _MIN_S, 0, 255),
        "min_v": _clamp_int(min_v, _MIN_V, 0, 255),
        "glare_s_max": _clamp_int(glare_s_max, _GLARE_S_MAX, 0, 255),
        "glare_v_min": _clamp_int(glare_v_min, _GLARE_V_MIN, 0, 255),
        "glare_v_clip": _clamp_int(glare_v_clip, _GLARE_V_CLIP, 0, 255),
    }


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


def _component_stats(mask: np.ndarray):
    """
    Return labels and stats for connected components.
    """
    if mask.size == 0:
        return None, None

    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary):
        return None, None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if num_labels <= 1:
        return None, None

    return labels, stats


def _largest_component_label(stats: np.ndarray, min_area: int) -> Optional[int]:
    """
    Return the label index of the largest component that meets min_area.
    """
    if stats is None or stats.shape[0] <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    if min_area > 0:
        valid = np.where(areas >= min_area)[0]
        if valid.size == 0:
            return None
        largest_idx = int(valid[np.argmax(areas[valid])])
    else:
        largest_idx = int(np.argmax(areas))

    return largest_idx + 1


def _roi_mask_from_bbox(shape, bbox) -> np.ndarray:
    """
    Build a rectangular ROI mask from a bbox.
    """
    x, y, w, h = bbox
    roi = np.zeros(shape, dtype=np.uint8)
    if w <= 0 or h <= 0:
        return roi
    roi[y : y + h, x : x + w] = 255
    return roi


def _fill_holes(
    mask: np.ndarray,
    kernel_size: int = 5,
    bbox=None,
) -> np.ndarray:
    """
    Fill holes in a binary mask using closing + flood fill.
    """
    if mask.size == 0:
        return mask

    if kernel_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if bbox is None:
        y0, x0 = 0, 0
        region = mask
    else:
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return mask
        y0, x0 = y, x
        region = mask[y : y + h, x : x + w]

    if region.size == 0:
        return mask

    region_bin = (region > 0).astype(np.uint8) * 255
    padded = cv2.copyMakeBorder(
        region_bin, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0
    )
    inv = cv2.bitwise_not(padded)
    flood = inv.copy()
    h, w = inv.shape
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ffmask, (0, 0), 0)
    filled = cv2.bitwise_or(padded, flood)
    filled = filled[1:-1, 1:-1]

    if bbox is None:
        return filled

    out = mask.copy()
    out[y0 : y0 + filled.shape[0], x0 : x0 + filled.shape[1]] = filled
    return out


def _dominant_hue_in_component(
    bgr_image: np.ndarray,
    labels: np.ndarray,
    label: int,
    bbox,
    reduce_glare: bool,
    hsv_params: Dict[str, int],
) -> Optional[int]:
    """
    Compute dominant hue inside a component ROI.
    """
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None

    roi_img = bgr_image[y : y + h, x : x + w]
    roi_labels = labels[y : y + h, x : x + w]
    roi_component = roi_labels == label
    if not np.any(roi_component):
        return None

    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    if reduce_glare:
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        glare = (s_channel <= hsv_params["glare_s_max"]) & (
            v_channel >= hsv_params["glare_v_min"]
        )
        if np.any(glare):
            hsv[:, :, 2] = np.where(glare, hsv_params["glare_v_clip"], v_channel)

    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    valid = roi_component & (s_channel >= hsv_params["min_s"]) & (
        v_channel >= hsv_params["min_v"]
    )
    if not np.any(valid):
        valid = roi_component
    h_vals = h_channel[valid]
    if h_vals.size == 0:
        return None

    hist = np.bincount(h_vals, minlength=180)
    return int(hist.argmax())


def _hsv_mask_for_hue(
    bgr_image: np.ndarray,
    bbox,
    dominant_h: int,
    reduce_glare: bool,
    hsv_params: Dict[str, int],
) -> np.ndarray:
    """
    Build a mask of ROI pixels that match a target hue.
    """
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return np.zeros(bgr_image.shape[:2], dtype=np.uint8)

    roi_img = bgr_image[y : y + h, x : x + w]
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    if reduce_glare:
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        glare = (s_channel <= hsv_params["glare_s_max"]) & (
            v_channel >= hsv_params["glare_v_min"]
        )
        if np.any(glare):
            hsv[:, :, 2] = np.where(glare, hsv_params["glare_v_clip"], v_channel)

    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    lower = dominant_h - hsv_params["hue_tol"]
    upper = dominant_h + hsv_params["hue_tol"]
    if lower < 0:
        hue_mask = (h_channel >= lower + 180) | (h_channel <= upper)
    elif upper >= 180:
        hue_mask = (h_channel >= lower) | (h_channel <= upper - 180)
    else:
        hue_mask = (h_channel >= lower) & (h_channel <= upper)

    match = hue_mask & (s_channel >= hsv_params["min_s"]) & (
        v_channel >= hsv_params["min_v"]
    )
    out = np.zeros(bgr_image.shape[:2], dtype=np.uint8)
    out[y : y + h, x : x + w] = (match * 255).astype(np.uint8)
    return out


def refine_door_mask_hsv(
    bgr_image: np.ndarray,
    door_mask: np.ndarray,
    min_area: int,
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
    Refine a door mask using HSV color selection.
    """
    hsv_params = _resolve_hsv_params(
        hue_tol,
        min_s,
        min_v,
        glare_s_max,
        glare_v_min,
        glare_v_clip,
    )

    if not use_roi:
        return _fill_holes(_filter_small_components(door_mask, min_area), 5)

    labels, stats = _component_stats(door_mask)
    if labels is None or stats is None:
        return _fill_holes(_filter_small_components(door_mask, min_area))

    largest_label = _largest_component_label(stats, min_area)
    if largest_label is None:
        return _fill_holes(_filter_small_components(door_mask, min_area))

    x = int(stats[largest_label, cv2.CC_STAT_LEFT])
    y = int(stats[largest_label, cv2.CC_STAT_TOP])
    w = int(stats[largest_label, cv2.CC_STAT_WIDTH])
    h = int(stats[largest_label, cv2.CC_STAT_HEIGHT])
    largest_bbox = (x, y, w, h)
    dominant_h = _dominant_hue_in_component(
        bgr_image, labels, largest_label, largest_bbox, reduce_glare, hsv_params
    )
    if dominant_h is None:
        return _fill_holes(_filter_small_components(door_mask, min_area))

    final_mask = np.zeros_like(door_mask, dtype=np.uint8)
    for label in range(1, stats.shape[0]):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if min_area > 0 and area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        bbox = (x, y, w, h)
        roi_mask = _roi_mask_from_bbox(door_mask.shape, bbox)
        roi_color = _hsv_mask_for_hue(
            bgr_image, bbox, dominant_h, reduce_glare, hsv_params
        )
        roi_color = _fill_holes(roi_color, 5, bbox)
        roi_color = cv2.bitwise_and(roi_color, roi_mask)
        final_mask = cv2.bitwise_or(final_mask, roi_color)

    return final_mask
