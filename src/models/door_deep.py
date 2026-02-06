"""
Door ROI + point cloud segmentation using precomputed masks.

Uses:
  - Raw NN mask for ROI extraction.
  - HSV mask (already computed elsewhere).
  - Depth map + rays to build point clouds inside ROI & mask overlap.
"""
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np

from src.utilities.helpers import points_from_rays_and_depth


def _to_numpy(arr):
    if arr is None:
        return None
    try:
        import cupy as cp

        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    except Exception:
        pass
    return np.asarray(arr)


def infer_door_mask_raw(bgr_image: np.ndarray) -> np.ndarray:
    """
    Run the TensorRT door model and return the raw (pre-HSV) binary mask.
    """
    from . import doorDetection as door_det  # local import to avoid circular deps

    if not door_det._lazy_init():  # type: ignore[attr-defined]
        raise RuntimeError("Model not initialized. Cannot run door_deep.")

    input_tensor = door_det._preprocess_inputs(bgr_image)  # type: ignore[attr-defined]
    output = door_det._runtime["model"].infer(input_tensor)  # type: ignore[attr-defined]
    return door_det._postprocess_outputs(output, bgr_image.shape[:2])  # type: ignore[attr-defined]


def _component_stats(mask: np.ndarray):
    """
    Return labels and stats for connected components.
    """
    if mask is None or mask.size == 0:
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


def _iter_component_bboxes(
    mask: np.ndarray, min_area: int
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Iterate component bboxes above min_area.
    Returns (labels, stats, entries), entries is list of (label, bbox, area).
    """
    labels, stats = _component_stats(mask)
    if labels is None or stats is None:
        return None, None, []

    entries = []
    for label in range(1, stats.shape[0]):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if min_area > 0 and area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if w <= 0 or h <= 0:
            continue
        entries.append((label, (x, y, w, h), area))

    return labels, stats, entries


def _roi_mask_from_bbox(
    shape: Tuple[int, int], bbox: Optional[Tuple[int, int, int, int]]
) -> np.ndarray:
    """
    Build a rectangular ROI mask from a bbox.
    """
    roi = np.zeros(shape, dtype=np.uint8)
    if bbox is None:
        return roi
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return roi
    roi[y : y + h, x : x + w] = 255
    return roi


def door_roi_pointclouds(
    door_mask_raw: np.ndarray,
    hsv_mask: np.ndarray,
    depth_m: np.ndarray,
    rays,
    stride: int = 4,
    min_area: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute door ROIs using the raw NN mask and return point clouds per ROI.

    HSV mask is provided by the caller. Points are taken from the overlap
    between raw NN mask and HSV mask, constrained by each ROI.

    Returns:
        dict with:
          - door_mask_raw: uint8 mask (H, W)
          - hsv_mask: uint8 mask (H, W)
          - combined_mask: uint8 mask (H, W) [raw & hsv]
          - rois: list of dicts per ROI:
                {label, bbox, roi_mask, roi_combined_mask, points_xyz}
    """
    if door_mask_raw is None or hsv_mask is None:
        raise ValueError("door_mask_raw and hsv_mask are required.")

    door_mask = np.asarray(door_mask_raw)
    if door_mask.ndim == 3:
        door_mask = door_mask[:, :, 0]
    hsv_mask = np.asarray(hsv_mask)
    if hsv_mask.ndim == 3:
        hsv_mask = hsv_mask[:, :, 0]

    if door_mask.shape != hsv_mask.shape:
        hsv_mask = cv2.resize(
            hsv_mask, (door_mask.shape[1], door_mask.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    if min_area is None:
        min_area = 0
    else:
        min_area = int(min_area)

    combined_mask = cv2.bitwise_and(door_mask, hsv_mask)

    depth_np = _to_numpy(depth_m)
    rays_np = _to_numpy(rays)
    if depth_np is None or rays_np is None:
        return {
            "door_mask_raw": door_mask,
            "hsv_mask": hsv_mask,
            "combined_mask": combined_mask,
            "rois": [],
        }

    H, W = depth_np.shape[:2]
    if door_mask.shape[:2] != (H, W):
        door_mask = cv2.resize(door_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        hsv_mask = cv2.resize(hsv_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        combined_mask = cv2.bitwise_and(door_mask, hsv_mask)

    if rays_np.shape[:2] != (H, W):
        depth_np = cv2.resize(
            depth_np,
            (rays_np.shape[1], rays_np.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        door_mask = cv2.resize(
            door_mask,
            (rays_np.shape[1], rays_np.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        hsv_mask = cv2.resize(
            hsv_mask,
            (rays_np.shape[1], rays_np.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        combined_mask = cv2.bitwise_and(door_mask, hsv_mask)

    _, _, entries = _iter_component_bboxes(door_mask, min_area)
    rois = []
    for label, bbox, area in entries:
        roi_mask = _roi_mask_from_bbox(door_mask.shape[:2], bbox)
        roi_combined = cv2.bitwise_and(combined_mask, roi_mask)
        depth_roi = depth_np.copy()
        depth_roi[roi_combined == 0] = 0
        points_xyz = points_from_rays_and_depth(rays_np, depth_roi, stride=stride)
        rois.append(
            {
                "label": label,
                "bbox": bbox,
                "area": area,
                "roi_mask": roi_mask,
                "roi_combined_mask": roi_combined,
                "points_xyz": points_xyz,
            }
        )

    return {
        "door_mask_raw": door_mask,
        "hsv_mask": hsv_mask,
        "combined_mask": combined_mask,
        "rois": rois,
    }


def door_points_from_masks(
    door_mask_raw: np.ndarray,
    hsv_mask: np.ndarray,
    depth_m: np.ndarray,
    rays,
    stride: int = 4,
    min_area: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Aggregate points from all ROIs using the overlap of raw NN mask and HSV mask.
    """
    res = door_roi_pointclouds(
        door_mask_raw,
        hsv_mask,
        depth_m,
        rays,
        stride=stride,
        min_area=min_area,
    )
    points_all = []
    for roi in res.get("rois") or []:
        pts = roi.get("points_xyz")
        if isinstance(pts, np.ndarray) and pts.size > 0:
            points_all.append(pts)

    if points_all:
        res["points_xyz"] = np.concatenate(points_all, axis=0)
    else:
        res["points_xyz"] = np.empty((0, 3), dtype=np.float32)
    return res


def door_roi_pointcloud(
    door_mask_raw: np.ndarray,
    hsv_mask: np.ndarray,
    depth_m: np.ndarray,
    rays,
    stride: int = 4,
    min_area: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Backwards-compatible wrapper that returns the first ROI (if any).
    """
    res = door_roi_pointclouds(
        door_mask_raw,
        hsv_mask,
        depth_m,
        rays,
        stride=stride,
        min_area=min_area,
    )
    rois = res.get("rois") or []
    first = rois[0] if rois else {}
    return {
        "door_mask_raw": res.get("door_mask_raw"),
        "hsv_mask": res.get("hsv_mask"),
        "combined_mask": res.get("combined_mask"),
        "roi_bbox": first.get("bbox"),
        "roi_mask": first.get("roi_mask"),
        "roi_combined_mask": first.get("roi_combined_mask"),
        "points_xyz": first.get("points_xyz", np.empty((0, 3), dtype=np.float32)),
        "rois": rois,
    }
