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


def _densest_seed(points: np.ndarray, voxel_size: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Find the densest voxel and return its centroid as seed plus a mask of points in that voxel.
    """
    if points is None or points.size == 0:
        return None, None
    if voxel_size <= 0:
        return None, None

    pmin = points.min(axis=0)
    idx = np.floor((points - pmin) / float(voxel_size)).astype(np.int32)
    voxels, counts = np.unique(idx, axis=0, return_counts=True)
    if voxels.size == 0:
        return None, None

    densest = voxels[int(np.argmax(counts))]
    mask = np.all(idx == densest, axis=1)
    if not np.any(mask):
        return None, None
    seed = points[mask].mean(axis=0)
    return seed, mask


def _fit_plane_pca(points: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float], Optional[np.ndarray]]:
    """
    Fit a plane to points via PCA. Returns (normal, d, centroid).
    """
    if points is None or points.shape[0] < 3:
        return None, None, None
    centroid = points.mean(axis=0)
    X = points - centroid
    cov = (X.T @ X) / max(1, X.shape[0])
    vals, vecs = np.linalg.eigh(cov)
    normal = vecs[:, int(np.argmin(vals))]
    n_norm = float(np.linalg.norm(normal))
    if n_norm < 1e-9:
        return None, None, None
    normal = normal / n_norm
    d = -float(np.dot(normal, centroid))
    return normal, d, centroid


def _is_parallel(normal: np.ndarray, ground_normal: np.ndarray, max_angle_deg: float) -> bool:
    """
    Check if normals are parallel within max_angle_deg.
    """
    if normal is None or ground_normal is None:
        return False
    n1 = np.asarray(normal, dtype=np.float32).reshape(-1)
    n2 = np.asarray(ground_normal, dtype=np.float32).reshape(-1)
    if n1.size != 3 or n2.size != 3:
        return False
    n1 /= max(1e-9, float(np.linalg.norm(n1)))
    n2 /= max(1e-9, float(np.linalg.norm(n2)))
    dot = float(abs(np.dot(n1, n2)))
    dot = max(-1.0, min(1.0, dot))
    angle = float(np.degrees(np.arccos(dot)))
    return angle <= float(max_angle_deg)


def _line_parallel_to_ground(
    points: np.ndarray,
    ground_normal: np.ndarray,
    max_angle_deg: float,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], float]:
    """
    Check if two points in 'points' can form a line parallel to ground with length >= min_length_m.
    Returns (ok, p1, p2, length).
    """
    if points is None or points.shape[0] < 2:
        return False, None, None, 0.0
    if ground_normal is None:
        return False, None, None, 0.0

    n = np.asarray(ground_normal, dtype=np.float32).reshape(-1)
    if n.size != 3:
        return False, None, None, 0.0
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-9:
        return False, None, None, 0.0
    n = n / n_norm

    # Build orthonormal basis (u, v) for the ground plane.
    axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(axis, n))) > 0.9:
        axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u = np.cross(n, axis)
    u /= max(1e-9, float(np.linalg.norm(u)))
    v = np.cross(n, u)
    v /= max(1e-9, float(np.linalg.norm(v)))

    # Project points to 2D coords on ground plane.
    coords = np.stack((points @ u, points @ v), axis=1)
    if coords.shape[0] < 2:
        return False, None, None, 0.0

    # Find farthest pair along the principal axis in 2D.
    centered = coords - coords.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(1, centered.shape[0])
    vals, vecs = np.linalg.eigh(cov)
    dir2 = vecs[:, int(np.argmax(vals))]
    proj = coords @ dir2
    i_min = int(np.argmin(proj))
    i_max = int(np.argmax(proj))
    p1 = points[i_min]
    p2 = points[i_max]
    vec = p2 - p1
    length = float(np.linalg.norm(vec))
    # Check line direction parallel to ground plane (orthogonal to normal).
    dot = float(abs(np.dot(vec, n)) / max(1e-9, length))
    # Convert max_angle_deg to dot threshold.
    max_angle_deg = float(max_angle_deg)
    max_dot = float(np.sin(np.deg2rad(max_angle_deg)))
    if dot > max_dot:
        return False, p1, p2, length

    return True, p1, p2, length


def door_roi_pointclouds(
    door_mask_raw: np.ndarray,
    hsv_mask: np.ndarray,
    depth_m: np.ndarray,
    rays,
    stride: int = 4,
    min_area: Optional[int] = None,
    ground_normal=None,
    ground_parallel_deg: float = 15.0,
    ground_line_parallel_deg: float = 10.0,
    density_voxel: float = 0.05,
    seed_radius_ratio: float = 0.12,
    min_plane_points: int = 50,
    max_density_points: int = 20000,
) -> Dict[str, Any]:
    """
    Compute door ROIs using the raw NN mask and return point clouds per ROI.

    HSV mask is provided by the caller. Points are taken from the overlap
    between raw NN mask and HSV mask, constrained by each ROI.
    ROIs are discarded if the fitted plane is not parallel to ground or if
    there is no line parallel to ground using border points of the ROI mask.

    Returns:
        dict with:
          - door_mask_raw: uint8 mask (H, W)
          - hsv_mask: uint8 mask (H, W)
          - combined_mask: uint8 mask (H, W) [raw & hsv]
          - rois: list of dicts per ROI:
                {label, bbox, roi_mask, roi_combined_mask, points_xyz, plane_n, plane_d}
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

    combined_mask_raw = cv2.bitwise_and(door_mask, hsv_mask)

    depth_np = _to_numpy(depth_m)
    rays_np = _to_numpy(rays)
    if depth_np is None or rays_np is None:
        return {
            "door_mask_raw": door_mask,
            "hsv_mask": hsv_mask,
            "combined_mask": combined_mask_raw,
            "rois": [],
        }

    H, W = depth_np.shape[:2]
    if door_mask.shape[:2] != (H, W):
        door_mask = cv2.resize(door_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        hsv_mask = cv2.resize(hsv_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        combined_mask_raw = cv2.bitwise_and(door_mask, hsv_mask)

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
        combined_mask_raw = cv2.bitwise_and(door_mask, hsv_mask)

    _, _, entries = _iter_component_bboxes(door_mask, min_area)
    ground_n = _to_numpy(ground_normal)
    rois = []
    hsv_mask_filtered = np.zeros_like(hsv_mask, dtype=np.uint8)
    for label, bbox, area in entries:
        roi_mask = _roi_mask_from_bbox(door_mask.shape[:2], bbox)
        roi_combined = cv2.bitwise_and(combined_mask_raw, roi_mask)
        # Border points (external points within the mask)
        mask_bin = (roi_combined > 0).astype(np.uint8)
        if np.any(mask_bin):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            eroded = cv2.erode(mask_bin, kernel, iterations=1)
            border = mask_bin & (eroded == 0)
            border_mask = (border * 255).astype(np.uint8)
        else:
            border_mask = np.zeros_like(roi_combined, dtype=np.uint8)
        depth_roi = depth_np.copy()
        depth_roi[roi_combined == 0] = 0
        points_xyz = points_from_rays_and_depth(rays_np, depth_roi, stride=stride)
        depth_border = depth_np.copy()
        depth_border[border_mask == 0] = 0
        border_points = points_from_rays_and_depth(
            rays_np, depth_border, stride=stride
        )
        plane_n = None
        plane_d = None
        keep = True
        if isinstance(points_xyz, np.ndarray) and points_xyz.shape[0] >= min_plane_points:
            pts_all = points_xyz
            pts_for_seed = pts_all
            if max_density_points and pts_for_seed.shape[0] > max_density_points:
                idx = np.random.choice(
                    pts_for_seed.shape[0], size=max_density_points, replace=False
                )
                pts_for_seed = pts_for_seed[idx]
            seed, _ = _densest_seed(pts_for_seed, density_voxel)
            if seed is None:
                keep = False
            else:
                # Only keep points within a radius based on the ROI 3D diagonal.
                pmin = pts_all.min(axis=0)
                pmax = pts_all.max(axis=0)
                diag = float(np.linalg.norm(pmax - pmin))
                radius = diag * float(seed_radius_ratio)
                if radius <= 0.0:
                    keep = False
                    near = np.empty((0, 3), dtype=np.float32)
                else:
                    diff = pts_all - seed
                    dist2 = np.sum(diff * diff, axis=1)
                    near_mask = dist2 <= radius ** 2
                    near = pts_all[near_mask]
                if near.shape[0] >= min_plane_points:
                    plane_n, plane_d, _ = _fit_plane_pca(near)
                    if plane_n is None:
                        keep = False
                    points_xyz = near
                else:
                    keep = False
        else:
            keep = False

        if keep and ground_n is not None:
            keep = _is_parallel(plane_n, ground_n, ground_parallel_deg)

        if keep and ground_n is not None:
            ok_line, _, _, _ = _line_parallel_to_ground(
                border_points,
                ground_n,
                ground_line_parallel_deg,
            )
            if not ok_line:
                keep = False

        if not keep:
            continue

        hsv_keep = cv2.bitwise_and(hsv_mask, roi_mask)
        hsv_mask_filtered = cv2.bitwise_or(hsv_mask_filtered, hsv_keep)

        rois.append(
            {
                "label": label,
                "bbox": bbox,
                "area": area,
                "roi_mask": roi_mask,
                "roi_combined_mask": roi_combined,
                "points_xyz": points_xyz,
                "plane_n": plane_n,
                "plane_d": plane_d,
            }
        )

    combined_mask_filtered = cv2.bitwise_and(door_mask, hsv_mask_filtered)

    return {
        "door_mask_raw": door_mask,
        "hsv_mask": hsv_mask_filtered,
        "combined_mask": combined_mask_filtered,
        "rois": rois,
    }


def door_points_from_masks(
    door_mask_raw: np.ndarray,
    hsv_mask: np.ndarray,
    depth_m: np.ndarray,
    rays,
    stride: int = 4,
    min_area: Optional[int] = None,
    ground_normal=None,
    ground_parallel_deg: float = 15.0,
    ground_line_parallel_deg: float = 10.0,
    density_voxel: float = 0.05,
    seed_radius_ratio: float = 0.12,
    min_plane_points: int = 50,
    max_density_points: int = 20000,
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
        ground_normal=ground_normal,
        ground_parallel_deg=ground_parallel_deg,
        ground_line_parallel_deg=ground_line_parallel_deg,
        density_voxel=density_voxel,
        seed_radius_ratio=seed_radius_ratio,
        min_plane_points=min_plane_points,
        max_density_points=max_density_points,
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
    ground_normal=None,
    ground_parallel_deg: float = 15.0,
    ground_line_parallel_deg: float = 10.0,
    density_voxel: float = 0.05,
    seed_radius_ratio: float = 0.12,
    min_plane_points: int = 50,
    max_density_points: int = 20000,
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
        ground_normal=ground_normal,
        ground_parallel_deg=ground_parallel_deg,
        ground_line_parallel_deg=ground_line_parallel_deg,
        density_voxel=density_voxel,
        seed_radius_ratio=seed_radius_ratio,
        min_plane_points=min_plane_points,
        max_density_points=max_density_points,
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
