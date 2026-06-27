"""
Door ROI + point cloud segmentation using precomputed masks.

Uses:
  - Raw NN mask + HSV mask overlap for ROI extraction.
  - HSV mask (already computed elsewhere).
  - Depth map + rays to build point clouds inside ROI & mask overlap.
"""
from typing import Optional, Tuple, Dict, Any

import time
import cv2
import numpy as np

from src.application.pipeline_utils import geometria

DEBUG_DOOR_DEEP = False


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


def _iter_component_bboxes(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Iterate component bboxes.
    Returns (labels, stats, entries), entries is list of (label, bbox, area).
    """
    labels, stats = _component_stats(mask)
    if labels is None or stats is None:
        return None, None, []

    entries = []
    for label in range(1, stats.shape[0]):
        area = int(stats[label, cv2.CC_STAT_AREA])
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


def _fit_plane_trimmed_pca(
    points: np.ndarray, keep_ratio: float = 0.7, iters: int = 2
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[np.ndarray]]:
    """
    Fast robust plane fit: PCA + trimming repeated a few iterations.
    Returns (normal, d, trimmed_points).
    """
    if points is None or points.shape[0] < 3:
        return None, None, None
    keep_ratio = float(keep_ratio)
    if keep_ratio <= 0.0:
        keep_ratio = 0.7
    if keep_ratio > 1.0:
        keep_ratio = 1.0
    iters = max(1, int(iters))

    pts = points
    for _ in range(iters):
        normal, d, _ = _fit_plane_pca(pts)
        if normal is None or d is None:
            return None, None, None
        dist = np.abs(pts @ normal + d)
        k = max(3, int(dist.size * keep_ratio))
        idx = np.argpartition(dist, k - 1)[:k]
        pts = pts[idx]
    return normal, d, pts


def _angle_between_normals_deg(
    normal: np.ndarray, ref_normal: np.ndarray
) -> Optional[float]:
    if normal is None or ref_normal is None:
        return None
    n1 = np.asarray(normal, dtype=np.float32).reshape(-1)
    n2 = np.asarray(ref_normal, dtype=np.float32).reshape(-1)
    if n1.size != 3 or n2.size != 3:
        return None
    n1 /= max(1e-9, float(np.linalg.norm(n1)))
    n2 /= max(1e-9, float(np.linalg.norm(n2)))
    dot = float(abs(np.dot(n1, n2)))
    dot = max(-1.0, min(1.0, dot))
    return float(np.degrees(np.arccos(dot)))


def _merge_close_regions(mask: np.ndarray, merge_gap_px: int) -> np.ndarray:
    if mask is None or mask.size == 0:
        return mask
    if not merge_gap_px or int(merge_gap_px) <= 0:
        return mask
    k = 2 * int(merge_gap_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # Close small gaps so nearby ROIs merge into one.
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _refine_mask_edges(
    seed_mask: np.ndarray,
    allowed_mask: np.ndarray,
    bgr_image: np.ndarray,
    edge_sigma: float = 0.33,
    edge_dilate: int = 1,
    max_iters: int = 64,
    min_island_pixels: int = 300,
    use_realsense: bool = True,
) -> np.ndarray:
    """
    Refine a mask using RGB edges as barriers and geodesic growth.
    """
    if seed_mask is None or seed_mask.size == 0:
        return seed_mask
    if bgr_image is None or bgr_image.size == 0:
        return seed_mask

    seed = np.asarray(seed_mask)
    if seed.ndim == 3:
        seed = seed[:, :, 0]

    allowed = np.asarray(allowed_mask) if allowed_mask is not None else seed
    if allowed.ndim == 3:
        allowed = allowed[:, :, 0]

    H, W = seed.shape[:2]
    if allowed.shape[:2] != (H, W):
        allowed = cv2.resize(allowed, (W, H), interpolation=cv2.INTER_NEAREST)

    img = np.asarray(bgr_image)
    if img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    med = float(np.median(gray))
    lower = int(max(0, (1.0 - edge_sigma) * med))
    upper = int(min(255, (1.0 + edge_sigma) * med))
    if lower == upper:
        lower = max(0, int(med * 0.5))
        upper = min(255, int(med * 1.5))
        if lower == upper:
            lower, upper = 50, 150

    edges = cv2.Canny(gray, lower, upper)
    if edge_dilate and edge_dilate > 0:
        k = 2 * int(edge_dilate) + 1
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        edges = cv2.dilate(edges, kernel_edge, iterations=1)
    barrier = edges > 0

    allowed_region = (allowed > 0) & (~barrier)
    current = (seed > 0) & allowed_region
    if not np.any(current):
        return seed

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(max_iters):
        dil = cv2.dilate(current.astype(np.uint8), kernel, iterations=1) > 0
        nxt = dil & allowed_region
        if np.array_equal(nxt, current):
            break
        current = nxt

    out = np.zeros((H, W), dtype=np.uint8)
    out[current] = 255

    if min_island_pixels and min_island_pixels > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            out, connectivity=8
        )
        if num_labels > 1:
            keep = np.zeros_like(out)
            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                if area >= min_island_pixels:
                    keep[labels == label] = 255
            out = keep

    if use_realsense:
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        out = cv2.dilate(out, kernel_dilate, iterations=1)

    return out


def door_roi_pointclouds(
    door_mask_raw: np.ndarray,
    hsv_mask: np.ndarray,
    depth_m: np.ndarray,
    rays,
    imagen_rgb: Optional[np.ndarray] = None,
    stride: int = 4,
    ground_normal=None,
    ground_parallel_deg: float = 15.0,
    merge_gap_px: int = 20,
    density_voxel: float = 0.05,
    seed_radius_ratio: float = 0.12,
    min_plane_points: int = 50,
    max_density_points: int = 20000,
    plane_inlier_dist: float = 0.02,
    plane_inlier_ratio: float = 0.30,
    trim_keep_ratio: float = 0.70,
    trim_iters: int = 2,
    edge_sigma: float = 0.33,
    edge_dilate: int = 1,
    max_iters: int = 64,
    min_island_pixels: int = 300,
    use_realsense: bool = True,
) -> Dict[str, Any]:
    """
    Compute door ROIs using the overlap of raw NN mask and HSV mask, and return
    point clouds per ROI.

    HSV mask is provided by the caller. Points are taken from the overlap
    between raw NN mask and HSV mask, constrained by each ROI.
    If merge_gap_px > 0, nearby regions are merged before extracting ROIs.
    ROIs are discarded if the fitted plane is not parallel to ground or if
    fewer than plane_inlier_ratio of points lie within plane_inlier_dist.
    Uses a fast trimmed PCA (iterative inlier pruning) for robustness.
    Note: density_voxel and seed_radius_ratio are kept for compatibility,
    but are not used in the trimmed PCA path.

    Returns:
        dict with:
          - door_mask_raw: uint8 mask (H, W)
          - hsv_mask: uint8 mask (H, W)
          - combined_mask: uint8 mask (H, W) [raw & hsv]
          - rois: list of dicts per ROI:
                {label, bbox, roi_mask, roi_combined_mask, points_xyz, plane_n, plane_d}
    """
    debug_print = DEBUG_DOOR_DEEP
    timings = {}
    t_start = time.perf_counter() if debug_print else None

    if door_mask_raw is None or hsv_mask is None:
        raise ValueError("door_mask_raw and hsv_mask are required.")

    t_stage = time.perf_counter() if debug_print else None
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

    combined_mask_raw = cv2.bitwise_and(door_mask, hsv_mask)
    combined_mask_rois = _merge_close_regions(combined_mask_raw, merge_gap_px)
    if debug_print:
        timings["prep_masks_s"] = time.perf_counter() - t_stage

    t_stage = time.perf_counter() if debug_print else None
    depth_np = _to_numpy(depth_m)
    rays_np = _to_numpy(rays)
    if depth_np is None or rays_np is None:
        return {
            "door_mask_raw": door_mask,
            "hsv_mask": hsv_mask,
            "combined_mask": combined_mask_raw,
            "rois": [],
            "timings": timings,
        }

    H, W = depth_np.shape[:2]
    if door_mask.shape[:2] != (H, W):
        door_mask = cv2.resize(door_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        hsv_mask = cv2.resize(hsv_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        combined_mask_raw = cv2.bitwise_and(door_mask, hsv_mask)
        combined_mask_rois = _merge_close_regions(combined_mask_raw, merge_gap_px)

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
        combined_mask_rois = _merge_close_regions(combined_mask_raw, merge_gap_px)
    if debug_print:
        timings["align_depth_rays_s"] = time.perf_counter() - t_stage

    t_stage = time.perf_counter() if debug_print else None
    _, _, entries = _iter_component_bboxes(combined_mask_rois)
    ground_n = _to_numpy(ground_normal)
    rois = []
    hsv_mask_filtered = np.zeros_like(hsv_mask, dtype=np.uint8)
    if debug_print:
        timings["roi_components_s"] = time.perf_counter() - t_stage

    t_stage = time.perf_counter() if debug_print else None
    for label, bbox, area in entries:
        roi_mask = _roi_mask_from_bbox(door_mask.shape[:2], bbox)
        roi_combined = cv2.bitwise_and(combined_mask_raw, roi_mask)
        depth_roi = depth_np.copy()
        depth_roi[roi_combined == 0] = 0
        points_xyz = geometria.points_from_rays_and_depth(rays_np, depth_roi, stride=stride)
        plane_n = None
        plane_d = None
        pts_all = None
        inlier_ratio = None
        inlier_mask = None
        angle_deg = None
        angle_ref = None
        keep = True
        if isinstance(points_xyz, np.ndarray) and points_xyz.shape[0] >= min_plane_points:
            pts_all = points_xyz
            pts_for_fit = pts_all
            if max_density_points and pts_for_fit.shape[0] > max_density_points:
                idx = np.random.choice(
                    pts_for_fit.shape[0], size=max_density_points, replace=False
                )
                pts_for_fit = pts_for_fit[idx]
            plane_n, plane_d, _ = _fit_plane_trimmed_pca(
                pts_for_fit, keep_ratio=trim_keep_ratio, iters=trim_iters
            )
            if plane_n is None or plane_d is None:
                keep = False
        else:
            keep = False

        if plane_n is not None and ground_n is not None:
            angle_ref = "ground"
            angle_deg = _angle_between_normals_deg(plane_n, ground_n)

        if plane_n is not None and plane_d is not None and pts_all is not None:
            dist = np.abs(pts_all @ plane_n + plane_d)
            if dist.size > 0:
                inlier_mask = dist <= float(plane_inlier_dist)
                inlier_ratio = float(np.mean(inlier_mask))
                points_xyz = pts_all[inlier_mask]
                if points_xyz.shape[0] < min_plane_points:
                    keep = False

        if keep and angle_deg is not None:
            keep = angle_deg >= (90.0 - float(ground_parallel_deg))

        if keep and inlier_ratio is not None:
            if inlier_ratio < float(plane_inlier_ratio):
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
    if debug_print:
        timings["roi_plane_filter_s"] = time.perf_counter() - t_stage

    t_stage = time.perf_counter() if debug_print else None
    combined_mask_filtered = cv2.bitwise_and(door_mask, hsv_mask_filtered)
    if imagen_rgb is not None:
        combined_mask_filtered = _refine_mask_edges(
            combined_mask_filtered,
            hsv_mask_filtered,
            imagen_rgb,
            edge_sigma=edge_sigma,
            edge_dilate=edge_dilate,
            max_iters=max_iters,
            min_island_pixels=min_island_pixels,
            use_realsense=use_realsense,
        )
    if debug_print:
        timings["refine_mask_edges_s"] = time.perf_counter() - t_stage
        timings["total_s"] = time.perf_counter() - t_start

    if debug_print:
        print(
            "[door_deep] tiempos (s): "
            f"prep={timings.get('prep_masks_s', 0):.4f} "
            f"align={timings.get('align_depth_rays_s', 0):.4f} "
            f"roi={timings.get('roi_components_s', 0):.4f} "
            f"fit={timings.get('roi_plane_filter_s', 0):.4f} "
            f"refine={timings.get('refine_mask_edges_s', 0):.4f} "
            f"total={timings.get('total_s', 0):.4f}"
        )

    return {
        "door_mask_raw": door_mask,
        "hsv_mask": hsv_mask_filtered,
        "combined_mask": combined_mask_filtered,
        "rois": rois,
        "timings": timings,
    }


def door_points_from_masks(
    door_mask_raw: np.ndarray,
    hsv_mask: np.ndarray,
    depth_m: np.ndarray,
    rays,
    imagen_rgb: Optional[np.ndarray] = None,
    stride: int = 4,
    ground_normal=None,
    ground_parallel_deg: float = 15.0,
    merge_gap_px: int = 20,
    density_voxel: float = 0.05,
    seed_radius_ratio: float = 0.12,
    min_plane_points: int = 50,
    max_density_points: int = 20000,
    plane_inlier_dist: float = 0.02,
    plane_inlier_ratio: float = 0.30,
    trim_keep_ratio: float = 0.70,
    trim_iters: int = 2,
    edge_sigma: float = 0.33,
    edge_dilate: int = 1,
    max_iters: int = 64,
    min_island_pixels: int = 300,
    use_realsense: bool = True,
) -> Dict[str, Any]:
    """
    Aggregate points from all ROIs using the overlap of raw NN mask and HSV mask.
    """
    res = door_roi_pointclouds(
        door_mask_raw,
        hsv_mask,
        depth_m,
        rays,
        imagen_rgb=imagen_rgb,
        stride=stride,
        ground_normal=ground_normal,
        ground_parallel_deg=ground_parallel_deg,
        merge_gap_px=merge_gap_px,
        density_voxel=density_voxel,
        seed_radius_ratio=seed_radius_ratio,
        min_plane_points=min_plane_points,
        max_density_points=max_density_points,
        plane_inlier_dist=plane_inlier_dist,
        plane_inlier_ratio=plane_inlier_ratio,
        trim_keep_ratio=trim_keep_ratio,
        trim_iters=trim_iters,
        edge_sigma=edge_sigma,
        edge_dilate=edge_dilate,
        max_iters=max_iters,
        min_island_pixels=min_island_pixels,
        use_realsense=use_realsense,
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


