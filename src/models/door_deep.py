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


def door_roi_pointclouds(
    door_mask_raw: np.ndarray,
    hsv_mask: np.ndarray,
    depth_m: np.ndarray,
    rays,
    stride: int = 4,
    min_area: Optional[int] = None,
    ground_normal=None,
    ground_parallel_deg: float = 15.0,
    density_voxel: float = 0.05,
    seed_radius_ratio: float = 0.12,
    min_plane_points: int = 50,
    max_density_points: int = 20000,
    plane_inlier_dist: float = 0.003,
    plane_inlier_ratio: float = 0.70,
    debug_print: bool = True,
) -> Dict[str, Any]:
    """
    Compute door ROIs using the raw NN mask and return point clouds per ROI.

    HSV mask is provided by the caller. Points are taken from the overlap
    between raw NN mask and HSV mask, constrained by each ROI.
    ROIs are discarded if the fitted plane is not parallel to ground or if
    fewer than plane_inlier_ratio of points lie within plane_inlier_dist.

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
    if debug_print:
        print(f"[door_deep] ROIs detectados (candidatos): {len(entries)}")
    for label, bbox, area in entries:
        roi_mask = _roi_mask_from_bbox(door_mask.shape[:2], bbox)
        roi_combined = cv2.bitwise_and(combined_mask_raw, roi_mask)
        depth_roi = depth_np.copy()
        depth_roi[roi_combined == 0] = 0
        points_xyz = points_from_rays_and_depth(rays_np, depth_roi, stride=stride)
        plane_n = None
        plane_d = None
        pts_all = None
        inlier_ratio = None
        angle_deg = None
        angle_ref = None
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

        if plane_n is not None and ground_n is not None:
            angle_ref = "ground"
            angle_deg = _angle_between_normals_deg(plane_n, ground_n)

        if plane_n is not None and plane_d is not None and pts_all is not None:
            dist = np.abs(pts_all @ plane_n + plane_d)
            if dist.size > 0:
                inlier_ratio = float(np.mean(dist <= float(plane_inlier_dist)))

        if keep and angle_deg is not None:
            keep = angle_deg >= (90.0 - float(ground_parallel_deg))

        if keep and inlier_ratio is not None:
            if inlier_ratio < float(plane_inlier_ratio):
                keep = False

        if debug_print:
            pts_count = (
                int(points_xyz.shape[0])
                if isinstance(points_xyz, np.ndarray)
                else 0
            )
            angle_txt = (
                f"{angle_deg:.2f} deg ({angle_ref})"
                if angle_deg is not None
                else "N/A"
            )
            ratio_txt = (
                f"{inlier_ratio * 100.0:.1f}%"
                if inlier_ratio is not None
                else "N/A"
            )
            print(
                f"[door_deep] ROI {label}: bbox={bbox} area={area} "
                f"pts={pts_count} angle={angle_txt} inliers={ratio_txt} keep={keep}"
            )

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

    if debug_print:
        print(f"[door_deep] ROIs validos (filtrados): {len(rois)}")

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
    density_voxel: float = 0.05,
    seed_radius_ratio: float = 0.12,
    min_plane_points: int = 50,
    max_density_points: int = 20000,
    plane_inlier_dist: float = 0.003,
    plane_inlier_ratio: float = 0.70,
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
        density_voxel=density_voxel,
        seed_radius_ratio=seed_radius_ratio,
        min_plane_points=min_plane_points,
        max_density_points=max_density_points,
        plane_inlier_dist=plane_inlier_dist,
        plane_inlier_ratio=plane_inlier_ratio,
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
    density_voxel: float = 0.05,
    seed_radius_ratio: float = 0.12,
    min_plane_points: int = 50,
    max_density_points: int = 20000,
    plane_inlier_dist: float = 0.003,
    plane_inlier_ratio: float = 0.70,
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
        density_voxel=density_voxel,
        seed_radius_ratio=seed_radius_ratio,
        min_plane_points=min_plane_points,
        max_density_points=max_density_points,
        plane_inlier_dist=plane_inlier_dist,
        plane_inlier_ratio=plane_inlier_ratio,
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
