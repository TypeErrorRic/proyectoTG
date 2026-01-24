"""
Fast wall-plane extraction using the existing wall mask and GPU least-squares fitting.

This avoids RANSAC by fitting planes directly to the wall pixels (optionally
refined once by distance-to-plane filtering), which is typically much faster.
"""
import time
from typing import Optional, Dict, Any, List

import cupy as cp
import numpy as np

from utilities.GroundDetection import _refine_plane
from src.models.wallDetection import wallDetection


def _to_cp(a, dtype=cp.float32):
    if a is None:
        return None
    return cp.asarray(a, dtype=dtype)


def _norm_up(up_axis) -> cp.ndarray:
    up = cp.asarray(up_axis, dtype=cp.float32)
    return up / (cp.linalg.norm(up) + 1e-9)


def get_wall_planes(
    mapaProfundidad: np.ndarray,
    imagenRGB: Optional[np.ndarray],
    rays_cp: cp.ndarray,
    H: int,
    W: int,
    wallParams: Optional[Dict[str, Any]] = None,
    wall_mask: Optional[np.ndarray] = None,
    ground_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Extract wall plane(s) from depth using an existing wall mask.

    If wall_mask is None, this function runs wallDetection (TensorRT) using
    imagenRGB and ground_mask to obtain it.

    Returns a dict with:
        - planes: list of { "n": <cp.ndarray>, "d": <cp.ndarray>, "num_inliers": int }
        - wall_mask: wall mask used (CPU, uint8) or None
    """
    if mapaProfundidad is None or rays_cp is None or H is None or W is None:
        return {"planes": [], "wall_mask": None}

    wallParams = wallParams or {}
    subsample_stride = max(1, int(wallParams.get("subsample_stride", 2) or 2))
    min_points = int(wallParams.get("min_points", 400) or 400)
    max_points_raw = wallParams.get("max_points", 120000)
    max_points = int(max_points_raw) if max_points_raw is not None else 120000
    dist_thresh = float(wallParams.get("dist_thresh", 0.03) or 0.03)
    refine = bool(wallParams.get("refine", True))
    refine_dist_mult = float(wallParams.get("refine_dist_mult", 1.6) or 1.6)
    max_planes = max(1, int(wallParams.get("max_planes", 1) or 1))
    enforce_vertical = bool(wallParams.get("enforce_vertical", True))
    max_up_dot = float(wallParams.get("max_up_dot", 0.35) or 0.35)
    up_axis = wallParams.get("up_axis", (0.0, -1.0, 0.0))
    return_cpu = bool(wallParams.get("return_cpu", False))
    debug_timing = bool(wallParams.get("debug_timing", False))

    t0 = time.perf_counter() if debug_timing else None

    if wall_mask is None:
        if imagenRGB is None:
            return {"planes": [], "wall_mask": None}
        if ground_mask is None:
            ground_mask = np.zeros((H, W), dtype=np.uint8)
        try:
            wall_mask, _ = wallDetection(mapaProfundidad, imagenRGB, ground_mask)
        except Exception as exc:
            print(f"[wall_planes] wallDetection failed: {exc}")
            return {"planes": [], "wall_mask": None}

    if wall_mask is None:
        return {"planes": [], "wall_mask": None}

    depth_cp = _to_cp(mapaProfundidad, dtype=cp.float32)
    rays_cp = _to_cp(rays_cp, dtype=cp.float32)
    wall_mask_cp = _to_cp(wall_mask, dtype=cp.uint8)

    if depth_cp is None or rays_cp is None or wall_mask_cp is None:
        return {"planes": [], "wall_mask": None}

    if wall_mask_cp.ndim == 3:
        wall_mask_cp = wall_mask_cp[:, :, 0]
    if ground_mask is not None:
        ground_mask_cp = _to_cp(ground_mask, dtype=cp.uint8)
        if ground_mask_cp is not None:
            if ground_mask_cp.ndim == 3:
                ground_mask_cp = ground_mask_cp[:, :, 0]
        else:
            ground_mask_cp = None
    else:
        ground_mask_cp = None

    if depth_cp.shape[:2] != rays_cp.shape[:2] or depth_cp.shape[:2] != wall_mask_cp.shape[:2]:
        return {"planes": [], "wall_mask": wall_mask}

    if subsample_stride > 1:
        depth_cp = depth_cp[::subsample_stride, ::subsample_stride]
        rays_cp = rays_cp[::subsample_stride, ::subsample_stride]
        wall_mask_cp = wall_mask_cp[::subsample_stride, ::subsample_stride]
        if ground_mask_cp is not None:
            ground_mask_cp = ground_mask_cp[::subsample_stride, ::subsample_stride]

    mask = wall_mask_cp > 0
    if ground_mask_cp is not None:
        mask = cp.logical_and(mask, ground_mask_cp == 0)

    depth_flat = depth_cp.reshape(-1)
    rays_flat = rays_cp.reshape(-1, 3)
    mask_flat = mask.reshape(-1)
    valid = cp.logical_and(mask_flat, depth_flat > 0)
    idx = cp.flatnonzero(valid)

    if int(idx.size) < min_points:
        return {"planes": [], "wall_mask": wall_mask}

    if max_points and int(idx.size) > max_points:
        try:
            idx = cp.random.choice(idx, size=max_points, replace=False)
        except Exception:
            idx = idx[:max_points]

    pts = rays_flat[idx] * depth_flat[idx, None]
    up_vec = _norm_up(up_axis)

    planes: List[Dict[str, Any]] = []
    active = cp.ones((int(pts.shape[0]),), dtype=cp.bool_)

    for _ in range(max_planes):
        pts_active = pts[active]
        if int(pts_active.shape[0]) < min_points:
            break

        fit = _refine_plane(pts_active, up_vec, "any")
        if fit is None:
            break
        n, d = fit

        if enforce_vertical:
            dot_up = cp.abs(cp.dot(n, up_vec))
            if float(dot_up.get()) > max_up_dot:
                break

        if refine:
            signed = pts_active @ n + d
            inliers = cp.abs(signed) <= (dist_thresh * refine_dist_mult)
            if int(inliers.sum().get()) >= min_points:
                fit2 = _refine_plane(pts_active[inliers], up_vec, "any")
                if fit2 is not None:
                    n, d = fit2

        signed = pts_active @ n + d
        inliers = cp.abs(signed) <= dist_thresh
        num_inliers = int(inliers.sum().get())

        if return_cpu:
            n_out = cp.asnumpy(n)
            d_out = float(cp.asnumpy(d))
        else:
            n_out = n
            d_out = d

        planes.append(
            {
                "n": n_out,
                "d": d_out,
                "num_inliers": num_inliers,
            }
        )

        if max_planes <= 1:
            break

        active_idx = cp.flatnonzero(active)
        if int(inliers.size) != int(active_idx.size):
            break
        active[active_idx[inliers]] = False

    if debug_timing and t0 is not None:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[wall_planes] total_ms={elapsed_ms:.2f}")

    return {"planes": planes, "wall_mask": wall_mask}
