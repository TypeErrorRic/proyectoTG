"""
Fast wall-plane extraction using GPU RANSAC (no ML model).

Uses the existing RANSAC implementation with a vertical orientation constraint
and builds a wall mask from the fitted plane(s).
"""
import time
from typing import Optional, Dict, Any, List

import cupy as cp
import numpy as np

from utilities.GroundDetection import ransac_plane_gpu, _refine_plane

# Variable para activar la impresion de tiempos de cada etapa en get_wall_planes
DEBUG_TIMING = True


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
    Extract wall plane(s) from depth using GPU RANSAC.

    Returns a dict with:
        - planes: list of { "n": <cp.ndarray>, "d": <cp.ndarray>, "num_inliers": int }
        - wall_mask: wall mask used (CPU, uint8) or None
    """
    if mapaProfundidad is None or rays_cp is None or H is None or W is None:
        return {"planes": [], "wall_mask": None}

    wallParams = wallParams or {}
    debug_timing = bool(wallParams.get("debug_timing", False))
    timing_on = DEBUG_TIMING or debug_timing

    if timing_on:
        _t_total_start = time.perf_counter()
        _t_params_start = time.perf_counter()
    subsample_stride = max(1, int(wallParams.get("subsample_stride", 2) or 2))
    min_points = int(wallParams.get("min_points", 400) or 400)
    max_points_raw = wallParams.get("max_points", 120000)
    max_points = int(max_points_raw) if max_points_raw is not None else 120000
    dist_thresh = float(wallParams.get("dist_thresh", 0.03) or 0.03)
    max_iters = int(wallParams.get("max_iters", 300) or 300)
    max_angle_deg = float(wallParams.get("max_angle_deg", 20.0) or 20.0)
    score_subset = wallParams.get("score_subset", 4096)
    score_subset = int(score_subset) if score_subset is not None else 4096
    batch_size = wallParams.get("batch_size", 1024)
    batch_size = int(batch_size) if batch_size is not None else None
    early_stop_ratio = float(wallParams.get("early_stop_ratio", 0.90) or 0.90)
    refine = bool(wallParams.get("refine", True))
    refine_dist_mult = float(wallParams.get("refine_dist_mult", 1.6) or 1.6)
    max_planes = max(1, int(wallParams.get("max_planes", 1) or 1))
    enforce_vertical = bool(wallParams.get("enforce_vertical", True))
    max_up_dot = float(wallParams.get("max_up_dot", 0.35) or 0.35)
    up_axis = wallParams.get("up_axis", (0.0, -1.0, 0.0))
    return_cpu = bool(wallParams.get("return_cpu", False))

    if timing_on:
        _t_params_end = time.perf_counter()
        _t_params_ms = (_t_params_end - _t_params_start) * 1000.0
        _t_convert_start = time.perf_counter()

    depth_full = _to_cp(mapaProfundidad, dtype=cp.float32)
    rays_full = _to_cp(rays_cp, dtype=cp.float32)
    if depth_full is None or rays_full is None:
        return {"planes": [], "wall_mask": None}

    if ground_mask is not None:
        ground_mask_cp = _to_cp(ground_mask, dtype=cp.uint8)
        if ground_mask_cp is not None:
            if ground_mask_cp.ndim == 3:
                ground_mask_cp = ground_mask_cp[:, :, 0]
        else:
            ground_mask_cp = None
    else:
        ground_mask_cp = None

    if timing_on:
        _t_convert_end = time.perf_counter()
        _t_convert_ms = (_t_convert_end - _t_convert_start) * 1000.0

    if depth_full.shape[:2] != rays_full.shape[:2]:
        return {"planes": [], "wall_mask": None}

    if timing_on:
        _t_subsample_start = time.perf_counter()

    depth_cp = depth_full
    rays_cp = rays_full
    if subsample_stride > 1:
        depth_cp = depth_cp[::subsample_stride, ::subsample_stride]
        rays_cp = rays_cp[::subsample_stride, ::subsample_stride]
        if ground_mask_cp is not None:
            ground_mask_cp = ground_mask_cp[::subsample_stride, ::subsample_stride]

    if timing_on:
        _t_subsample_end = time.perf_counter()
        _t_subsample_ms = (_t_subsample_end - _t_subsample_start) * 1000.0
        _t_pointcloud_start = time.perf_counter()

    mask = depth_cp > 0
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

    if timing_on:
        _t_pointcloud_end = time.perf_counter()
        _t_pointcloud_ms = (_t_pointcloud_end - _t_pointcloud_start) * 1000.0
        _t_ransac_ms = 0.0
        _t_refine_ms = 0.0

    planes: List[Dict[str, Any]] = []
    active = cp.ones((int(pts.shape[0]),), dtype=cp.bool_)

    for _ in range(max_planes):
        pts_active = pts[active]
        if int(pts_active.shape[0]) < min_points:
            break

        if timing_on:
            _t_ransac_start = time.perf_counter()

        res = ransac_plane_gpu(
            pts_active,
            dist_thresh=dist_thresh,
            max_iters=max_iters,
            min_inliers=min_points,
            up_axis=up_axis,
            max_angle_deg=max_angle_deg,
            seed=42,
            score_subset=score_subset,
            orientation="vertical",
            early_stop_ratio=early_stop_ratio,
            batch_size=batch_size,
        )
        if res is None:
            break
        n, d = res["n"], res["d"]

        if timing_on:
            _t_ransac_end = time.perf_counter()
            _t_ransac_ms += (_t_ransac_end - _t_ransac_start) * 1000.0

        if enforce_vertical:
            dot_up = cp.abs(cp.dot(n, up_vec))
            if float(dot_up.get()) > max_up_dot:
                break

        if refine:
            if timing_on:
                _t_refine_start = time.perf_counter()

            signed = pts_active @ n + d
            inliers = cp.abs(signed) <= (dist_thresh * refine_dist_mult)
            if int(inliers.sum().get()) >= min_points:
                fit2 = _refine_plane(pts_active[inliers], up_vec, "any")
                if fit2 is not None:
                    n, d = fit2
            if timing_on:
                _t_refine_end = time.perf_counter()
                _t_refine_ms += (_t_refine_end - _t_refine_start) * 1000.0

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

    if timing_on:
        _t_mask_start = time.perf_counter()

    if planes:
        mask_cp = cp.zeros((H, W), dtype=cp.bool_)
        gm_full = None
        if ground_mask is not None:
            gm_full = _to_cp(ground_mask, dtype=cp.uint8)
            if gm_full is not None and gm_full.ndim == 3:
                gm_full = gm_full[:, :, 0]
        for plane in planes:
            n_use = plane["n"] if not return_cpu else _to_cp(plane["n"])
            d_use = plane["d"] if not return_cpu else _to_cp(plane["d"])
            dotnr = cp.tensordot(rays_full, n_use, axes=([2], [0]))
            dists = cp.abs(depth_full * dotnr + d_use)
            mask_i = (dists <= dist_thresh) & (depth_full > 0)
            if gm_full is not None and gm_full.shape[:2] == mask_i.shape:
                mask_i = cp.logical_and(mask_i, gm_full == 0)
            mask_cp = cp.logical_or(mask_cp, mask_i)
        wall_mask = cp.where(mask_cp, cp.uint8(255), cp.uint8(0)).get()
    else:
        wall_mask = None

    if timing_on:
        _t_mask_end = time.perf_counter()
        _t_mask_ms = (_t_mask_end - _t_mask_start) * 1000.0
        _t_total_ms = (_t_mask_end - _t_total_start) * 1000.0
        print(
            f"[get_wall_planes timing] params: {_t_params_ms:.2f}ms | "
            f"convert: {_t_convert_ms:.2f}ms | "
            f"subsample: {_t_subsample_ms:.2f}ms | "
            f"pointcloud: {_t_pointcloud_ms:.2f}ms | "
            f"ransac: {_t_ransac_ms:.2f}ms | "
            f"refine: {_t_refine_ms:.2f}ms | "
            f"mask: {_t_mask_ms:.2f}ms | "
            f"TOTAL: {_t_total_ms:.2f}ms"
        )

    return {"planes": planes, "wall_mask": wall_mask}
