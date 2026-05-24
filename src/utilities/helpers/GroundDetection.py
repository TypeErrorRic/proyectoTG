"""
GPU-accelerated RANSAC and ground-plane masking utilities.

Provides CUDA-friendly plane fitting, ground mask creation, and postprocessing
to keep the segmentation pipeline on GPU.
"""
import math
import time
from typing import Optional, Dict, Any

import cupy as cp
import numpy as np

_last_ransac_ms: Optional[float] = None

# Variable para activar la impresión de tiempos de cada etapa en get_ground
DEBUG_TIMING = False

def _to_xp(a):
    """Ensure array lives on GPU (CuPy)."""
    return cp.asarray(a)

def ransac_plane_gpu(points,
                     dist_thresh=0.02,
                     max_iters=2000,
                     min_inliers=500,
                     up_axis=(0.0, -1.0, 0.0),
                     max_angle_deg=20.0,
                     seed=42,
                     batch_size=None,
                     point_chunk=None,
                     score_subset=None,
                     orientation: str = 'any',
                     early_stop_ratio: float = 0.92):
    """
    GPU-optimized RANSAC for a horizontal plane (floor/ceiling).

    Performance tweaks:
    - Evaluates many models per batch (vectorized) to reduce kernel launches.
    - Avoids host<->GPU sync on every iteration; only once per batch.
    - Uses cosine-based orientation check instead of acos to save compute.

    Parameters:
    - points: (N,3) array (NumPy or CuPy; converted internally)
    - dist_thresh: inlier tolerance (m)
    - max_iters: total RANSAC iterations (approx. models evaluated)
    - min_inliers: minimum inliers to accept
    - up_axis: world "up" vector (e.g., (0,-1,0) RealSense; (0,0,1) Z-up)
    - max_angle_deg: |angle(n, up_axis)| <= threshold (implemented with cosine)
    - seed: RNG seed
    - batch_size: batch size for models (auto)
    - point_chunk: block size of points for inlier counting (auto)
    - score_subset: points to score models per batch (best validated with ALL points). Reduces K*N on Jetson.
    - orientation: "any" (floor or ceiling), "ground" (normal opposite to up),
      "ceiling" (normal aligned with up), "vertical" (normal perpendicular to up)

    Return: dict with "n", "d", "inliers_mask", "inliers_idx", "num_inliers"
    """
    P = _to_xp(points).astype(cp.float32)
    N = int(P.shape[0])
    if N < 3:
        return None

    # Heuristic based on GPU size
    small_gpu = False
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        mp = int(props.get('multiProcessorCount', 0))
        mem = int(props.get('totalGlobalMem', 0))
        # Jetson Nano ~ 1-2 SM y < 4GB
        small_gpu = (mp <= 4) or (mem and mem < 4 * 1024**3)
    except Exception:
        small_gpu = True  # conservador

    if batch_size is None:
        batch_size = 128 if small_gpu else 1024

    if point_chunk is None:
        point_chunk = 8192 if small_gpu else 16384

    if score_subset is None:
        score_subset = min(4096 if small_gpu else 16384, N)
    else:
        score_subset = min(int(score_subset), N)

    # Normaliza up una vez
    up = cp.asarray(up_axis, dtype=cp.float32)
    up = up / (cp.linalg.norm(up) + 1e-9)
    cos_thresh = math.cos(math.radians(float(max_angle_deg)))

    # RNG en GPU
    rng_state = cp.random.RandomState(seed)
    rand_fn = lambda shape: rng_state.randint(0, N, size=shape, dtype=cp.int32)

    best_count = -1
    best_n = None
    best_d = None

    # Subconjunto fijo para puntuar modelos por lote (GPU)
    try:
        samp_idx = rng_state.choice(N, size=score_subset, replace=False).astype(cp.int32)
    except Exception:
        # Fallback if choice fails: use randint and trim uniques
        tmp = rng_state.randint(0, N, size=score_subset * 2, dtype=cp.int32)
        samp_idx = cp.unique(tmp)[:score_subset]
        if samp_idx.size < score_subset:
            # Fill remaining slots if unique count is short
            pad = score_subset - int(samp_idx.size)
            samp_idx = cp.concatenate([samp_idx, tmp[:pad]])
    P_samp = P[samp_idx]
    # Chunk inlier counting to avoid materializing full KxS on small GPUs
    score_chunk = min(int(P_samp.shape[0]), 4096 if small_gpu else 8192)

    remaining = int(max_iters)
    start_time = time.perf_counter()
    processed_batches = 0
    while remaining > 0:
        K = int(min(batch_size, remaining))
        remaining -= K

        # 1) Sample indices (with replacement; degenerate filtered by norm)
        idxs = rand_fn((K, 3))
        a = P[idxs[:, 0]]
        b = P[idxs[:, 1]]
        c = P[idxs[:, 2]]

        # 2) Modelo por lote
        ab = b - a
        ac = c - a
        n = cp.cross(ab, ac)  # (K,3)
        norm = cp.linalg.norm(n, axis=1)  # (K,)
        valid = norm > 1e-8
        # Avoid division by zero
        n_unit = cp.where(valid[:, None], n / (norm[:, None] + 1e-12), 0)
        d = -cp.sum(n_unit * a, axis=1)

        # 3) Orientation filter via cosine with optional preference
        dot_up = n_unit @ up  # (K,)
        if orientation == 'ground':
            # normal opuesta a up (~-1)
            cond = dot_up <= -cos_thresh
        elif orientation == 'ceiling':
            # normal alineada con up (~+1)
            cond = dot_up >= cos_thresh
        elif orientation == 'vertical':
            # normal perpendicular to up (~0)
            sin_thresh = math.sin(math.radians(float(max_angle_deg)))
            cond = cp.abs(dot_up) <= sin_thresh
        else:  # 'any'
            cond = cp.abs(dot_up) >= cos_thresh
        valid = cp.logical_and(valid, cond)

        # 4) Conteo de inliers sobre SUBMUESTRA para elegir mejor modelo del lote
        #    Troceado para reducir memoria en GPU pequeñas
        counts = cp.zeros((K,), dtype=cp.int32)
        for start_s in range(0, int(P_samp.shape[0]), score_chunk):
            end_s = min(int(P_samp.shape[0]), start_s + score_chunk)
            Ps_block = P_samp[start_s:end_s]
            dists_s = cp.abs(n_unit @ Ps_block.T + d[:, None])  # (K,chunk)
            counts += cp.sum(dists_s <= dist_thresh, axis=1, dtype=cp.int32)

        # Invalidate non-valid models
        counts = cp.where(valid, counts, -cp.ones_like(counts))

        # 5) Mejor del lote
        batch_best_idx = int(cp.argmax(counts).get())
        batch_best_count = int(counts[batch_best_idx].get())

        if batch_best_count > best_count and batch_best_count >= min_inliers:
            best_count = batch_best_count
            best_n = n_unit[batch_best_idx]
            best_d = d[batch_best_idx]

        processed_batches += 1
        # Early-stop por calidad del modelo (en la submuestra)
        if score_subset and batch_best_count >= int(early_stop_ratio * int(score_subset)):
            break

    if best_count < 0:
        return None

    # 6) Recompute inlier mask of best model over ALL points (one pass)
    mask = cp.zeros((N,), dtype=bool)
    if N <= point_chunk:
        dists = cp.abs(best_n[None, :] @ P.T + best_d)
        mask = (dists[0] <= dist_thresh)
    else:
        # por bloques
        out = []
        for start in range(0, N, point_chunk):
            end = min(N, start + point_chunk)
            Pc = P[start:end]
            dists = cp.abs(best_n[None, :] @ Pc.T + best_d)[0]
            out.append(dists <= dist_thresh)
        mask = cp.concatenate(out, axis=0)
    inliers_idx = cp.flatnonzero(mask)

    final_count = int(mask.sum().get())

    return {
        'n': cp.asarray(best_n),
        'd': cp.asarray(best_d),
        'inliers_mask': cp.asarray(mask),
        'inliers_idx': cp.asarray(inliers_idx),
        'num_inliers': final_count,
    }

# Global variables to store the last detected ground plane parameters
# These are used to keep the last valid plane and threshold between frames
last_n_cp = None  # Last normal vector of the detected ground plane (CuPy array)
last_d_cp = None  # Last 'd' parameter of the detected ground plane (CuPy scalar)
_debug_ransac_times = []  # Sliding window of recent RANSAC runtimes (ms)
_debug_ransac_counter = 0  # Number of RANSAC executions (for debug logging)


def _refine_plane(points_cp, up_vec, orientation: str):
    """
    Fit a plane (least-squares) to points_cp on GPU.

    Returns (n, d) or None if it fails.
    """
    if points_cp is None:
        return None
    n_pts = int(points_cp.shape[0])
    if n_pts < 3:
        return None
    try:
        center = cp.mean(points_cp, axis=0)
        X = points_cp - center
        denom = max(n_pts - 1, 1)
        cov = (X.T @ X) / denom
        vals, vecs = cp.linalg.eigh(cov)
        min_idx = int(cp.argmin(vals))
        n_vec = vecs[:, min_idx]
        # Orient the normal consistently with the requested orientation
        dot_up = float(cp.dot(n_vec, up_vec).get())
        if orientation == "ground" and dot_up > 0:
            n_vec = -n_vec
        elif orientation == "ceiling" and dot_up < 0:
            n_vec = -n_vec
        d_val = -cp.dot(n_vec, center)
        return n_vec, d_val
    except Exception:
        return None


def _refine_with_full_res(
    depth_cp,
    rays_cp,
    n_cp,
    d_cp,
    dist_thresh,
    dist_mult,
    max_points,
    orientation,
    up_axis,
):
    """
    Refit the plane using all full-resolution inliers (optionally subsampled).
    """
    try:
        depth_flat = depth_cp.reshape(-1)
        rays_flat = rays_cp.reshape(-1, 3)
        # Signed distances to current plane
        signed = depth_flat * (rays_flat @ n_cp) + d_cp
        inliers = (cp.abs(signed) <= dist_thresh * dist_mult) & (depth_flat > 0)
        total = int(inliers.sum().get())
        if total < 3:
            return None
        idx = cp.flatnonzero(inliers)
        if max_points and total > max_points:
            idx = cp.random.choice(idx, size=max_points, replace=False)
        pts = rays_flat[idx] * depth_flat[idx, None]
        up_vec = cp.asarray(up_axis, dtype=cp.float32)
        up_vec = up_vec / (cp.linalg.norm(up_vec) + 1e-9)
        return _refine_plane(pts, up_vec, orientation)
    except Exception:
        return None

def get_ground(
        mapaProfundidad: np.ndarray,
        rays_cp: cp.ndarray,
        H: int,
        W: int,
        groundParams: Dict[str, Any],
        depth_cp: Optional[cp.ndarray] = None,
        ) -> Optional[np.ndarray]:
    """
    Detects the ground plane using RANSAC and returns the RGB image
    with the ground mask overlaid in green.

    Args:
        rgb_image (np.ndarray): RGB image (height x width x 3)
        mapaProfundidad (np.ndarray): Depth map (height x width)
        depth_cp (cp.ndarray | None): Optional depth map already on GPU.
        rays_cp (cp.ndarray): Precomputed rays (height x width x 3, CuPy array)
        H (int): Image height
        W (int): Image width
        groundParams (Dict[str, Any]): Dictionary of RANSAC and segmentation parameters:
            - subsample_stride (int): Subsampling stride for RANSAC (default: 4)
            - min_inliers (int): Minimum inliers to accept a plane (default: 600)
            - dist_thresh (float): Distance threshold for inliers (default: 0.03)
            - max_iters (int): Maximum RANSAC iterations (default: 500)
            - up_axis (tuple): World up vector (default: (0.0, -1.0, 0.0))
            - max_angle_deg (float): Max angle for plane orientation (default: 45.0)
            - seed (int): Random seed (default: 42)
            - score_subset (int): Number of points for scoring models (default: 2048)
        - orientation (str): Plane orientation ('ground', 'ceiling', 'any')
        - early_stop_ratio (float): Early stop ratio for RANSAC (default: 0.92)
        - batch_size (int): Batch size for RANSAC models (default: 256)
        - debug_ransac (bool): If True, log timing information.
        - debug_ransac_every (int): Log average timing every N runs (default: 20).

    Returns:
        np.ndarray | None: BGR image with ground mask overlay, or None if no valid data.
    """
    global last_n_cp, last_d_cp, _debug_ransac_times, _debug_ransac_counter, _last_ransac_ms

    # Timing para debug
    if DEBUG_TIMING:
        _t_total_start = time.perf_counter()

    # Guard against missing inputs (e.g., first frames or sensor not ready)
    if (mapaProfundidad is None and depth_cp is None) or rays_cp is None or H is None or W is None:
        _last_ransac_ms = None
        return None

    # Note: this function returns a binary mask (H x W) for floor pixels.
    # The RGB overlay is applied later in helpers.apply_mask_to_rgb.
    # RGB se aplica posteriormente en helpers.apply_mask_to_rgb.
    groundParams = groundParams or {}

    if DEBUG_TIMING:
        _t_params_start = time.perf_counter()
    # Extract parameters from groundParams dictionary
    try:
        subsample_stride = max(1, int(groundParams.get("subsample_stride") or 1))
    except Exception:
        subsample_stride = 1
    min_inliers = int(groundParams.get("min_inliers", 400) or 400)
    dist_thresh = float(groundParams.get("dist_thresh", 0.03) or 0.03)
    max_iters = int(groundParams.get("max_iters", 500) or 500)
    up_axis = groundParams.get("up_axis", (0.0, -1.0, 0.0))
    max_angle_deg = float(groundParams.get("max_angle_deg", 60.0) or 60.0)
    seed = int(groundParams.get("seed", 42) or 42)
    score_subset = groundParams.get("score_subset", 4096)
    score_subset = int(score_subset) if score_subset is not None else 4096
    orientation = groundParams.get("orientation", "ground") or "ground"
    early_stop_ratio = float(groundParams.get("early_stop_ratio", 0.92) or 0.92)
    batch_size = groundParams.get("batch_size", 128)
    batch_size = int(batch_size) if batch_size is not None else None
    debug_ransac = bool(groundParams.get("debug_ransac", False))
    dre = groundParams.get("debug_ransac_every", 20)
    debug_ransac_every = max(1, int(20 if dre is None else dre))
    # Nuevos controles
    low_height_pct = float(groundParams.get("low_height_pct", 0.0) or 0.0)
    roi_bottom_fraction = float(groundParams.get("roi_bottom_fraction", 1.0) or 1.0)
    roi_expand_step = float(groundParams.get("roi_expand_step", 0.0) or 0.0)
    max_agg_points = int(groundParams.get("max_agg_points", 0) or 0)
    refine_full_res = bool(groundParams.get("refine_full_res", False))
    refine_max_points = int(groundParams.get("refine_max_points", 0) or 0)
    refine_dist_mult = float(groundParams.get("refine_dist_mult", 1.0) or 1.0)

    if DEBUG_TIMING:
        _t_params_end = time.perf_counter()
        _t_params_ms = (_t_params_end - _t_params_start) * 1000.0

    # Convert depth map to CuPy array for RANSAC
    if DEBUG_TIMING:
        _t_convert_start = time.perf_counter()
    if depth_cp is None:
        try:
            depth_cp = cp.asarray(mapaProfundidad, dtype=cp.float32)
        except Exception:
            return None
    else:
        try:
            depth_cp = cp.asarray(depth_cp, dtype=cp.float32)
        except Exception:
            return None
    try:
        rays_cp = cp.asarray(rays_cp, dtype=cp.float32)
    except Exception:
        return None
    if rays_cp is None or depth_cp is None:
        return None
    if rays_cp.shape[:2] != depth_cp.shape[:2]:
        return None

    if DEBUG_TIMING:
        _t_convert_end = time.perf_counter()
        _t_convert_ms = (_t_convert_end - _t_convert_start) * 1000.0

    # Subsample depth and rays for RANSAC efficiency
    if DEBUG_TIMING:
        _t_subsample_start = time.perf_counter()
    try:
        Dsub = depth_cp[::subsample_stride, ::subsample_stride]
        Rsub = rays_cp[::subsample_stride, ::subsample_stride]
    except Exception:
        return None
    # ROI adaptable: arranca en fraccion inferior, expande si faltan puntos
    sub_h = Dsub.shape[0]
    roi_bottom_fraction = min(max(roi_bottom_fraction, 0.05), 1.0)
    roi_expand_step = max(roi_expand_step, 0.0)
    Droi, Rroi = Dsub, Rsub
    roi_used = roi_bottom_fraction
    for _ in range(4):
        start = int(sub_h * max(0.0, 1.0 - roi_used))
        Dtmp = Dsub[start:, :]
        Rtmp = Rsub[start:, :]
        valid_tmp = Dtmp > 0
        if int(cp.sum(valid_tmp)) >= int(min_inliers):
            Droi, Rroi = Dtmp, Rtmp
            break
        roi_used = min(1.0, roi_used + roi_expand_step)
    Dsub, Rsub = Droi, Rroi
    valid = Dsub > 0

    if DEBUG_TIMING:
        _t_subsample_end = time.perf_counter()
        _t_subsample_ms = (_t_subsample_end - _t_subsample_start) * 1000.0
        _t_ransac_ms = 0.0
        _t_pointcloud_ms = 0.0
        _t_refine_ms = 0.0

    if int(cp.sum(valid)) >= 3:
        if DEBUG_TIMING:
            _t_pointcloud_start = time.perf_counter()
        # Prepare 3D point cloud for RANSAC
        Psub = (Rsub.reshape(-1, 3) * Dsub.reshape(-1, 1)).astype(cp.float32)
        Psub = Psub[valid.reshape(-1)]

        # Sesgar hacia los puntos más bajos (altura)
        if low_height_pct > 0.0 and Psub.size:
            up_vec = cp.asarray(up_axis, dtype=cp.float32)
            up_vec = up_vec / (cp.linalg.norm(up_vec) + 1e-9)
            try:
                heights = Psub @ up_vec
                cutoff = cp.percentile(heights, low_height_pct)
                keep_low = heights <= cutoff
                if int(keep_low.sum().get()) >= 3:
                    Psub = Psub[keep_low]
            except Exception:
                pass

        # Opcional: limitar el número de puntos usados por RANSAC
        if max_agg_points > 0 and int(Psub.shape[0]) > max_agg_points:
            try:
                idx = cp.random.choice(int(Psub.shape[0]), size=max_agg_points, replace=False)
                Psub = Psub[idx]
            except Exception:
                pass

        if DEBUG_TIMING:
            _t_pointcloud_end = time.perf_counter()
            _t_pointcloud_ms = (_t_pointcloud_end - _t_pointcloud_start) * 1000.0

        if int(Psub.shape[0]) >= int(min_inliers):
            # Run RANSAC plane fitting on the subsampled points
            t0 = time.perf_counter()
            res = ransac_plane_gpu(
                Psub,
                dist_thresh=dist_thresh,
                max_iters=max_iters,
                min_inliers=min_inliers,
                up_axis=up_axis,
                max_angle_deg=max_angle_deg,
                seed=seed,
                score_subset=score_subset,
                orientation=orientation,
                early_stop_ratio=early_stop_ratio,
                batch_size=batch_size,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            _last_ransac_ms = elapsed_ms
            if debug_ransac:
                _debug_ransac_counter += 1
                _debug_ransac_times.append(elapsed_ms)
                if len(_debug_ransac_times) > debug_ransac_every:
                    _debug_ransac_times.pop(0)
                if _debug_ransac_counter % debug_ransac_every == 0:
                    window = _debug_ransac_times[-debug_ransac_every:]
                    avg_ms = sum(window) / len(window)
                    print(f"[ransac] Promedio últimas {debug_ransac_every} ejecuciones: {avg_ms:.2f} ms")

            if DEBUG_TIMING:
                _t_ransac_ms = elapsed_ms

            if res is not None:
                # Store last valid plane parameters
                last_n_cp = res['n']
                last_d_cp = res['d']
                if refine_full_res:
                    if DEBUG_TIMING:
                        _t_refine_start = time.perf_counter()
                    refined = _refine_with_full_res(
                        depth_cp,
                        rays_cp,
                        last_n_cp,
                        last_d_cp,
                        dist_thresh,
                        refine_dist_mult,
                        refine_max_points,
                        orientation,
                        up_axis,
                    )
                    if refined is not None:
                        last_n_cp, last_d_cp = refined
                    if DEBUG_TIMING:
                        _t_refine_end = time.perf_counter()
                        _t_refine_ms = (_t_refine_end - _t_refine_start) * 1000.0
            else:
                last_n_cp = None
                last_d_cp = None
        else:
            _last_ransac_ms = None
    else:
        # Not enough valid data, skip RANSAC for this frame
        _last_ransac_ms = None
        pass

    # Build mask for the best current plane (if available)
    if DEBUG_TIMING:
        _t_mask_start = time.perf_counter()
    if last_n_cp is not None:
        dotnr = cp.tensordot(rays_cp, last_n_cp, axes=([2], [0]))
        dists = cp.abs(depth_cp * dotnr + last_d_cp)
        valid_depth = depth_cp > 0
        mask_cp = (dists <= dist_thresh) & valid_depth
    else:
        mask_cp = cp.zeros((H, W), dtype=cp.bool_)

    if DEBUG_TIMING:
        _t_mask_end = time.perf_counter()
        _t_mask_ms = (_t_mask_end - _t_mask_start) * 1000.0

    # Convertir máscara a uint8
    ground_mask_cp = cp.where(mask_cp, cp.uint8(255), cp.uint8(0))

    if DEBUG_TIMING:
        _t_total_ms = (_t_mask_end - _t_total_start) * 1000.0
        print(f"[get_ground timing] params: {_t_params_ms:.2f}ms | "
              f"convert: {_t_convert_ms:.2f}ms | "
              f"subsample: {_t_subsample_ms:.2f}ms | "
              f"pointcloud: {_t_pointcloud_ms:.2f}ms | "
              f"ransac: {_t_ransac_ms:.2f}ms | "
              f"refine: {_t_refine_ms:.2f}ms | "
              f"mask: {_t_mask_ms:.2f}ms | "
              f"TOTAL: {_t_total_ms:.2f}ms")

    return ground_mask_cp.get()


def get_last_ransac_ms(copy: bool = True) -> Optional[float]:
    """
    Return the duration (ms) of the last RANSAC plane fit, or None if unavailable.
    """
    if _last_ransac_ms is None:
        return None
    return float(_last_ransac_ms)


class CaminoTransitable:
    """Fachada simple para deteccion de suelo/camino transitable."""

    def __init__(self) -> None:
        self.debug_timing = DEBUG_TIMING
        self.last_ransac_ms = _last_ransac_ms
        self.last_n_cp = last_n_cp
        self.last_d_cp = last_d_cp
        self.debug_ransac_times = _debug_ransac_times
        self.debug_ransac_counter = _debug_ransac_counter

    def _sincronizar_estado(self) -> None:
        self.debug_timing = DEBUG_TIMING
        self.last_ransac_ms = _last_ransac_ms
        self.last_n_cp = last_n_cp
        self.last_d_cp = last_d_cp
        self.debug_ransac_times = _debug_ransac_times
        self.debug_ransac_counter = _debug_ransac_counter

    def detectar(self, *args: Any, **kwargs: Any) -> Any:
        resultado = get_ground(*args, **kwargs)
        self._sincronizar_estado()
        return resultado

    def obtener_tiempo_ransac_ms(self) -> Optional[float]:
        self._sincronizar_estado()
        return None if self.last_ransac_ms is None else float(self.last_ransac_ms)

    def obtener_estado_global(self) -> Dict[str, Any]:
        self._sincronizar_estado()
        return {
            "debug_timing": self.debug_timing,
            "last_ransac_ms": self.last_ransac_ms,
            "last_n_cp": self.last_n_cp,
            "last_d_cp": self.last_d_cp,
            "debug_ransac_times": self.debug_ransac_times,
            "debug_ransac_counter": self.debug_ransac_counter,
        }


camino_transitable = CaminoTransitable()
