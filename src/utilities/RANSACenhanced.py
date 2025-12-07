"""
GPU-accelerated RANSAC for ground-plane detection
following the algorithm in:

"Fast and Accurate Ground Plane Detection for the Visually
Impaired from 3D Organized Point Clouds" (Zeineldin & El-Fishawy, 2016)

Pipeline:
1) Depth -> 3D point cloud
2) Preprocessing: passthrough filter + voxelization
3) Surface normal estimation (radius search r = 4 cm)
4) Enhanced RANSAC with normals (Algorithm 2)
5) Ground mask by distance-to-plane threshold
"""

import math
import os
import pathlib
import sys
import time
from typing import Optional, Dict, Any

import numpy as np
import cupy as cp
import cv2

try:
    from . import helpers, viewCamera  # type: ignore
except ImportError:
    # Fallback when executed as a script: add project src/ to sys.path
    _ROOT = pathlib.Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.append(str(_ROOT))
    import utilities.helpers as helpers  # type: ignore
    import utilities.viewCamera as viewCamera  # type: ignore

_last_ransac_ms: Optional[float] = None

# ---------------------------------------------------------------------
# Utilidades básicas
# ---------------------------------------------------------------------

def _to_cp(a):
    """Ensure array lives on GPU (CuPy)."""
    return cp.asarray(a)


def _fit_plane_least_squares(points_cp: cp.ndarray):
    """
    Ajusta un plano n·x + d = 0 por mínimos cuadrados
    a un conjunto de puntos 3D (N,3) en GPU.
    Devuelve (n_vec, d).
    """
    n_pts = int(points_cp.shape[0])
    if n_pts < 3:
        return None, None

    center = cp.mean(points_cp, axis=0)
    X = points_cp - center
    denom = max(n_pts - 1, 1)
    cov = (X.T @ X) / float(denom)  # (3,3)

    vals, vecs = cp.linalg.eigh(cov)
    min_idx = int(cp.argmin(vals))
    n_vec = vecs[:, min_idx]
    d_val = -cp.dot(n_vec, center)
    return n_vec, d_val


# ---------------------------------------------------------------------
# Voxelización (preprocesamiento del paper)
# ---------------------------------------------------------------------

def _voxel_downsample_points(points_cp: cp.ndarray,
                             voxel_size: float) -> cp.ndarray:
    """
    Voxel-grid filter:
    - Divide el espacio en cubos de lado voxel_size.
    - En cada voxel, conserva el centroide de los puntos que caen allí.

    Esto corresponde a la “voxelization” del preprocesamiento.
    """
    if voxel_size <= 0.0 or points_cp.size == 0:
        return points_cp

    pts = points_cp.astype(cp.float32)
    mins = cp.min(pts, axis=0)
    scaled = (pts - mins[None, :]) / float(voxel_size)
    idx = cp.floor(scaled).astype(cp.int32)  # (N,3)

    K = cp.int64(1_000_000)
    key = (idx[:, 0].astype(cp.int64)
           + idx[:, 1].astype(cp.int64) * K
           + idx[:, 2].astype(cp.int64) * K * K)

    unique_keys, inv = cp.unique(key, return_inverse=True)
    n_vox = unique_keys.size

    counts = cp.bincount(inv, minlength=n_vox).astype(cp.float32)
    sum_x = cp.bincount(inv, weights=pts[:, 0], minlength=n_vox)
    sum_y = cp.bincount(inv, weights=pts[:, 1], minlength=n_vox)
    sum_z = cp.bincount(inv, weights=pts[:, 2], minlength=n_vox)

    centroids = cp.stack(
        [sum_x / (counts + 1e-9),
         sum_y / (counts + 1e-9),
         sum_z / (counts + 1e-9)],
        axis=1
    )
    return centroids.astype(cp.float32)


# ---------------------------------------------------------------------
# Estimación de normales por vecindario (r = 4 cm)
# ---------------------------------------------------------------------

def _estimate_normals_radius(points_cp: cp.ndarray,
                             radius: float,
                             viewpoint=(0.0, 0.0, 0.0),
                             block_size: int = 64) -> cp.ndarray:
    """
    Estima normales locales usando vecindario en radio fijo (r),
    como se describe en el paper (radius search r = 4 cm).

    Para cada punto:
      - Toma vecinos dentro de 'radius'
      - Ajusta plano por mínimos cuadrados
      - Normal = autovector asociado al menor autovalor de la covarianza
      - Orienta la normal hacia el 'viewpoint' (típicamente el sensor)

    points_cp: (N,3) en GPU.
    radius: radio en metros (por defecto 0.04 m).
    """
    pts = points_cp.astype(cp.float32)
    N = int(pts.shape[0])
    if N == 0:
        return cp.zeros_like(pts)

    r2 = float(radius * radius)
    normals = cp.zeros_like(pts)
    vp = cp.asarray(viewpoint, dtype=cp.float32)

    for start in range(0, N, block_size):
        end = min(N, start + block_size)
        Pi = pts[start:end]  # (b,3)
        b = int(Pi.shape[0])

        # Distancias de cada punto del bloque a todos los puntos
        diff = pts[cp.newaxis, :, :] - Pi[:, cp.newaxis, :]  # (b,N,3)
        dist2 = cp.sum(diff ** 2, axis=2)                    # (b,N)
        neighbor_mask = dist2 <= r2                          # (b,N)

        for i in range(b):
            mask_i = neighbor_mask[i]
            count_i = int(mask_i.sum().get())
            if count_i < 3:
                normals[start + i] = cp.array([0.0, 0.0, 0.0], dtype=cp.float32)
                continue

            neigh = pts[mask_i]  # (M_i,3)
            n_vec, _ = _fit_plane_least_squares(neigh)
            if n_vec is None:
                normals[start + i] = cp.array([0.0, 0.0, 0.0], dtype=cp.float32)
                continue

            # Orientar normal hacia el punto de vista (sensor)
            p_i = Pi[i]
            to_vp = vp - p_i
            if cp.dot(n_vec, to_vp) < 0:
                n_vec = -n_vec

            # Normalizar
            n_vec = n_vec / (cp.linalg.norm(n_vec) + 1e-9)
            normals[start + i] = n_vec

    return normals.astype(cp.float32)


# ---------------------------------------------------------------------
# RANSAC mejorado con normales (Algoritmo 2 del paper)
# ---------------------------------------------------------------------

def ransac_plane_gpu(points,
                     point_normals,
                     dist_thresh: float = 0.02,
                     normal_angle_deg: float = 20.0,
                     max_iters: int = 2000,
                     min_inliers: int = 500,
                     seed: int = 42,
                     batch_size: int = 128,
                     up_axis=(0.0, -1.0, 0.0),
                     up_angle_deg: float = 60.0,
                     orientation: str = "ground"):
    """
    GPU implementation of Algorithm 2 (RANSAC with normals).

    Inlier if:
      distance_error < dist_thresh
      normal_error   < normal_angle_deg (degrees)
      up/orientation respected within up_angle_deg

    points: (N,3) NumPy or CuPy.
    point_normals: (N,3) NumPy or CuPy (precomputed local normals).
    """
    P = _to_cp(points).astype(cp.float32)
    Nrm = _to_cp(point_normals).astype(cp.float32)
    N = int(P.shape[0])

    if N < 3:
        return None

    if Nrm.shape[0] != N:
        raise ValueError("point_normals debe tener la misma longitud que points")

    cos_norm_thresh = math.cos(math.radians(float(normal_angle_deg)))
    cos_up_thresh = math.cos(math.radians(float(up_angle_deg)))
    rad_to_deg = 180.0 / math.pi
    up = cp.asarray(up_axis, dtype=cp.float32)
    up = up / (cp.linalg.norm(up) + 1e-9)

    rng = cp.random.RandomState(seed)

    best_plane_n = None
    best_plane_d = None
    best_distance_error = float("inf")
    best_normal_error = float("inf")
    best_inliers_mask = None
    best_score = None

    remaining = int(max_iters)

    while remaining > 0:
        K = min(batch_size, remaining)
        remaining -= K

        # 1) random_inliers := n random selected points (usamos n=3)
        idxs = rng.randint(0, N, size=(K, 3), dtype=cp.int32)
        a = P[idxs[:, 0]]
        b = P[idxs[:, 1]]
        c = P[idxs[:, 2]]

        # Plano inicial por producto cruzado de 3 puntos
        ab = b - a
        ac = c - a
        n = cp.cross(ab, ac)        # (K,3)
        norm = cp.linalg.norm(n, axis=1)  # (K,)
        valid = norm > 1e-8
        n_unit = cp.where(valid[:, None], n / (norm[:, None] + 1e-12), 0.0)
        d = -cp.sum(n_unit * a, axis=1)   # (K,)

        # OrientaciÃ³n respecto a up_axis
        dot_up = n_unit @ up  # (K,)
        if orientation == "ground":
            valid = cp.logical_and(valid, dot_up <= -cos_up_thresh)
        elif orientation == "ceiling":
            valid = cp.logical_and(valid, dot_up >= cos_up_thresh)
        else:  # any
            valid = cp.logical_and(valid, cp.abs(dot_up) >= cos_up_thresh)

        valid_idx = cp.nonzero(valid)[0]
        if valid_idx.size == 0:
            continue  # muestras degeneradas (colineales o duplicadas)

        # 2) surface_normal := media de las normales locales de random_inliers
        surf_normals = cp.zeros_like(n_unit)
        for j in range(K):
            idx_j = idxs[j]              # indices de los 3 random_inliers
            n_loc = Nrm[idx_j]           # (3,3)
            nn = cp.mean(n_loc, axis=0)
            nn = nn / (cp.linalg.norm(nn) + 1e-9)
            surf_normals[j] = nn

        # 3) Evaluar todos los puntos (distancia + angulo de normal)
        dists = cp.zeros((K, N), dtype=cp.float32)
        cos_angles = cp.zeros((K, N), dtype=cp.float32)

        dists_valid = cp.abs(n_unit[valid_idx] @ P.T + d[valid_idx][:, None])   # (K_valid,N)
        cos_angles_valid = cp.abs(surf_normals[valid_idx] @ Nrm.T)              # (K_valid,N)

        dists[valid_idx] = dists_valid
        cos_angles[valid_idx] = cos_angles_valid

        inliers_mask = (dists <= dist_thresh) & (cos_angles >= cos_norm_thresh)  # (K,N)
        counts = cp.sum(inliers_mask, axis=1)       # (K,)

        # 4) Para cada modelo con suficientes inliers, refinamos el plano
        for j in range(K):
            count_j = int(counts[j].get())
            if count_j < min_inliers:
                continue  # no es un buen plano

            mask_j = inliers_mask[j]
            pts_j = P[mask_j]      # consensus_set
            n_ref, d_ref = _fit_plane_least_squares(pts_j)
            if n_ref is None:
                continue

            # Recalcular errores con el plano ajustado a consensus_set
            dists_j = cp.abs(pts_j @ n_ref + d_ref)  # distancias (|n*x + d|)
            # Error de distancia: media de las distancias
            this_distance_error = float(cp.mean(dists_j).get())

            # Error de normales: media del angulo entre normal del plano y normales locales
            normals_j = Nrm[mask_j]
            cos_j = cp.clip(cp.abs(normals_j @ n_ref), 0.0, 1.0)
            angles_j = cp.arccos(cos_j)  # en radianes
            this_normal_error = float(cp.mean(angles_j).get()) * rad_to_deg  # devolver en grados

            # Seleccion: mas inliers, luego menor error medio
            score = (count_j, -this_distance_error, -this_normal_error)
            if (best_score is None) or (score > best_score):
                best_score = score
                best_distance_error = this_distance_error
                best_normal_error = this_normal_error
                best_plane_n = n_ref
                best_plane_d = d_ref
                best_inliers_mask = mask_j

    if best_plane_n is None:
        return None

    result = {
        "n": best_plane_n,
        "d": best_plane_d,
        "inliers_mask": best_inliers_mask,
        "num_inliers": int(best_inliers_mask.sum().get()),
        "distance_error": best_distance_error,
        "normal_error": best_normal_error,  # en grados
    }
    return result

# ---------------------------------------------------------------------
# get_ground: pipeline completo del paper (sin ROI ni postprocesos extra)
# ---------------------------------------------------------------------

def get_ground(
        mapaProfundidad: np.ndarray,
        rays_cp: cp.ndarray,
        H: int,
        W: int,
        groundParams: Dict[str, Any]
        ) -> Optional[np.ndarray]:
    """
    Detecta el plano de suelo siguiendo el algoritmo del paper
    y devuelve una máscara binaria (H x W) en uint8 (0/255).

    Pasos:
      1) Depth -> 3D point cloud
      2) Passthrough filter (max_depth_m)
      3) Voxelization (voxel_size)
      4) Estimación de normales (radio normal_radius_m)
      5) RANSAC mejorado con normales (Algorithm 2)
      6) Máscara de suelo: puntos con |n·x + d| <= dist_thresh
    """
    global _last_ransac_ms

    if mapaProfundidad is None or rays_cp is None or H is None or W is None:
        _last_ransac_ms = None
        return None

    groundParams = groundParams or {}

    # Parámetros (todos directos del algoritmo del paper)
    dist_thresh = float(groundParams.get("dist_thresh", 0.02) or 0.02)           # d = 2 cm
    normal_angle_deg = float(groundParams.get("normal_angle_deg", 20.0) or 20.0) # m (ángulo máximo)
    max_iters = int(groundParams.get("max_iters", 2000) or 2000)
    min_inliers = int(groundParams.get("min_inliers", 500) or 500)
    seed = int(groundParams.get("seed", 42) or 42)
    batch_size = int(groundParams.get("batch_size", 64) or 64)
    up_axis = groundParams.get("up_axis", (0.0, -1.0, 0.0))
    up_angle_deg = float(groundParams.get("up_angle_deg", 60.0) or 60.0)
    orientation = groundParams.get("orientation", "ground") or "ground"

    max_depth_m = float(groundParams.get("max_depth_m", 3.0) or 3.0)             # passthrough
    voxel_size = float(groundParams.get("voxel_size", 0.03) or 0.03)             # tamaño de voxel
    normal_radius_m = float(groundParams.get("normal_radius_m", 0.04) or 0.04)   # r = 4 cm

    t0 = time.perf_counter()
    # Conversión a CuPy
    try:
        depth_cp = cp.asarray(mapaProfundidad, dtype=cp.float32)
    except Exception:
        return None
    try:
        rays_cp = cp.asarray(rays_cp, dtype=cp.float32)
    except Exception:
        return None

    if rays_cp.shape[:2] != depth_cp.shape[:2]:
        return None

    # 1) Passthrough filter: limitar por rango de profundidad
    valid_depth = (depth_cp > 0)
    if max_depth_m > 0.0:
        valid_depth &= (depth_cp <= max_depth_m)

    # Puntos 3D completos (H,W,3)
    points_full = rays_cp * depth_cp[..., None]  # (H,W,3)

    # Aplanar puntos válidos para preprocesamiento
    valid_flat = valid_depth.reshape(-1)
    if int(valid_flat.sum().get()) < 3:
        _last_ransac_ms = None
        return None

    P_raw = points_full.reshape(-1, 3)[valid_flat]  # (N_raw,3)
    t1 = time.perf_counter()  # fin paso 1: Depth->3D + passthrough

    # 2) Voxelization (preprocessing)
    P_vox = _voxel_downsample_points(P_raw, voxel_size)
    if int(P_vox.shape[0]) < 3:
        _last_ransac_ms = None
        return None
    t2 = time.perf_counter()  # fin paso 2: voxelización

    # 3) Estimación de normales (radius search r = 4 cm)
    N_vox = _estimate_normals_radius(P_vox, normal_radius_m)
    t3 = time.perf_counter()  # fin paso 3: normales

    # 4) RANSAC mejorado con normales (Algorithm 2)
    t0 = time.perf_counter()
    res = ransac_plane_gpu(
        P_vox,
        N_vox,
        dist_thresh=dist_thresh,
        normal_angle_deg=normal_angle_deg,
        max_iters=max_iters,
        min_inliers=min_inliers,
        seed=seed,
        batch_size=batch_size,
        up_axis=up_axis,
        up_angle_deg=up_angle_deg,
        orientation=orientation,
    )
    _last_ransac_ms = (time.perf_counter() - t0) * 1000.0
    t4 = time.perf_counter()  # fin paso 4: RANSAC

    if res is None:
        return None

    n_best = res["n"]
    d_best = res["d"]

    # 5) Máscara de suelo en resolución completa
    pts_flat_full = points_full.reshape(-1, 3)
    dists_full = cp.abs(pts_flat_full @ n_best + d_best)  # (H*W,)
    mask_flat = (dists_full <= dist_thresh) & valid_flat  # solo donde hay profundidad válida
    mask_cp = mask_flat.reshape(H, W)

    # Convertir a uint8 (0/255) en CPU
    ground_mask = cp.where(mask_cp, cp.uint8(255), cp.uint8(0))
    t5 = time.perf_counter()  # fin paso 5: máscara

    # Tiempos por paso (ms)
    print(
        "[timing] step1 depth->3D+passthrough: "
        f"{(t1 - t0)*1000:.2f} ms  "
        "step2 voxel: "
        f"{(t2 - t1)*1000:.2f} ms  "
        "step3 normals: "
        f"{(t3 - t2)*1000:.2f} ms  "
        "step4 ransac: "
        f"{(t4 - t3)*1000:.2f} ms  "
        "step5 mask: "
        f"{(t5 - t4)*1000:.2f} ms"
    )
    return ground_mask.get()


def get_last_ransac_ms(copy: bool = True) -> Optional[float]:
    """
    Duración (ms) de la última ejecución de RANSAC (o None si no hay).
    """
    if _last_ransac_ms is None:
        return None
    return float(_last_ransac_ms)



def _demo_run(frame_idx: int = 0, ground_params=None) -> None:
    """
    Demo interactivo: recorre el dataset, aplica get_ground y muestra overlay.
    Presiona ESC o q para salir.
    """
    idx = int(frame_idx)
    ground_params = ground_params or {}
    while True:
        rgb, depth = helpers.load_dataset_frame(idx)
        if rgb is None or depth is None:
            print("[demo] No se pudo cargar una imagen del dataset.")
            break

        H, W = depth.shape[:2]
        rays_np = viewCamera.compute_normalized_rays(H, W)
        rays_cp = cp.asarray(rays_np, dtype=cp.float32)

        t0 = time.perf_counter()
        mask = get_ground(depth, rays_cp, H, W, ground_params)
        total_ms = (time.perf_counter() - t0) * 1000.0
        ransac_ms = get_last_ransac_ms()

        if mask is None:
            print(f"[demo] get_ground devolvio None. total_ms={total_ms:.2f} ms")
            break

        overlay = helpers.apply_mask_to_rgb(rgb, mask)
        if overlay is not None:
            cv2.imshow("Ground (RANSACenhanced)", overlay)
        else:
            cv2.imshow("Ground (RANSACenhanced)", rgb)

        if ransac_ms is None:
            print(f"[demo] idx={idx}  total={total_ms:.2f} ms (RANSAC no reporto)")
        else:
            print(f"[demo] idx={idx}  total={total_ms:.2f} ms   RANSAC={ransac_ms:.2f} ms")

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        idx += 1

    cv2.destroyAllWindows()


if __name__ == '__main__':
    _demo_run(1, {})
