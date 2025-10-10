#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import numpy as np

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    XP_TORCH = True
except Exception:
    torch = None
    DEVICE = "cpu"
    XP_TORCH = False

def _to_t(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x


def plane_from_3pts_t(a, b, c, eps=1e-9):
    """
    a,b,c: (...,3) tensores torch (device=DEVICE)
    return: n (...,3) normal unitaria, d (...,) escalar del plano n·x + d = 0
    """
    ab = b - a
    ac = c - a
    n = torch.cross(ab, ac, dim=-1)
    norm = torch.linalg.norm(n, dim=-1, keepdim=True) + eps
    n = n / norm
    d = -(n * a).sum(dim=-1)
    return n, d


def point_plane_dist_t(n, d, pts):
    """
    n: (k,3), d: (k,), pts: (N,3)  -> dist (k,N)
    """
    return (n @ pts.T + d[:, None]).abs()


def angle_between_t(u, v, eps=1e-9):
    """
    u: (...,3), v: (...,3)  -> ángulo en rad (broadcast sobre la primera dim)
    """
    u = u / (torch.linalg.norm(u, dim=-1, keepdim=True) + eps)
    v = v / (torch.linalg.norm(v, dim=-1, keepdim=True) + eps)
    cosang = torch.clamp((u * v).sum(dim=-1), -1.0, 1.0)
    return torch.arccos(cosang)


def ransac_plane_torch(points,
                       dist_thresh=0.02,
                       max_iters=2000,
                       min_inliers=500,
                       up_axis=(0.0, -1.0, 0.0),
                       max_angle_deg=20.0,
                       seed=42):
    """
    RANSAC para un plano horizontal (suelo/techo) usando Torch (GPU si disponible).
    """
    if not XP_TORCH:
        raise RuntimeError("PyTorch no está disponible en este entorno.")

    rng = np.random.default_rng(seed)
    P = _to_t(points).to(dtype=torch.float32, device=DEVICE)
    N = P.shape[0]
    if N < 3:
        return None

    up = torch.tensor(up_axis, dtype=torch.float32, device=DEVICE)
    best = {'count': -1}

    for _ in range(max_iters):
        # elegir 3 índices distintos (usamos NumPy por simplicidad)
        i, j, k = rng.choice(N, size=3, replace=False)
        a, b, c = P[i], P[j], P[k]

        # modelo
        n, d = plane_from_3pts_t(a, b, c)

        if not torch.isfinite(d).all():
            continue
        if torch.linalg.norm(n) < 1e-6:
            continue

        # orientación cercana a ±up
        ang = angle_between_t(n[None, :], torch.stack([up, -up], dim=0))  # (2,)
        ang_min = ang.min().item()
        if math.degrees(ang_min) > max_angle_deg:
            continue

        # inliers
        dists = point_plane_dist_t(n[None, :], d[None], P)[0]  # (N,)
        mask = dists <= dist_thresh
        count = int(mask.sum().item())

        if count > best['count'] and count >= min_inliers:
            best = {'n': n, 'd': d, 'mask': mask, 'count': count}

    if best['count'] < 0:
        return None

    inliers_idx = torch.nonzero(best['mask'], as_tuple=False).squeeze(1)
    return {
        'n': best['n'],
        'd': best['d'],
        'inliers_mask': best['mask'],
        'inliers_idx': inliers_idx,
        'num_inliers': best['count'],
    }


def extract_floor_and_ceiling_torch(points,
                                    dist_thresh=0.02,
                                    max_iters=3000,
                                    min_inliers=800,
                                    up_axis=(0.0, -1.0, 0.0),
                                    max_angle_deg=20.0,
                                    seed=42):
    """
    Encuentra dos planos horizontales (suelo y techo) y los clasifica por “altura”.
    """
    if not XP_TORCH:
        raise RuntimeError("PyTorch no está disponible en este entorno.")

    P = _to_t(points).to(dtype=torch.float32, device=DEVICE)
    up = torch.tensor(up_axis, dtype=torch.float32, device=DEVICE)

    res1 = ransac_plane_torch(P, dist_thresh, max_iters, min_inliers, up_axis, max_angle_deg, seed)
    if res1 is None:
        return None, None

    # quitar inliers del primer plano
    keep = ~res1['inliers_mask']
    P2 = P[keep]
    res2 = ransac_plane_torch(P2, dist_thresh, max_iters, min_inliers, up_axis, max_angle_deg, seed + 1)
    if res2 is None:
        pts1 = P[res1['inliers_idx']]
        h1 = (pts1 @ up).mean().item()
        floor, ceiling = (res1, None) if h1 < 0 else (None, res1)
        return floor, ceiling

    # clasificar por proyección sobre +up
    pts1 = P[res1['inliers_idx']]
    pts2 = P2[res2['inliers_idx']]
    h1 = (pts1 @ up).mean().item()
    h2 = (pts2 @ up).mean().item()

    if h1 < h2:
        floor, ceiling = res1, res2
    else:
        floor, ceiling = res2, res1
    return floor, ceiling


# =============================
# Implementación con NumPy (CPU)
# =============================

def plane_from_3pts_np(a, b, c, eps=1e-9):
    ab = b - a
    ac = c - a
    n = np.cross(ab, ac)
    norm = np.linalg.norm(n) + eps
    n = n / norm
    d = -np.dot(n, a)
    return n, d


def angle_between_np(u, v, eps=1e-9):
    u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + eps)
    v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + eps)
    cosang = np.clip(np.sum(u * v, axis=-1), -1.0, 1.0)
    return np.arccos(cosang)


def ransac_plane_numpy(points,
                       dist_thresh=0.02,
                       max_iters=2000,
                       min_inliers=500,
                       up_axis=(0.0, -1.0, 0.0),
                       max_angle_deg=20.0,
                       seed=42):
    if points is None:
        return None
    P = np.asarray(points, dtype=np.float32)
    N = P.shape[0]
    if N < 3:
        return None

    rng = np.random.default_rng(seed)
    up = np.asarray(up_axis, dtype=np.float32)
    best = {'count': -1}

    for _ in range(max_iters):
        i, j, k = rng.choice(N, size=3, replace=False)
        a, b, c = P[i], P[j], P[k]

        n, d = plane_from_3pts_np(a, b, c)
        if not np.isfinite(d):
            continue
        if np.linalg.norm(n) < 1e-6:
            continue

        ang = angle_between_np(n[None, :], np.stack([up, -up], axis=0))
        ang_min = float(ang.min())
        if math.degrees(ang_min) > max_angle_deg:
            continue

        # Distancias a plano: |n·x + d|
        dists = np.abs(P @ n + d)
        mask = dists <= dist_thresh
        count = int(mask.sum())

        if count > best['count'] and count >= min_inliers:
            best = {'n': n, 'd': d, 'mask': mask, 'count': count}

    if best['count'] < 0:
        return None

    inliers_idx = np.nonzero(best['mask'])[0]
    return {
        'n': best['n'],
        'd': best['d'],
        'inliers_mask': best['mask'],
        'inliers_idx': inliers_idx,
        'num_inliers': best['count'],
    }


def extract_floor_and_ceiling_numpy(points,
                                    dist_thresh=0.02,
                                    max_iters=3000,
                                    min_inliers=800,
                                    up_axis=(0.0, -1.0, 0.0),
                                    max_angle_deg=20.0,
                                    seed=42):
    P = np.asarray(points, dtype=np.float32)
    up = np.asarray(up_axis, dtype=np.float32)

    res1 = ransac_plane_numpy(P, dist_thresh, max_iters, min_inliers, up_axis, max_angle_deg, seed)
    if res1 is None:
        return None, None

    keep = ~res1['inliers_mask']
    P2 = P[keep]
    res2 = ransac_plane_numpy(P2, dist_thresh, max_iters, min_inliers, up_axis, max_angle_deg, seed + 1)
    if res2 is None:
        pts1 = P[res1['inliers_idx']]
        h1 = float((pts1 @ up).mean())
        floor, ceiling = (res1, None) if h1 < 0 else (None, res1)
        return floor, ceiling

    pts1 = P[res1['inliers_idx']]
    pts2 = P2[res2['inliers_idx']]
    h1 = float((pts1 @ up).mean())
    h2 = float((pts2 @ up).mean())

    if h1 < h2:
        floor, ceiling = res1, res2
    else:
        floor, ceiling = res2, res1
    return floor, ceiling


# =============================
# Wrapper: usa Torch si existe
# =============================

def extract_floor_and_ceiling(points,
                              dist_thresh=0.02,
                              max_iters=3000,
                              min_inliers=800,
                              up_axis=(0.0, -1.0, 0.0),
                              max_angle_deg=20.0,
                              seed=42,
                              verbose=False):
    if XP_TORCH:
        return extract_floor_and_ceiling_torch(points, dist_thresh, max_iters, min_inliers, up_axis, max_angle_deg, seed)
    else:
        return extract_floor_and_ceiling_numpy(points, dist_thresh, max_iters, min_inliers, up_axis, max_angle_deg, seed)


def get_backend_info():
    """Retorna información del backend en uso.

    Returns
    -------
    dict: { 'framework': 'Torch'|'NumPy', 'device': 'GPU'|'CPU' }
    """
    if XP_TORCH:
        device = 'GPU' if (torch is not None and torch.cuda.is_available()) else 'CPU'
        return {'framework': 'Torch', 'device': device}
    else:
        return {'framework': 'NumPy', 'device': 'CPU'}


def get_backend_string():
    info = get_backend_info()
    return f"{info['framework']} ({info['device']})"


# =======================
# Ejemplo de uso mínimo
# =======================
if __name__ == "__main__":
    np.random.seed(0)
    N = 50000
    xy = np.random.uniform(-3, 3, size=(N//2, 2))
    z_floor = np.zeros((N//2, 1)) + np.random.normal(0, 0.005, size=(N//2, 1))
    floor_pts = np.hstack([xy, z_floor])

    xy2 = np.random.uniform(-3, 3, size=(N//2, 2))
    z_ceil = np.full((N//2, 1), 2.5) + np.random.normal(0, 0.005, size=(N//2, 1))
    ceil_pts = np.hstack([xy2, z_ceil])
    pts = np.vstack([floor_pts, ceil_pts]).astype(np.float32)

    floor, ceiling = extract_floor_and_ceiling(
        pts,
        dist_thresh=0.02,
        max_iters=1500,
        min_inliers=1500,
        up_axis=(0.0, 0.0, 1.0),
        max_angle_deg=15.0,
    )

    print("Backend:", get_backend_string())

    def _to_np_arr(x):
        if XP_TORCH and (torch is not None) and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    if floor is not None:
        n = _to_np_arr(floor['n'])
        d = float(_to_np_arr(floor['d']))
        print("Floor inliers:", floor['num_inliers'])
        print("Floor plane: n =", n, " d =", d)
    if ceiling is not None:
        n = _to_np_arr(ceiling['n'])
        d = float(_to_np_arr(ceiling['d']))
        print("Ceiling inliers:", ceiling['num_inliers'])
        print("Ceiling plane: n =", n, " d =", d)
