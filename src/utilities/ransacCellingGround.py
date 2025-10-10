import math
import numpy as np

try:
    import cupy as cp
    xp = cp  # backend: GPU
    GPU = True
except Exception:
    xp = np  # backend: CPU
    GPU = False


def _to_xp(a):
    return xp.asarray(a) if not isinstance(a, (xp.ndarray,)) else a


def plane_from_3pts(a, b, c, eps=1e-9):
    """
    a,b,c: (...,3)
    Return: n (...,3) unit normal, d (...,) so that n·x + d = 0
    """
    ab = b - a
    ac = c - a
    n = xp.cross(ab, ac)
    norm = xp.linalg.norm(n, axis=-1, keepdims=True) + eps
    n = n / norm
    d = -xp.sum(n * a, axis=-1)
    return n, d


def point_plane_dist(n, d, pts):
    """
    n: (...,3), d: (...,)
    pts: (N,3)
    Return: ( ... , N ) absolute distances
    """
    # Broadcast: (k,3)·(N,3) -> (k,N)
    return xp.abs(n @ pts.T + d[..., None])


def angle_between(u, v, eps=1e-9):
    u = u / (xp.linalg.norm(u, axis=-1, keepdims=True) + eps)
    v = v / (xp.linalg.norm(v, axis=-1, keepdims=True) + eps)
    cosang = xp.clip(xp.sum(u * v, axis=-1), -1.0, 1.0)
    return xp.arccos(cosang)  # rad


def ransac_plane_gpu(points,
                     dist_thresh=0.02,
                     max_iters=2000,
                     min_inliers=500,
                     up_axis=(0.0, -1.0, 0.0),
                     max_angle_deg=20.0,
                     seed=42):
    """
    RANSAC de un plano 'horizontal' (suelo/techo).
    - points: (N,3) (xp array o numpy; se convierte)
    - dist_thresh: tolerancia (m)
    - max_iters: iteraciones RANSAC
    - min_inliers: inliers mínimos para aceptar
    - up_axis: vector 'vertical' del mundo (p.ej. (0,-1,0) RealSense; (0,0,1) mundo Z-up)
    - max_angle_deg: |ángulo(n, ±up_axis)| <= umbral
    Return: dict con 'n', 'd', 'inliers_mask', 'inliers_idx', 'num_inliers'
    """
    rng = np.random.default_rng(seed)
    P = _to_xp(points).astype(xp.float32)
    N = P.shape[0]
    if N < 3:
        return None

    up = xp.asarray(up_axis, dtype=xp.float32)
    best = {'count': -1}

    for _ in range(max_iters):
        # Muestra 3 índices distintos (en CPU para robustez) y trae a backend
        i, j, k = rng.choice(N, size=3, replace=False)
        a, b, c = P[i], P[j], P[k]

        # Modelo
        n, d = plane_from_3pts(a, b, c)

        # Descarta degenerados
        if not xp.isfinite(d):
            continue
        if xp.linalg.norm(n) < 1e-6:
            continue

        # Orientación: cercano a ± up_axis
        ang = angle_between(n[None, :], xp.stack([up, -up], axis=0))  # returns shape (2,)
        ang_min = float(xp.min(ang).get() if GPU else xp.min(ang))
        if math.degrees(ang_min) > max_angle_deg:
            continue

        # Inliers
        dists = point_plane_dist(n[None, :], d[None, ...], P)[0]  # (N,)
        mask = dists <= dist_thresh
        count = int(mask.sum().get() if GPU else mask.sum())

        if count > best['count'] and count >= min_inliers:
            best = {
                'n': n,
                'd': d,
                'mask': mask,
                'count': count
            }

    if best['count'] < 0:
        return None

    inliers_idx = xp.flatnonzero(best['mask'])
    return {
        'n': xp.asarray(best['n']),
        'd': xp.asarray(best['d']),
        'inliers_mask': xp.asarray(best['mask']),
        'inliers_idx': xp.asarray(inliers_idx),
        'num_inliers': int(best['count'])
    }


def extract_floor_and_ceiling(points,
                              dist_thresh=0.02,
                              max_iters=3000,
                              min_inliers=800,
                              up_axis=(0.0, -1.0, 0.0),
                              max_angle_deg=20.0,
                              seed=42):
    """
    1) Encuentra primer plano horizontal (suelo o techo).
    2) Elimina sus inliers y vuelve a buscar el segundo.
    3) Clasifica como 'floor' (menor proyección sobre +up) y 'ceiling' (mayor).
    """
    P = _to_xp(points).astype(xp.float32)
    up = xp.asarray(up_axis, dtype=xp.float32)

    res1 = ransac_plane_gpu(P, dist_thresh, max_iters, min_inliers, up_axis, max_angle_deg, seed)
    if res1 is None:
        return None, None

    # Quitar inliers del primer plano
    mask1 = res1['inliers_mask']
    keep = ~mask1
    P2 = P[keep]
    res2 = ransac_plane_gpu(P2, dist_thresh, max_iters, min_inliers, up_axis, max_angle_deg, seed + 1)
    if res2 is None:
        # Sólo un plano encontrado: intenta clasificarlo como 'floor' y deja 'ceiling' en None
        # Clasificación por “altura” media de inliers
        pts1 = P[res1['inliers_idx']]
        # proyección escalar sobre +up
        h1 = xp.mean(pts1 @ up)
        floor, ceiling = (res1, None) if (h1.get() if GPU else h1) < 0 else (None, res1)
        return floor, ceiling

    # Clasificar por altura (proyección sobre +up)
    pts1 = P[res1['inliers_idx']]
    pts2 = P2[res2['inliers_idx']]
    h1 = xp.mean(pts1 @ up)
    h2 = xp.mean(pts2 @ up)

    if (h1.get() if GPU else h1) < (h2.get() if GPU else h2):
        floor, ceiling = res1, res2
    else:
        floor, ceiling = res2, res1

    return floor, ceiling


# =======================
# Ejemplo de uso mínimo
# =======================
if __name__ == "__main__":
    # Simulación rápida: plano z=0 (suelo) y z=2.5 (techo) + ruido
    np.random.seed(0)
    N = 50000
    xy = np.random.uniform(-3, 3, size=(N//2, 2))
    z_floor = np.zeros((N//2, 1)) + np.random.normal(0, 0.005, size=(N//2, 1))
    floor_pts = np.hstack([xy, z_floor])

    xy2 = np.random.uniform(-3, 3, size=(N//2, 2))
    z_ceil = np.full((N//2, 1), 2.5) + np.random.normal(0, 0.005, size=(N//2, 1))
    ceil_pts = np.hstack([xy2, z_ceil])

    pts = np.vstack([floor_pts, ceil_pts]).astype(np.float32)

    # Aquí asumimos mundo Z-up -> up_axis=(0,0,1)
    floor, ceiling = extract_floor_and_ceiling(
        pts,
        dist_thresh=0.02,
        max_iters=1500,
        min_inliers=1500,
        up_axis=(0.0, 0.0, 1.0),
        max_angle_deg=15.0
    )

    backend = "GPU (CuPy)" if GPU else "CPU (NumPy)"
    print(f"Backend: {backend}")
    if floor:
        print("Floor inliers:", floor['num_inliers'])
        n = floor['n'].get() if GPU else floor['n']
        d = floor['d'].get() if GPU else floor['d']
        print("Floor plane: n =", n, " d =", float(d))
    if ceiling:
        print("Ceiling inliers:", ceiling['num_inliers'])
        n = ceiling['n'].get() if GPU else ceiling['n']
        d = ceiling['d'].get() if GPU else ceiling['d']
        print("Ceiling plane: n =", n, " d =", float(d))
