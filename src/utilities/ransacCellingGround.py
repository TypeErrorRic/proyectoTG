import math
import numpy as np
import cv2
from viewCamera import extract_pointcloud

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
                     seed=42,
                     batch_size=None,
                     point_chunk=None,
                     score_subset=None):
    """
    RANSAC de un plano 'horizontal' (suelo/techo) optimizado para GPU.

    Cambios clave de rendimiento:
    - Evalúa muchos modelos por lote (vectorizado) para reducir lanzamientos de kernel.
    - Evita sincronizaciones Host<->GPU en cada iteración; sólo por lote.
    - Usa criterio de orientación sin arccos (umbral en coseno) para ahorrar cómputo.

    Parámetros:
    - points: (N,3) (xp array o numpy; se convierte)
    - dist_thresh: tolerancia (m)
    - max_iters: iteraciones RANSAC totales (aprox. modelos evaluados)
    - min_inliers: inliers mínimos para aceptar
    - up_axis: vector 'vertical' del mundo (p.ej. (0,-1,0) RealSense; (0,0,1) mundo Z-up)
    - max_angle_deg: |ángulo(n, up_axis)| <= umbral (se implementa con coseno)
    - seed: semilla RNG
    - batch_size: tamaño de lote de modelos (auto)
    - point_chunk: tamaño de bloque de puntos para contar inliers (auto)
    - score_subset: número de puntos para puntuar modelos por lote (luego se
      valida el mejor con TODOS los puntos). Reduce K*N en Jetson.

    Return: dict con 'n', 'd', 'inliers_mask', 'inliers_idx', 'num_inliers'
    """
    P = _to_xp(points).astype(xp.float32)
    N = int(P.shape[0])
    if N < 3:
        return None

    # Heurísticas por backend y tamaño de GPU
    small_gpu = False
    if GPU:
        try:
            props = cp.cuda.runtime.getDeviceProperties(0)
            mp = int(props.get('multiProcessorCount', 0))
            mem = int(props.get('totalGlobalMem', 0))
            # Jetson Nano ~ 1-2 SM y < 4GB
            small_gpu = (mp <= 4) or (mem and mem < 4 * 1024**3)
        except Exception:
            small_gpu = True  # conservador

    if batch_size is None:
        if GPU and not small_gpu:
            batch_size = 1024
        elif GPU and small_gpu:
            batch_size = 128
        else:
            batch_size = 64

    if point_chunk is None:
        if GPU and not small_gpu:
            point_chunk = 16384
        elif GPU and small_gpu:
            point_chunk = 8192
        else:
            point_chunk = 8192

    if score_subset is None:
        if GPU and not small_gpu:
            score_subset = min(16384, N)
        elif GPU and small_gpu:
            score_subset = min(4096, N)
        else:
            score_subset = min(8192, N)
    else:
        score_subset = min(int(score_subset), N)

    # Normaliza up una vez
    up = xp.asarray(up_axis, dtype=xp.float32)
    up = up / (xp.linalg.norm(up) + 1e-9)
    cos_thresh = math.cos(math.radians(float(max_angle_deg)))

    # RNG: permitir en GPU si disponible
    if GPU:
        rng_state = cp.random.RandomState(seed)
        rand_fn = lambda shape: rng_state.randint(0, N, size=shape, dtype=cp.int32)
    else:
        rng_state = np.random.default_rng(seed)
        rand_fn = lambda shape: rng_state.integers(0, N, size=shape, dtype=np.int32)

    best_count = -1
    best_n = None
    best_d = None

    # Subconjunto fijo para puntuar modelos por lote
    if GPU:
        try:
            samp_idx = rng_state.choice(N, size=score_subset, replace=False).astype(cp.int32)
        except Exception:
            # Fallback si choice falla: usa randint y recorta únicos simples
            tmp = rng_state.randint(0, N, size=score_subset * 2, dtype=cp.int32)
            samp_idx = cp.unique(tmp)[:score_subset]
            if samp_idx.size < score_subset:
                # rellena si faltan
                pad = score_subset - int(samp_idx.size)
                samp_idx = cp.concatenate([samp_idx, tmp[:pad]])
        P_samp = P[samp_idx]
    else:
        samp_idx = rng_state.choice(N, size=score_subset, replace=False).astype(np.int32)
        P_samp = P[samp_idx]

    remaining = int(max_iters)
    while remaining > 0:
        K = int(min(batch_size, remaining))
        remaining -= K

        # 1) Muestreo de índices (con reemplazo; degenerados se filtran por norma)
        idxs = rand_fn((K, 3))
        a = P[idxs[:, 0]]
        b = P[idxs[:, 1]]
        c = P[idxs[:, 2]]

        # 2) Modelo por lote
        ab = b - a
        ac = c - a
        n = xp.cross(ab, ac)  # (K,3)
        norm = xp.linalg.norm(n, axis=1)  # (K,)
        valid = norm > 1e-8
        # Evitar división por cero
        n_unit = xp.where(valid[:, None], n / (norm[:, None] + 1e-12), 0)
        d = -xp.sum(n_unit * a, axis=1)

        # 3) Filtro de orientación mediante coseno
        cosang = xp.abs(n_unit @ up)  # (K,)
        valid = xp.logical_and(valid, cosang >= cos_thresh)

        # 4) Conteo de inliers sobre SUBMUESTRA para elegir mejor modelo del lote
        #    Mucho más eficiente en Jetson que KxN directo.
        counts = xp.zeros((K,), dtype=xp.int32)
        dists_s = xp.abs(n_unit @ P_samp.T + d[:, None])  # (K,S)
        counts = xp.sum(dists_s <= dist_thresh, axis=1)

        # Invalida modelos no válidos
        counts = xp.where(valid, counts, -xp.ones_like(counts))

        # 5) Mejor del lote
        batch_best_idx = int((xp.argmax(counts)).get() if GPU else int(xp.argmax(counts)))
        batch_best_count = int((counts[batch_best_idx]).get() if GPU else int(counts[batch_best_idx]))

        if batch_best_count > best_count and batch_best_count >= min_inliers:
            best_count = batch_best_count
            best_n = n_unit[batch_best_idx]
            best_d = d[batch_best_idx]

    if best_count < 0:
        return None

    # 6) Recalcular máscara de inliers del mejor modelo sobre TODOS los puntos (una vez)
    mask = xp.zeros((N,), dtype=bool)
    if N <= point_chunk:
        dists = xp.abs(best_n[None, :] @ P.T + best_d)
        mask = (dists[0] <= dist_thresh)
    else:
        # por bloques
        out = []
        for start in range(0, N, point_chunk):
            end = min(N, start + point_chunk)
            Pc = P[start:end]
            dists = xp.abs(best_n[None, :] @ Pc.T + best_d)[0]
            out.append(dists <= dist_thresh)
        mask = xp.concatenate(out, axis=0)
    inliers_idx = xp.flatnonzero(mask)

    final_count = int((mask.sum()).get() if GPU else int(mask.sum()))

    return {
        'n': xp.asarray(best_n),
        'd': xp.asarray(best_d),
        'inliers_mask': xp.asarray(mask),
        'inliers_idx': xp.asarray(inliers_idx),
        'num_inliers': final_count,
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


def detect_ground(frames, max_height_threshold=0.5, min_points=100):
    """
    Detecta el plano del suelo usando RANSAC y genera una máscara.

    Args:
        frames: Frames de la cámara RealSense
        max_height_threshold: Altura máxima esperada para considerar puntos como suelo (metros)
        min_points: Mínimo número de puntos para realizar RANSAC

    Returns:
        tuple: (máscara_binaria, coeficientes_plano)
        - máscara_binaria: np.array de forma (H,W) con True en píxeles del suelo
        - coeficientes_plano: [a,b,c,d] del plano ax + by + cz + d = 0
    """
    # Obtener nube de puntos organizada
    points_xyz, _ = extract_pointcloud(frames, with_colors=False,
                                     filter_invalid=True, organized=True)

    if points_xyz is None:
        return None, None

    # Usar ransac_plane_gpu con orientación vertical de RealSense
    result = ransac_plane_gpu(
        points_xyz.reshape(-1, 3),
        dist_thresh=0.02,
        max_iters=2000,
        min_inliers=min_points,
        up_axis=(0.0, -1.0, 0.0),  # RealSense: Y es hacia abajo
        max_angle_deg=20.0
    )

    if result is None:
        return None, None

    # Convertir máscara a formato de imagen
    H, W = points_xyz.shape[:2]
    ground_mask = result['inliers_mask'].reshape(H, W)
    
    # Coeficientes del plano
    n = result['n']
    d = result['d']
    plane_coef = [float(n[0]), float(n[1]), float(n[2]), float(d)]

    return ground_mask.astype(np.uint8), plane_coef


def apply_ground_mask_to_rgb(rgb_image, ground_mask):
    """
    Aplica la máscara del suelo a una imagen RGB.

    Args:
        rgb_image: Imagen RGB/BGR original
        ground_mask: Máscara binaria del suelo

    Returns:
        np.array: Imagen con el suelo marcado
    """
    if ground_mask is None or rgb_image is None:
        return rgb_image
    # Crear copia de la imagen
    result = rgb_image.copy()
    # Aplicar tinte verde semi-transparente al suelo
    overlay = result.copy()
    overlay[ground_mask > 0] = (0, 255, 0)  # Verde en BGR
    # Combinar original con overlay
    cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
    return result

# =======================
# Ejemplo de uso mínimo
# =======================
if __name__ == "__main__":
    # Simulación: plano z=0 (suelo) y z=2.5 (techo) con ligero ruido
    np.random.seed(0)
    N = 50000
    xy = np.random.uniform(-3, 3, size=(N // 2, 2))
    z_floor = np.random.normal(0.0, 0.005, size=(N // 2, 1))
    floor_pts = np.hstack([xy, z_floor])

    xy2 = np.random.uniform(-3, 3, size=(N // 2, 2))
    z_ceil = np.full((N // 2, 1), 2.5) + np.random.normal(0.0, 0.005, size=(N // 2, 1))
    ceil_pts = np.hstack([xy2, z_ceil])

    pts = np.vstack([floor_pts, ceil_pts]).astype(np.float32)

    # Mundo Z-up -> up_axis=(0,0,1)
    floor, ceiling = extract_floor_and_ceiling(
        pts,
        dist_thresh=0.02,
        max_iters=1500,
        min_inliers=1500,
        up_axis=(0.0, 0.0, 1.0),
        max_angle_deg=15.0,
        seed=42,
    )

    backend = "GPU (CuPy)" if GPU else "CPU (NumPy)"
    print(f"Backend: {backend}")

    if floor is not None:
        print("Floor inliers:", floor['num_inliers'])
        n = floor['n'].get() if GPU else floor['n']
        d = floor['d'].get() if GPU else floor['d']
        print("Floor plane: n =", np.asarray(n), " d =", float(d))

    if ceiling is not None:
        print("Ceiling inliers:", ceiling['num_inliers'])
        n = ceiling['n'].get() if GPU else ceiling['n']
        d = ceiling['d'].get() if GPU else ceiling['d']
        print("Ceiling plane: n =", np.asarray(n), " d =", float(d))

    # Ejemplo (comentado) con RealSense
    # frames = ...  # Obtener frames de la cámara RealSense
    # ground_mask, plane_coef = detect_ground(frames)
    # if ground_mask is not None:
    #     print("Ground plane coefficients:", plane_coef)
    #     rgb_image = ...  # Imagen RGB correspondiente
    #     result_image = apply_ground_mask_to_rgb(rgb_image, ground_mask)
    #     # Mostrar o guardar result_image según sea necesario
