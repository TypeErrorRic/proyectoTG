import math
import numpy as np
import cv2
import pyrealsense2 as rs
from src.utilities.viewCamera import (
    init_camera,
    extract_rgb,
    extract_depth_meters,
    precompute_rays_for_stream,
)
import time
import cupy as cp
from typing import Optional

# Parámetros por defecto para mantener FPS aceptable
GROUND_EVERY = 20           # calcular RANSAC cada 20 frames
DIST_THRESH_RUN = 0.03      # tolerancia más estricta
MAX_ITERS_RUN = 500         # menos iteraciones para reducir retardo
MIN_INLIERS_RUN = 600       # umbral acorde a subsampling
SUBSAMPLE_STRIDE = 4        # muestreo 1/s^2 para RANSAC
RANSAC_TIME_BUDGET_MS = 50  # presupuesto de tiempo por ejecución de RANSAC

def _to_xp(a):
    """Asegura arreglo en GPU (CuPy)."""
    return cp.asarray(a)


def _to_numpy(a):
    """Convierte un arreglo (posiblemente CuPy) a NumPy, evitando copias innecesarias."""
    if a is None or isinstance(a, np.ndarray):
        return a
    try:
        # Arreglo CuPy -> NumPy
        if 'cupy' in str(type(a)):
            return a.get()
    except Exception:
        pass
    return np.asarray(a)



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
                     time_budget_ms: Optional[float] = None,
                     early_stop_ratio: float = 0.92):
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
        - orientation: 'any' (suelo o techo), 'ground' (preferir normal opuesta a up),
            'ceiling' (normal alineada con up)

    Return: dict con 'n', 'd', 'inliers_mask', 'inliers_idx', 'num_inliers'
    """
    P = _to_xp(points).astype(cp.float32)
    N = int(P.shape[0])
    if N < 3:
        return None

    # Heurística por tamaño de GPU
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
        # Fallback si choice falla: usa randint y recorta únicos simples
        tmp = rng_state.randint(0, N, size=score_subset * 2, dtype=cp.int32)
        samp_idx = cp.unique(tmp)[:score_subset]
        if samp_idx.size < score_subset:
            # rellena si faltan
            pad = score_subset - int(samp_idx.size)
            samp_idx = cp.concatenate([samp_idx, tmp[:pad]])
    P_samp = P[samp_idx]

    remaining = int(max_iters)
    start_time = time.perf_counter()
    processed_batches = 0
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
        n = cp.cross(ab, ac)  # (K,3)
        norm = cp.linalg.norm(n, axis=1)  # (K,)
        valid = norm > 1e-8
        # Evitar división por cero
        n_unit = cp.where(valid[:, None], n / (norm[:, None] + 1e-12), 0)
        d = -cp.sum(n_unit * a, axis=1)

        # 3) Filtro de orientación mediante coseno con preferencia opcional
        dot_up = n_unit @ up  # (K,)
        if orientation == 'ground':
            # normal opuesta a up (~-1)
            cond = dot_up <= -cos_thresh
        elif orientation == 'ceiling':
            # normal alineada con up (~+1)
            cond = dot_up >= cos_thresh
        else:  # 'any'
            cond = cp.abs(dot_up) >= cos_thresh
        valid = cp.logical_and(valid, cond)

        # 4) Conteo de inliers sobre SUBMUESTRA para elegir mejor modelo del lote
        #    Mucho más eficiente en Jetson que KxN directo.
        counts = cp.zeros((K,), dtype=cp.int32)
        dists_s = cp.abs(n_unit @ P_samp.T + d[:, None])  # (K,S)
        counts = cp.sum(dists_s <= dist_thresh, axis=1)

        # Invalida modelos no válidos
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
        # Time budget (si aplica): cortar si se excede, sin esperar a 2 lotes
        if time_budget_ms is not None:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            if elapsed_ms >= time_budget_ms:
                break

    if best_count < 0:
        return None

    # 6) Recalcular máscara de inliers del mejor modelo sobre TODOS los puntos (una vez)
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


def apply_ground_mask_to_rgb(rgb_image, ground_mask, processed_base=None):
    """
    Aplica la máscara del suelo a una imagen RGB.

    Args:
        rgb_image: Imagen RGB/BGR original
        ground_mask: Máscara binaria del suelo
        processed_base: Imagen procesada base (unsharp) a usar en lugar de rgb_image

    Returns:
        np.array: Imagen con el suelo marcado
    """
    # Base: priorizar la imagen procesada del secundario si está disponible; si no, usar RGB
    base = _to_numpy(processed_base) if processed_base is not None else None
    result = base if base is not None else _to_numpy(rgb_image)
    if result is None:
        return None

    # Asegurar máscara válida; si viene None, usar máscara vacía
    mask = _to_numpy(ground_mask)
    if mask is None:
        mask = np.zeros(result.shape[:2], dtype=np.uint8)

    # Normalizar máscara a 2D y tipo booleano
    if mask.ndim == 3 and mask.shape[-1] in (1, 3):
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if mask.shape[-1] == 3 else mask.squeeze(-1)
    if mask.ndim == 1 and mask.size == result.shape[0] * result.shape[1]:
        mask = mask.reshape(result.shape[:2])
    if mask.shape[:2] != result.shape[:2]:
        # Intentar redimensionar de forma segura (nearest) para máscaras
        mask = cv2.resize(mask, (result.shape[1], result.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask_bool = (mask > 0)

    # Crear overlay verde sobre el suelo
    overlay = result.copy()
    overlay[mask_bool] = (0, 255, 0)  # Verde en BGR
    # Combinar original con overlay
    cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
    return result 


last_n_cp = None
last_d_cp = None
last_thresh = DIST_THRESH_RUN

def get_ground(rgb_image: np.ndarray, mapaProfundidad: np.ndarray, rays_cp: cp.ndarray, H: int, W: int, subsample_stride=SUBSAMPLE_STRIDE, min_inliers=MIN_INLIERS_RUN) -> Optional[np.ndarray]:
    """
    Detecta el plano de suelo (RANSAC) y devuelve la imagen RGB
    con la máscara del suelo pintada en verde.

    Args:
        rgb_image: Imagen RGB
        mapaProfundidad: Mapa de profundidad
        rays_cp: Rayos precalculados en cupy
        H, W: Alto y ancho de la imagen
        subsample_stride: Submuestreo para RANSAC
        min_inliers: Mínimo de inliers para aceptar plano

    Returns:
        np.ndarray | None: imagen BGR con el suelo pintado o None si no hay datos válidos.
    """
    # Convertir depth a cupy para RANSAC
    depth_cp = cp.asarray(mapaProfundidad, dtype=cp.float32)
    # Submuestreo para RANSAC
    Dsub = depth_cp[::subsample_stride, ::subsample_stride]
    Rsub = rays_cp[::subsample_stride, ::subsample_stride]
    # Limitar al 50% inferior para sesgar hacia el suelo
    sub_h = Dsub.shape[0]
    if sub_h >= 2:
        Dsub = Dsub[sub_h//2:, :]
        Rsub = Rsub[sub_h//2:, :]
    valid = Dsub > 0
    if int(cp.sum(valid)) >= min_inliers:
        Psub = (Rsub.reshape(-1, 3) * Dsub.reshape(-1, 1)).astype(cp.float32)
        Psub = Psub[valid.reshape(-1)]

        res = ransac_plane_gpu(
            Psub,
            dist_thresh=last_thresh,
            max_iters=MAX_ITERS_RUN,
            min_inliers=min_inliers,
            up_axis=(0.0, -1.0, 0.0),
            max_angle_deg=45.0,
            seed=42,
            score_subset=2048,
            orientation='ground',
            time_budget_ms=RANSAC_TIME_BUDGET_MS,
            early_stop_ratio=0.92,
            batch_size=256,
        )

        if res is not None:
            last_n_cp = res['n']
            last_d_cp = res['d']
            # last_thresh se mantiene igual
        else:
            last_n_cp = None
            last_d_cp = None
    else:
        # Datos insuficientes, no se ejecuta RANSAC este frame
        pass

    # Construir máscara del mejor plano actual (si existe)
    if last_n_cp is not None:
        dotnr = cp.tensordot(rays_cp, last_n_cp, axes=([2], [0]))
        dists = cp.abs(depth_cp * dotnr + last_d_cp)
        valid_depth = depth_cp > 0
        mask = (dists <= last_thresh) & valid_depth
        ground_mask = _to_numpy(mask).astype(np.uint8)
    else:
        ground_mask = np.zeros((H, W), dtype=np.uint8)

    img = apply_ground_mask_to_rgb(rgb_image, ground_mask)

    return img
