import math
import numpy as np
import cv2
from viewCamera import init_camera, extract_rgb, extract_depth_meters, precompute_rays_from_pipeline
import time
import cupy as cp
import threading
# =======================
# Ejemplo de uso mínimo
# =======================

def process_rgb_pipeline(rgb_image, result_dict):
    """
    Procesa la imagen RGB con el nuevo enfoque:
    1) Blanco y negro
    2) Filtro bilateral (d=7)
    3) CLAHE (clip=2.0, tiles=24x24)
    4) Sobel (ksize=5) y magnitud de gradiente
    5) Cierre morfológico (7x7)

    Guarda:
    - result_dict['processed_base']: imagen 3 canales del resultado final
    - result_dict['processed_img']: lo mismo para mostrar en ventana paralela
    """
    if rgb_image is None:
        result_dict['processed_img'] = None
        result_dict['processed_base'] = None
        return

    # 1. Escala de grises (sin recorte)
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    # 2. Filtro bilateral (sin recorte)
    bilateral = cv2.bilateralFilter(gray, d=7, sigmaColor=75, sigmaSpace=75)

    # 3. CLAHE antes de Sobel (sin recorte)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(24, 24))
    enhanced = clahe.apply(bilateral)

    # 4. Gradiente Sobel (sin recorte)
    gx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=5)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    mag_u8 = mag.astype(np.uint8)

    # 5. Cierre morfológico (sin recorte)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(mag_u8, cv2.MORPH_CLOSE, kernel)

    # Salidas (sin recorte)
    processed_base = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    result_dict['processed_base'] = processed_base
    result_dict['processed_img'] = processed_base.copy()


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

        # 3) Filtro de orientación mediante coseno
        cosang = cp.abs(n_unit @ up)  # (K,)
        valid = cp.logical_and(valid, cosang >= cos_thresh)

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
    if ground_mask is None:
        return processed_base if processed_base is not None else rgb_image
    
    # Usar imagen procesada si está disponible, sino usar RGB original
    if processed_base is not None:
        result = _to_numpy(processed_base)
    else:
        result = _to_numpy(rgb_image)
    
    mask = _to_numpy(ground_mask)

    if result is None:
        return rgb_image

    # Normalizar máscara a 2D y tipo booleano
    if mask is None:
        return result
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

# =======================
# Ejemplo de uso mínimo
# =======================
if __name__ == "__main__":
    # Visualiza la imagen RGB con el suelo segmentado (en vivo)
    print("Inicializando cámara RealSense…")
    pipeline = init_camera(640, 480, 640, 480, 30)

    # Precomputar rayos (una sola vez) usando utilidades de viewCamera
    rays_np, H, W = precompute_rays_from_pipeline(pipeline)
    rays_cp = cp.asarray(rays_np)  # (H,W,3)

    # Parámetros runtime para mantener FPS
    ground_every = 5             # calcular RANSAC cada N frames
    dist_thresh_run = 0.04       # tolerancia ligeramente mayor
    max_iters_run = 800          # menos iteraciones
    min_inliers_run = 600        # umbral acorde a subsampling
    subsample_stride = 4         # muestreo 1/s^2 para RANSAC

    last_n_cp = None
    last_d_cp = None
    last_thresh = dist_thresh_run
    frame_idx = 0
    t0 = time.perf_counter()
    fps_avg = 0.0
    cv2.namedWindow('Detección de Suelo - RealSense', cv2.WINDOW_NORMAL)
    try:
        result_dict = {'processed_img': None, 'processed_base': None}
        rgb_thread = None
        while True:
            frames = pipeline.wait_for_frames()

            # Extraer RGB y Depth (m) de forma ligera
            rgb_image = extract_rgb(frames)
            depth_m = extract_depth_meters(frames)
            if rgb_image is None or depth_m is None:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                continue

            # Lanzar procesamiento paralelo de la imagen RGB
            if rgb_thread is None or not rgb_thread.is_alive():
                rgb_thread = threading.Thread(target=process_rgb_pipeline, args=(rgb_image, result_dict))
                rgb_thread.start()

            # Asegurar shapes esperadas
            # No recortar ni redimensionar la imagen RGB en ningún paso
            if depth_m.shape[0] != H or depth_m.shape[1] != W:
                # Redimensionar depth con nearest si la cámara entrega otra resolución
                depth_m = cv2.resize(depth_m, (W, H), interpolation=cv2.INTER_NEAREST)

            frame_idx += 1
            depth_cp = cp.asarray(depth_m, dtype=cp.float32)
            ground_mask = None

            # Detectar suelo solo cada 'ground_every' frames; entre medias, reusar el plano
            ran_now = (frame_idx % ground_every) == 1 or (last_n_cp is None)
            if ran_now:
                # Submuestreo para RANSAC
                Dsub = depth_cp[::subsample_stride, ::subsample_stride]
                Rsub = rays_cp[::subsample_stride, ::subsample_stride]
                # Puntos válidos
                valid = Dsub > 0
                if int(cp.sum(valid)) < min_inliers_run:
                    img = rgb_image
                else:
                    Psub = (Rsub.reshape(-1, 3) * Dsub.reshape(-1, 1)).astype(cp.float32)
                    Psub = Psub[valid.reshape(-1)]
                    Psub_np = Psub
                    res = ransac_plane_gpu(Psub_np, dist_thresh=dist_thresh_run, max_iters=max_iters_run,
                                           min_inliers=min_inliers_run, up_axis=(0.0, -1.0, 0.0),
                                           max_angle_deg=20.0, seed=42,
                                           score_subset=8192)
                    if res is not None:
                        last_n_cp = res['n']
                        last_d_cp = res['d']
                        last_thresh = dist_thresh_run
                    else:
                        last_n_cp = None
                        last_d_cp = None

            # Construir máscara con el mejor plano actual (si existe)
            if last_n_cp is not None:
                dotnr = cp.tensordot(rays_cp, last_n_cp, axes=([2], [0]))
                dists = cp.abs(depth_cp * dotnr + last_d_cp)
                valid_depth = depth_cp > 0
                mask = (dists <= last_thresh) & valid_depth
                ground_mask = _to_numpy(mask).astype(np.uint8)
            else:
                ground_mask = np.zeros((H, W), dtype=np.uint8)

            # Aplicar máscara a la imagen procesada (unsharp)
            img = apply_ground_mask_to_rgb(rgb_image, ground_mask, result_dict.get('processed_base'))

            # HUD simple con FPS y estado
            dt = time.perf_counter() - t0
            t0 = time.perf_counter()
            fps = 1.0 / max(dt, 1e-6)
            fps_avg = 0.9 * fps_avg + 0.1 * fps if fps_avg > 0 else fps
            inl = int(np.sum(ground_mask > 0)) if ground_mask is not None else 0

            hud1 = f"FPS:{fps_avg:4.1f}  {'RANSAC' if ran_now else 'mask'}  Inliers:{inl}"
            cv2.putText(img, hud1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(img, "Suelo detectado en verde | ESC para salir", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 1, cv2.LINE_AA)

            # Mostrar imagen procesada en ventana aparte
            if result_dict['processed_img'] is not None:
                cv2.imshow('Procesamiento RGB paralelo', result_dict['processed_img'])
            cv2.imshow('Detección de Suelo - RealSense', img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
