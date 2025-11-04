import math
import numpy as np
import cv2
import pyrealsense2 as rs
from viewCamera import (
    init_camera,
    extract_rgb,
    extract_depth_meters,
    precompute_rays_for_stream,
)
import time
import cupy as cp
import threading
from typing import Optional

# (Alineación GPU Depth->Color movida a viewCamera.make_depth_to_color_aligner)

# =======================
# Estado y parámetros runtime (para get_ground)
# =======================

# Diccionario de estado para evitar variables globales sueltas
_runtime = {
    'initialized': False,
    'pipeline': None,
    'rays_cp': None,
    'H': None,
    'W': None,
    'align_depth_fn': None,
    'params': None,
    'last_n_cp': None,
    'last_d_cp': None,
    'last_thresh': None,
    'frame_idx': 0,
    'result_dict': {'processed_base': None},
    'rgb_thread': None,            # worker persistente
    'rgb_done_event': None,        # señala cuando hay processed_base listo
    'rgb_work_event': None,        # señala cuando hay nuevo RGB para procesar
    'rgb_lock': None,              # protege acceso a rgb_input
    'rgb_input': None,             # buzón tamaño 1 para el worker
    'fps_t0': None,
    'subsample_stride': None,
}

# Parámetros por defecto para mantener FPS aceptable
GROUND_EVERY = 20           # calcular RANSAC cada 20 frames
DIST_THRESH_RUN = 0.03      # tolerancia más estricta
MAX_ITERS_RUN = 500         # menos iteraciones para reducir retardo
MIN_INLIERS_RUN = 600       # umbral acorde a subsampling
SUBSAMPLE_STRIDE = 4        # muestreo 1/s^2 para RANSAC
RANSAC_TIME_BUDGET_MS = 50  # presupuesto de tiempo por ejecución de RANSAC

def process_rgb_pipeline(rgb_image, result_dict, done_event: Optional[threading.Event] = None):
    """
    Aplica Watershed guiado por gradientes para extraer formas a partir de la imagen RGB.

    Resumen del pipeline:
    1) Gray + CLAHE suave
    2) Filtro bilateral (preserva bordes)
    3) Gradiente (Sobel) + cierre morfológico + unsharp ligero
    4) Marcadores (foreground/background) con distanceTransform
    5) cv2.watershed sobre la imagen original
    6) Devuelve una imagen BGR con contornos del Watershed en rojo

    Guarda en:
    - result_dict['processed_base']: imagen BGR con formas delineadas (3 canales)
    """
    if rgb_image is None:
        result_dict['processed_base'] = None
        if done_event is not None:
            try:
                done_event.set()
            except Exception:
                pass
        return

    # 1. Escala de grises (sin recorte)
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    # 2. CLAHE ligero (uniforme)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(32, 32))
    gray = clahe.apply(gray)

    # 3. Filtro bilateral (más ligero)
    bilateral = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    # 4. Gradiente Sobel (sin recorte)
    gx = cv2.Sobel(bilateral, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(bilateral, cv2.CV_64F, 0, 1, ksize=5)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    mag_u8 = mag.astype(np.uint8)

    # 5. Cierre morfológico medio para consolidar bordes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(mag_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 6. Unsharp mask (ligero) para realzar gradientes
    amount = 1.5
    blurred = cv2.GaussianBlur(closed, (0, 0), sigmaX=1.0, sigmaY=1.0)
    sharp = cv2.addWeighted(closed, 1.0 + amount, blurred, -amount, 0)

    # 7. Bordes binarios con Otsu
    _, edges_otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges_otsu = cv2.morphologyEx(edges_otsu, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    # 8. Preparar marcadores para Watershed a partir de regiones internas
    regions = cv2.bitwise_not(edges_otsu)  # interior de formas
    regions = cv2.morphologyEx(regions, cv2.MORPH_OPEN, kernel, iterations=1)
    sure_bg = cv2.dilate(regions, kernel, iterations=2)

    # Foreground seguro con distance transform
    dist = cv2.distanceTransform(regions, distanceType=cv2.DIST_L2, maskSize=5)
    # Normalizamos y umbralizamos automáticamente (Otsu) para robustez
    dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, sure_fg = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 9. Marcadores iniciales
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # dejar fondo como 1
    markers[unknown == 255] = 0  # regiones desconocidas a 0

    # 10. Watershed sobre la imagen original
    img_ws = rgb_image if rgb_image.ndim == 3 else cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_ws.copy(), markers)

    # 11. Visualización: mostrar OTSU (cerrado) con contornos del Watershed en rojo
    base_otsu = cv2.cvtColor(edges_otsu, cv2.COLOR_GRAY2BGR)
    base_otsu[markers == -1] = (0, 0, 255)  # contornos Watershed en rojo
    result_dict['processed_base'] = base_otsu
    # Señalizar que el procesamiento terminó para este frame
    if done_event is not None:
        try:
            done_event.set()
        except Exception:
            pass


def _rgb_worker_loop():
    """Hilo persistente que duerme hasta que haya un RGB en el buzón, lo procesa y vuelve a dormir."""
    while True:
        # Espera a que el principal indique que hay trabajo
        _runtime['rgb_work_event'].wait()
        try:
            _runtime['rgb_work_event'].clear()
        except Exception:
            pass

        # Tomar snapshot del RGB actual bajo lock
        rgb = None
        try:
            if _runtime['rgb_lock'] is not None:
                with _runtime['rgb_lock']:
                    rgb = _runtime['rgb_input']
            else:
                rgb = _runtime.get('rgb_input', None)
        except Exception:
            rgb = _runtime.get('rgb_input', None)

        # Procesar (esto actualizará processed_base y hará set del done_event)
        try:
            process_rgb_pipeline(rgb, _runtime['result_dict'], _runtime['rgb_done_event'])
        except Exception:
            # En caso de error, señalizar finalización para no bloquear al principal
            try:
                _runtime['rgb_done_event'].set()
            except Exception:
                pass


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

def _lazy_init():
    """Inicializa cámara y rayos en la primera llamada."""
    if _runtime['initialized']:
        return
    print("Inicializando cámara RealSense…")
    pipeline, params = init_camera(
        color_width=640,
        color_height=480,
        depth_width=640,
        depth_height=480,
        fps=30,
        stride=2,        # submuestreo para nube
        yaw=-45.0,
        pitch=25.0,
        roll=0.0,
        fov=60.0,
        point_size=1
    )
    rays_np, H, W, align_depth_fn = precompute_rays_for_stream(pipeline, rs.stream.color)
    _runtime['pipeline'] = pipeline
    _runtime['rays_cp'] = cp.asarray(rays_np)
    _runtime['H'] = H
    _runtime['W'] = W
    _runtime['align_depth_fn'] = align_depth_fn
    _runtime['params'] = params
    _runtime['last_thresh'] = DIST_THRESH_RUN
    # Usar el stride de params para el submuestreo de RANSAC si viene configurado
    try:
        _runtime['subsample_stride'] = int(params.get('stride', SUBSAMPLE_STRIDE))
    except Exception:
        _runtime['subsample_stride'] = SUBSAMPLE_STRIDE
    # Sincronización productor/consumidor para el secundario
    _runtime['rgb_done_event'] = threading.Event()  # "salida lista"
    _runtime['rgb_work_event'] = threading.Event()  # "entrada disponible"
    _runtime['rgb_lock'] = threading.Lock()
    _runtime['rgb_input'] = None
    # Iniciar el worker persistente (daemon)
    _runtime['rgb_thread'] = threading.Thread(target=_rgb_worker_loop, daemon=True)
    _runtime['rgb_thread'].start()
    _runtime['initialized'] = True


def get_ground() -> Optional[np.ndarray]:
    """
    Obtiene un frame, detecta el plano de suelo (RANSAC) y devuelve la imagen RGB
    con la máscara del suelo pintada en verde.

    Returns:
        np.ndarray | None: imagen BGR con el suelo pintado o None si no hay frame.
    """
    _lazy_init()

    pipeline = _runtime['pipeline']
    H, W = _runtime['H'], _runtime['W']
    align_depth_fn = _runtime['align_depth_fn']

    frames = pipeline.wait_for_frames()

    # Extraer RGB y Depth nativos
    rgb_image = extract_rgb(frames)
    depth_m = align_depth_fn(frames) if align_depth_fn is not None else extract_depth_meters(frames)
    if rgb_image is None or depth_m is None:
        return None

    # Publicar el RGB en el buzón y despertar al worker secundario
    try:
        if _runtime['rgb_done_event'] is not None:
            _runtime['rgb_done_event'].clear()
        if _runtime['rgb_lock'] is not None:
            with _runtime['rgb_lock']:
                _runtime['rgb_input'] = rgb_image
        else:
            _runtime['rgb_input'] = rgb_image
        if _runtime['rgb_work_event'] is not None:
            _runtime['rgb_work_event'].set()
    except Exception:
        pass

    # Asegurar shape del depth al tamaño de COLOR
    if depth_m.shape[0] != H or depth_m.shape[1] != W:
        depth_m = cv2.resize(depth_m, (W, H), interpolation=cv2.INTER_NEAREST)

    _runtime['frame_idx'] += 1
    depth_cp = cp.asarray(depth_m, dtype=cp.float32)

    # Decidir si ejecutar RANSAC ahora (y medir retardo)
    ran_now = (_runtime['frame_idx'] % GROUND_EVERY) == 1 or (_runtime['last_n_cp'] is None)
    if ran_now:
        # Submuestreo para RANSAC
        sub_stride = _runtime['subsample_stride'] or SUBSAMPLE_STRIDE
        Dsub = depth_cp[::sub_stride, ::sub_stride]
        Rsub = _runtime['rays_cp'][::sub_stride, ::sub_stride]
        # Limitar al 50% inferior para sesgar hacia el suelo
        sub_h = Dsub.shape[0]
        if sub_h >= 2:
            Dsub = Dsub[sub_h//2:, :]
            Rsub = Rsub[sub_h//2:, :]
        valid = Dsub > 0
        if int(cp.sum(valid)) >= MIN_INLIERS_RUN:
            Psub = (Rsub.reshape(-1, 3) * Dsub.reshape(-1, 1)).astype(cp.float32)
            Psub = Psub[valid.reshape(-1)]

            res = ransac_plane_gpu(
                Psub,
                dist_thresh=DIST_THRESH_RUN,
                max_iters=MAX_ITERS_RUN,
                min_inliers=MIN_INLIERS_RUN,
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
                _runtime['last_n_cp'] = res['n']
                _runtime['last_d_cp'] = res['d']
                _runtime['last_thresh'] = DIST_THRESH_RUN
            else:
                _runtime['last_n_cp'] = None
                _runtime['last_d_cp'] = None
        else:
            # Datos insuficientes, no se ejecuta RANSAC este frame
            pass

    # Construir máscara del mejor plano actual (si existe)
    H, W = _runtime['H'], _runtime['W']
    if _runtime['last_n_cp'] is not None:
        dotnr = cp.tensordot(_runtime['rays_cp'], _runtime['last_n_cp'], axes=([2], [0]))
        dists = cp.abs(depth_cp * dotnr + _runtime['last_d_cp'])
        valid_depth = depth_cp > 0
        mask = (dists <= _runtime['last_thresh']) & valid_depth
        ground_mask = _to_numpy(mask).astype(np.uint8)
    else:
        ground_mask = np.zeros((H, W), dtype=np.uint8)

    # Tomar processed_base si ya está listo; si no, esperar un instante muy corto y re-chequear
    processed_base = None
    try:
        ev = _runtime['rgb_done_event']
        if ev is not None and ev.is_set():
            processed_base = _runtime['result_dict'].get('processed_base')
        else:
            # Espera mínima (re-check) para intentar captar el resultado sin bloquear FPS
            if ev is not None:
                ev.wait(timeout=0.003)
            processed_base = _runtime['result_dict'].get('processed_base')
    except Exception:
        processed_base = _runtime['result_dict'].get('processed_base')

    img = apply_ground_mask_to_rgb(rgb_image, ground_mask, processed_base)

    # Calcular y pegar FPS en blanco sobre la imagen
    now = time.perf_counter()
    if _runtime['fps_t0'] is None:
        _runtime['fps_t0'] = now
        fps = 0.0
    else:
        dt = now - _runtime['fps_t0']
        _runtime['fps_t0'] = now
        fps = 1.0 / max(dt, 1e-6)
    cv2.putText(img, f"FPS: {fps:4.1f}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    return img


# =======================
# Ejecución directa del módulo
# =======================
if __name__ == "__main__":
    cv2.namedWindow('Detección de Suelo - RealSense', cv2.WINDOW_NORMAL)
    try:
        while True:
            img = get_ground()
            if img is None:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                continue
            cv2.imshow('Detección de Suelo - RealSense', img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    finally:
        if _runtime['pipeline'] is not None:
            _runtime['pipeline'].stop()
        cv2.destroyAllWindows()
