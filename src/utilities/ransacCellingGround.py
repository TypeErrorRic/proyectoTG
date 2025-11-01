import math
import numpy as np
import cv2
from typing import Optional
from viewCamera import extract_pointcloud, render_pointcloud, init_camera
import time
import cupy as cp


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
    P = _to_xp(points).astype(cp.float32)
    up = cp.asarray(up_axis, dtype=cp.float32)

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
        h1 = cp.mean(pts1 @ up)
        floor, ceiling = (res1, None) if float(h1.get()) < 0 else (None, res1)
        return floor, ceiling

    # Clasificar por altura (proyección sobre +up)
    pts1 = P[res1['inliers_idx']]
    pts2 = P2[res2['inliers_idx']]
    h1 = cp.mean(pts1 @ up)
    h2 = cp.mean(pts2 @ up)

    if float(h1.get()) < float(h2.get()):
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

    # Convertir máscara a formato de imagen (NumPy)
    H, W = points_xyz.shape[:2]
    ground_mask = _to_numpy(result['inliers_mask']).reshape(H, W)
    
    # Coeficientes del plano
    n = _to_numpy(result['n'])
    d = _to_numpy(result['d'])
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
    # Asegurar arrays NumPy
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


def _build_colored_pointcloud(points_np: np.ndarray, inliers_idx_cp: cp.ndarray,
                              green=(0, 255, 0), base=(200, 200, 200),
                              base_colors_np: Optional[np.ndarray] = None):
    """
    Devuelve (points_np, colors_np) donde los puntos en 'inliers_idx_cp' se colorean de verde.
    - points_np: np.ndarray (N,3) float32 en metros (no se copia si ya lo es)
    - inliers_idx_cp: cp.ndarray de índices de inliers (GPU)
    - green/base: tuplas BGR
    """
    N = int(points_np.shape[0])
    if base_colors_np is not None and isinstance(base_colors_np, np.ndarray) and base_colors_np.shape == (N, 3):
        colors_gpu = cp.asarray(base_colors_np, dtype=cp.uint8)
    else:
        colors_gpu = cp.full((N, 3), base, dtype=cp.uint8)
    colors_gpu[inliers_idx_cp] = cp.asarray(green, dtype=cp.uint8)
    colors_np = colors_gpu.get()
    # Asegurar dtype/contiguidad de puntos
    if not (isinstance(points_np, np.ndarray) and points_np.dtype == np.float32):
        points_np = np.asarray(points_np, dtype=np.float32)
    return points_np, colors_np


def demo_main_return_colored_pointcloud():
    """
    Genera una nube de puntos sintética (suelo + techo), detecta el suelo con RANSAC
    y retorna (points_np, colors_np) con el suelo coloreado en verde.
    """
    np.random.seed(0)
    N = 50000
    xy = np.random.uniform(-3, 3, size=(N // 2, 2))
    z_floor = np.random.normal(0.0, 0.005, size=(N // 2, 1))
    floor_pts = np.hstack([xy, z_floor])

    xy2 = np.random.uniform(-3, 3, size=(N // 2, 2))
    z_ceil = np.full((N // 2, 1), 2.5) + np.random.normal(0.0, 0.005, size=(N // 2, 1))
    ceil_pts = np.hstack([xy2, z_ceil])

    pts = np.vstack([floor_pts, ceil_pts]).astype(np.float32)

    # Detectar suelo/techo (mundo Z-up)
    floor, ceiling = extract_floor_and_ceiling(
        pts,
        dist_thresh=0.02,
        max_iters=1500,
        min_inliers=1500,
        up_axis=(0.0, 0.0, 1.0),
        max_angle_deg=15.0,
        seed=42,
    )

    if floor is not None:
        points_out, colors_out = _build_colored_pointcloud(pts, floor['inliers_idx'])
    else:
        # Sin suelo detectado: todo gris
        colors_out = np.full((pts.shape[0], 3), (200, 200, 200), dtype=np.uint8)
        points_out = pts

    return points_out, colors_out

# =======================
# Ejemplo de uso mínimo
# =======================
if __name__ == "__main__":
    # Visualiza la nube de puntos de la RealSense coloreando el suelo en verde (en vivo)
    print("Inicializando cámara RealSense…")
    pipeline = init_camera(640, 480, 640, 480, 30)
    # Parámetros runtime para mantener FPS
    ground_every = 10            # calcular RANSAC cada N frames
    dist_thresh_run = 0.03       # tolerancia algo mayor para robustez
    max_iters_run = 800          # menos iteraciones para acelerar
    min_inliers_run = 1200

    # Parámetros de visualización (controles interactivos tipo viewCamera)
    yaw, pitch, roll = -45.0, 25.0, 0.0
    fov = 60.0
    point_size = 1
    add_tz = 0.0
    pan_tx, pan_ty = 0.0, 0.0

    step_angle = 5.0
    step_zoom = 0.2   # metros
    step_pan = 0.05   # metros
    step_fov = 5.0

    last_n_cp = None
    last_d_cp = None
    last_thresh = dist_thresh_run
    frame_idx = 0
    t0 = time.perf_counter()
    fps_avg = 0.0
    cv2.namedWindow('PointCloud - Suelo en verde (RealSense)', cv2.WINDOW_NORMAL)
    try:
        while True:
            frames = pipeline.wait_for_frames()
            points_xyz, colors_bgr = extract_pointcloud(frames, with_colors=True, filter_invalid=True, organized=False)
            if points_xyz is None or len(points_xyz) == 0:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                continue

            frame_idx += 1
            pts_np = np.asarray(points_xyz, dtype=np.float32)

            # Detectar suelo solo cada 'ground_every' frames; entre medias, reusar el plano
            ran_now = (frame_idx % ground_every) == 1 or (last_n_cp is None)
            if ran_now:
                res = ransac_plane_gpu(pts_np, dist_thresh=dist_thresh_run, max_iters=max_iters_run,
                                       min_inliers=min_inliers_run, up_axis=(0.0, -1.0, 0.0),
                                       max_angle_deg=20.0, seed=42)
                if res is not None:
                    last_n_cp = res['n']
                    last_d_cp = res['d']
                    last_thresh = dist_thresh_run
                    inds = res['inliers_idx']
                    pts_np, colors_np = _build_colored_pointcloud(pts_np, inds, base_colors_np=colors_bgr)
                else:
                    last_n_cp = None
                    last_d_cp = None
                    colors_np = colors_bgr if colors_bgr is not None else np.full((pts_np.shape[0], 3), (200, 200, 200), dtype=np.uint8)
            else:
                # Reusar plano previo: máscara rápida en GPU
                Pc = cp.asarray(pts_np, dtype=cp.float32)
                dists = cp.abs(last_n_cp[None, :] @ Pc.T + last_d_cp)[0]
                mask = dists <= last_thresh
                inds = cp.flatnonzero(mask)
                colors_np = colors_bgr if colors_bgr is not None else np.full((pts_np.shape[0], 3), (200, 200, 200), dtype=np.uint8)
                _, colors_np = _build_colored_pointcloud(pts_np, inds, base_colors_np=colors_np)

            img = render_pointcloud(pts_np, colors_np,
                                     out_size=(720, 720),
                                     yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
                                     fov_deg=fov, point_size=point_size,
                                     add_tz=add_tz, tx=pan_tx, ty=pan_ty)
            # HUD simple con FPS y estado
            dt = time.perf_counter() - t0
            t0 = time.perf_counter()
            fps = 1.0 / max(dt, 1e-6)
            fps_avg = 0.9 * fps_avg + 0.1 * fps if fps_avg > 0 else fps
            hud1 = f"FPS:{fps_avg:4.1f}  {'RANSAC' if ran_now else 'mask'}  N={pts_np.shape[0]}"
            hud2 = f"Yaw:{yaw:.0f}  Pitch:{pitch:.0f}  Roll:{roll:.0f}  FOV:{fov:.0f}  Size:{point_size}  Zoff:{add_tz:+.2f}  Pan({pan_tx:+.2f},{pan_ty:+.2f})"
            cv2.putText(img, hud1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(img, hud2, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 1, cv2.LINE_AA)
            cv2.putText(img, "Controles: WASD rotar | Q/E roll | Z/X FOV | +/- tamaño | I/K/J/L paneo | [/ ] z-off | R reset | ESC salir", (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA)
            cv2.imshow('PointCloud - Suelo en verde (RealSense)', img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC para salir
                break
            # Controles interactivos de visualización
            if key == ord('a'):
                yaw -= step_angle
            elif key == ord('d'):
                yaw += step_angle
            elif key == ord('w'):
                pitch += step_angle
            elif key == ord('s'):
                pitch -= step_angle
            elif key == ord('q'):
                roll -= step_angle
            elif key == ord('e'):
                roll += step_angle
            elif key == ord('z'):
                fov = max(20.0, fov - step_fov)
            elif key == ord('x'):
                fov = min(120.0, fov + step_fov)
            elif key in (ord('+'), ord('=')):
                point_size = min(6, point_size + 1)
            elif key in (ord('-'), ord('_')):
                point_size = max(1, point_size - 1)
            elif key == ord('i'):
                pan_ty -= step_pan
            elif key == ord('k'):
                pan_ty += step_pan
            elif key == ord('j'):
                pan_tx -= step_pan
            elif key == ord('l'):
                pan_tx += step_pan
            elif key == ord(']'):
                add_tz += step_zoom
            elif key == ord('['):
                add_tz -= step_zoom
            elif key == ord('r'):
                yaw, pitch, roll = -45.0, 25.0, 0.0
                fov = 60.0
                point_size = 1
                add_tz = 0.0
                pan_tx, pan_ty = 0.0, 0.0
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
