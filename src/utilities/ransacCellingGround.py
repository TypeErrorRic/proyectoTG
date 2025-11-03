
import numpy as np
import cv2
import time
import cupy as cp
from viewCamera import extract_pointcloud_gpu, init_camera, extract_rgb

def obtener_nube_puntos(frames, stride=1, skip_top_ratio=0.25, max_distance_m=3.5):
    """
    Extrae la nube de puntos usando extract_pointcloud_gpu de viewCamera.
    """
    pts_gpu = extract_pointcloud_gpu(frames, stride=stride, skip_top_ratio=skip_top_ratio, max_distance_m=max_distance_m)
    return pts_gpu

def pintar_mascara_suelo(rgb_image, ground_mask):
    """
    Pinta la máscara sobre la imagen RGB.
    """
    if ground_mask is None or rgb_image is None:
        return rgb_image
    result = rgb_image.copy()
    mask = ground_mask
    if mask.shape[:2] != result.shape[:2]:
        mask = cv2.resize(mask, (result.shape[1], result.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_bool = (mask > 0)
    overlay = result.copy()
    overlay[mask_bool] = (0, 255, 0)
    cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
    return result


def ransac_plane_gpu(points,
                     dist_thresh=0.02,
                     max_iters=1000,
                     min_inliers=500,
                     up_axis=(0.0, -1.0, 0.0),
                     max_angle_deg=20.0,
                     seed=42):
    """
    Aplica RANSAC para encontrar el plano del suelo en la nube de puntos.
    Retorna una máscara booleana de los puntos que pertenecen al plano.
    """
    import math
    P = cp.asarray(points).astype(cp.float32)
    N = int(P.shape[0])
    if N < 3:
        return None
    up = cp.asarray(up_axis, dtype=cp.float32)
    up = up / (cp.linalg.norm(up) + 1e-9)
    cos_thresh = math.cos(math.radians(float(max_angle_deg)))
    rng_state = cp.random.RandomState(seed)
    rand_fn = lambda shape: rng_state.randint(0, N, size=shape, dtype=cp.int32)
    best_count = -1
    best_n = None
    best_d = None
    for _ in range(max_iters):
        idxs = rand_fn((3,))
        a, b, c = P[idxs[0]], P[idxs[1]], P[idxs[2]]
        ab = b - a
        ac = c - a
        n = cp.cross(ab, ac)
        norm = cp.linalg.norm(n)
        if norm < 1e-8:
            continue
        n_unit = n / (norm + 1e-12)
        d = -cp.sum(n_unit * a)
        cosang = cp.abs(n_unit @ up)
        if cosang < cos_thresh:
            continue
        dists = cp.abs(n_unit[None, :] @ P.T + d)[0]
        mask = dists <= dist_thresh
        count = int(cp.sum(mask).get())
        if count > best_count and count >= min_inliers:
            best_count = count
            best_n = n_unit
            best_d = d
            best_mask = mask
    if best_count < 0:
        return None
    return best_mask

def visualizar_resultado():
    """
    Visualiza el resultado de la máscara aplicada en la imagen RGB y muestra retardos.
    """
    print("Inicializando cámara RealSense…")
    pipeline = init_camera(640, 480, 640, 480, 30)
    cv2.namedWindow('Visualización Suelo', cv2.WINDOW_NORMAL)
    try:
        while True:
            t0 = time.perf_counter()
            frames = pipeline.wait_for_frames()
            t1 = time.perf_counter()

            # Extraer imagen RGB
            rgb_image = extract_rgb(frames)
            t2 = time.perf_counter()

            # Extraer nube de puntos (GPU)
            pts_gpu = obtener_nube_puntos(frames)
            t3 = time.perf_counter()

            # Calcular plano suelo usando RANSAC
            ground_mask = None
            if pts_gpu is not None and pts_gpu.shape[0] > 0:
                try:
                    mask = ransac_plane_gpu(pts_gpu, dist_thresh=0.02, max_iters=1000, min_inliers=500)
                    if mask is not None:
                        ground_mask = cp.asnumpy(mask).astype(np.uint8)
                except Exception:
                    ground_mask = None
            t4 = time.perf_counter()

            # Pintar máscara sobre imagen
            img = pintar_mascara_suelo(rgb_image, ground_mask) if ground_mask is not None else rgb_image
            t5 = time.perf_counter()

            # Mostrar retardos
            print(f"Adquisición: {(t1-t0)*1000:.1f} ms | RGB: {(t2-t1)*1000:.1f} ms | Nube: {(t3-t2)*1000:.1f} ms | RANSAC: {(t4-t3)*1000:.1f} ms | Pintar: {(t5-t4)*1000:.1f} ms", flush=True)

            cv2.imshow('Visualización Suelo', img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    visualizar_resultado()
