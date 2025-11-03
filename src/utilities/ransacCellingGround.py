
import numpy as np
import cv2
import time
import cupy as cp
from viewCamera import extract_pointcloud_gpu, init_camera, extract_rgb
import math

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
                     seed=42,
                     **kwargs):
    """
    Aplica RANSAC para encontrar el plano del suelo en la nube de puntos.
    Retorna una máscara booleana de los puntos que pertenecen al plano.
    """
    # Convertir a cupy y reducir la nube si es muy grande
    P = cp.asarray(points).astype(cp.float32)
    N = int(P.shape[0])
    if N < 3:
        return None
    # Parámetros configurables para velocidad/calidad
    max_points = kwargs.get('max_points', 2000) if 'max_points' in kwargs else 2000
    iteraciones = min(max_iters, kwargs.get('max_iters', 100)) if 'max_iters' in kwargs else min(max_iters, 100)
    # Muestreo estratificado si hay demasiados puntos
    if N > max_points:
        # Estratificar por X (horizontal)
        x = cp.asnumpy(P[:, 0])
        bins = np.linspace(np.min(x), np.max(x), 10)
        idx_sample = []
        for i in range(len(bins)-1):
            idx_bin = np.where((x >= bins[i]) & (x < bins[i+1]))[0]
            if len(idx_bin) > 0:
                n_bin = max(1, int(max_points/9))
                idx_sample.extend(list(np.random.choice(idx_bin, min(n_bin, len(idx_bin)), replace=False)))
        idx_sample = np.array(idx_sample)
        P = P[idx_sample]
        N = P.shape[0]
        idx_sample = cp.asarray(idx_sample)
    else:
        idx_sample = cp.arange(N)
    up = cp.asarray(up_axis, dtype=cp.float32)
    up = up / (cp.linalg.norm(up) + 1e-9)
    cos_thresh = math.cos(math.radians(float(max_angle_deg)))
    rng_state = cp.random.RandomState(seed)
    rand_fn = lambda shape: rng_state.randint(0, N, size=shape, dtype=cp.int32)
    best_count = -1
    best_n = None
    best_d = None
    best_mask = None
    for _ in range(iteraciones):
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
        count = cp.count_nonzero(mask)
        if count > best_count and count >= min_inliers:
            best_count = count
            best_n = n_unit
            best_d = d
            best_mask = mask
    if best_count < 0:
        return None
    # Crear máscara del tamaño original
    mask_full = cp.zeros(points.shape[0], dtype=bool)
    mask_full[idx_sample.get()] = best_mask.get()
    return mask_full

def visualizar_resultado():
    """
    Visualiza el resultado de la máscara aplicada en la imagen RGB y muestra retardos.
    """
    print("Inicializando cámara RealSense…")
    pipeline = init_camera(640, 480, 640, 480, 30)
    cv2.namedWindow('Visualización Suelo', cv2.WINDOW_NORMAL)
    def render_puntos_suelo(puntos, mask_suelo, out_size=(480, 480)):
        """
        Renderiza la nube de puntos frontal, pintando suelo en verde y el resto en gris.
        """
        # Proyección simple: X, Z (asumiendo Y es altura)
        img = np.zeros((out_size[1], out_size[0], 3), dtype=np.uint8)
        if puntos is None or puntos.shape[0] == 0:
            return img
        # Normalizar X y Z para que quepan en la imagen
        x = puntos[:, 0]
        z = puntos[:, 2]
        # Evitar valores extremos
        x_min, x_max = np.percentile(x, [1, 99])
        z_min, z_max = np.percentile(z, [1, 99])
        x_img = ((x - x_min) / (x_max - x_min + 1e-6) * (out_size[0] - 1)).astype(int)
        z_img = ((z - z_min) / (z_max - z_min + 1e-6) * (out_size[1] - 1)).astype(int)
        # Colorear suelo y no suelo
        for i in range(puntos.shape[0]):
            color = (0, 255, 0) if mask_suelo[i] else (180, 180, 180)
            cv2.circle(img, (x_img[i], z_img[i]), 1, color, -1)
        return img
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
            puntos_ransac = None
            mask_ransac = None
            if pts_gpu is not None and pts_gpu.shape[0] > 0:
                mask = ransac_plane_gpu(pts_gpu, dist_thresh=0.02, max_iters=1000, min_inliers=500)
                if mask is not None:
                    mask_ransac = mask.get()
                    ground_mask = cp.asnumpy(mask).astype(np.uint8)
                    puntos_ransac = cp.asnumpy(pts_gpu)[mask_ransac]
            t4 = time.perf_counter()

            # Visualizar nube de puntos frontal del resultado RANSAC
            if pts_gpu is not None and mask_ransac is not None:
                puntos_cpu = cp.asnumpy(pts_gpu)
                nube_img = render_puntos_suelo(puntos_cpu, mask_ransac, out_size=(480, 480))
                cv2.imshow('Nube RANSAC Frontal', nube_img)

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
