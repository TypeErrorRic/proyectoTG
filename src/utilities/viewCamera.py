"""
viewCamera.py
Funciones mínimas para:
- Extraer nube de puntos
- Filtro voxel por grilla
- Visualización y comunicación con ransacCellingGround
"""

import numpy as np
from typing import Optional
import pyrealsense2 as rs
import cv2
import time
import cupy as cp  # Requiere GPU/CUDA; no se agrega fallback a CPU por solicitud

# Pipeline global opcional para asegurar inicialización desde utilidades
_PIPELINE = None

def extract_pointcloud(frames: rs.composite_frame) -> np.ndarray:
    """Extrae la nube de puntos (Nx3) del frame de RealSense."""
    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        return None
    pc = rs.pointcloud()
    points = pc.calculate(depth_frame)
    verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    valid = verts[:, 2] > 0
    verts = verts[valid]
    return verts

def voxel_grid(points_xyz: np.ndarray, voxel_size: float = 0.01, min_points_per_voxel: int = 3) -> np.ndarray:
    """Filtro voxel simple en GPU con CuPy: reduce densidad y ruido de la nube de puntos.

    Agrupa por voxel y promedia los puntos por celda, descartando celdas con pocos puntos.
    """
    if points_xyz is None or len(points_xyz) == 0:
        return points_xyz

    hash_factor = 100_000  # evitar colisiones; usar enteros de 64 bits para prevenir overflow

    pts_gpu = cp.asarray(points_xyz, dtype=cp.float32)
    voxel_indices = cp.floor(pts_gpu / voxel_size).astype(cp.int64)
    voxel_hash = (
        voxel_indices[:, 0]
        + voxel_indices[:, 1] * hash_factor
        + voxel_indices[:, 2] * hash_factor * hash_factor
    ).astype(cp.int64)

    unique_hashes, inverse_indices = cp.unique(voxel_hash, return_inverse=True)
    n_voxels = int(unique_hashes.shape[0])

    # Sumas por voxel con bincount (compatible con versiones antiguas de CuPy)
    sums = cp.stack([
        cp.bincount(inverse_indices, weights=pts_gpu[:, 0], minlength=n_voxels),
        cp.bincount(inverse_indices, weights=pts_gpu[:, 1], minlength=n_voxels),
        cp.bincount(inverse_indices, weights=pts_gpu[:, 2], minlength=n_voxels),
    ], axis=1).astype(cp.float32)

    counts = cp.bincount(inverse_indices, minlength=n_voxels).astype(cp.int32)

    # Promedio por voxel y filtro por mínimo de puntos
    filtered_points = sums / counts[:, cp.newaxis]
    keep_mask = (counts >= int(max(1, min_points_per_voxel)))
    filtered_points = filtered_points[keep_mask]
    return cp.asnumpy(filtered_points)

def ensure_camera(color_w=640, color_h=480, depth_w=640, depth_h=480, fps=30):
    """Asegura que haya un pipeline inicializado (global)."""
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = init_camera(
            color_width=color_w,
            color_height=color_h,
            depth_width=depth_w,
            depth_height=depth_h,
            fps=fps,
        )
    return _PIPELINE


def get_voxel_for_ransac(frames: Optional[rs.composite_frame] = None,
                         voxel_size: float = 0.01,
                         min_points_per_voxel: int = 3,
                         pipeline: Optional[rs.pipeline] = None):
    """Extrae la nube de puntos, aplica filtro voxel y la retorna lista para RANSAC.

    - Si no se proveen frames, asegura e impulsa la inicialización de la cámara y toma un frame.
    - También acepta un pipeline explícito para tomar frames.
    """
    if frames is None:
        pipe = pipeline if pipeline is not None else ensure_camera()
        frames = pipe.wait_for_frames()

    points_xyz = extract_pointcloud(frames)
    if points_xyz is None:
        return None
    points_voxel = voxel_grid(points_xyz, voxel_size=voxel_size, min_points_per_voxel=min_points_per_voxel)
    return points_voxel

def render_pointcloud(points_xyz: np.ndarray,
                      out_size=(720, 720),
                      yaw_deg: float = -45.0,
                      pitch_deg: float = 25.0,
                      roll_deg: float = 0.0,
                      tz: Optional[float] = None,
                      tx: float = 0.0,
                      ty: float = 0.0,
                      fov_deg: float = 60.0) -> np.ndarray:
    """Renderiza la nube de puntos en 2D (sin colores) usando GPU (CuPy).

    Permite navegar con parámetros de cámara: yaw/pitch/roll (grados), traslación (tx, ty, tz) y FOV.
    Si tz es None, se auto-ajusta para situar la nube delante de la cámara.
    """
    if points_xyz is None or len(points_xyz) == 0:
        return np.zeros((out_size[0], out_size[1], 3), dtype=np.uint8)

    H, W = out_size
    img = np.zeros((H, W, 3), dtype=np.uint8)

    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]],
        dtype=np.float32,
    )
    Ry = np.array(
        [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]],
        dtype=np.float32,
    )
    Rz = np.array(
        [[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]],
        dtype=np.float32,
    )
    R = (Rz @ Ry @ Rx).astype(np.float32)

    pts = cp.asarray(points_xyz, dtype=cp.float32)
    center = cp.median(pts, axis=0)
    pts_centered = pts - center
    R_cp = cp.asarray(R)
    pts_rot = pts_centered @ R_cp.T
    z = pts_rot[:, 2]
    if tz is None:
        tz = max(0.5, -float(cp.percentile(z, 5)) + 1.5)
    pts_cam = pts_rot + cp.asarray([tx, ty, tz], dtype=cp.float32)
    f = 0.5 * H / np.tan(np.deg2rad(fov_deg) * 0.5)
    Z = cp.clip(pts_cam[:, 2], 1e-3, None)
    x_proj = (pts_cam[:, 0] * f) / Z
    y_proj = (pts_cam[:, 1] * f) / Z
    u = (W * 0.5 + x_proj).astype(cp.int32)
    v = (H * 0.5 - y_proj).astype(cp.int32)
    mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v = cp.asnumpy(u[mask]), cp.asnumpy(v[mask])

    img[v, u] = (200, 200, 200)
    return img

def init_camera(
    color_width: int = 640,
    color_height: int = 480,
    depth_width: int = 640,
    depth_height: int = 480,
    fps: int = 30,
):
    """
    Inicializa el pipeline de Intel RealSense con streams de color y profundidad.

    Parámetros:
      - color_width/height: resolución para el stream de color.
      - depth_width/height: resolución para el stream de profundidad.
      - fps: cuadros por segundo para ambos streams.

    Retorna:
      - pipeline inicializado y en ejecución.
    """
    pipeline = rs.pipeline()
    config = rs.config()

    # Habilita streams (ajusta resolución según tu modelo D435/D415/L515)
    config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, fps)

    pipeline.start(config)
    return pipeline

if __name__ == "__main__":
    pipeline = init_camera()
    print("Presiona ESC para salir...")
    import ransacCellingGround
    try:
        cv2.namedWindow('RANSAC PointCloud', cv2.WINDOW_NORMAL)
        # Estado de cámara para navegación
        yaw_deg, pitch_deg, roll_deg = -45.0, 25.0, 0.0
        tx, ty = 0.0, 0.0
        tz = None  # auto al inicio
        fov_deg = 60.0
        voxel_size = 0.01
        min_pts = 3
        while True:
            t0 = time.perf_counter()
            frames = pipeline.wait_for_frames()
            t1 = time.perf_counter()
            points_voxel = get_voxel_for_ransac(frames, voxel_size=voxel_size, min_points_per_voxel=min_pts)
            t2 = time.perf_counter()
            ransac_result = None
            if points_voxel is not None:
                ransac_result = ransacCellingGround.ransac_plane_gpu(points_voxel)
            t3 = time.perf_counter()
            img = render_pointcloud(points_voxel,
                                     out_size=(720, 720),
                                     yaw_deg=yaw_deg,
                                     pitch_deg=pitch_deg,
                                     roll_deg=roll_deg,
                                     tz=tz,
                                     tx=tx,
                                     ty=ty,
                                     fov_deg=fov_deg)
            t4 = time.perf_counter()
            hud = f"Adquisición: {(t1 - t0) * 1000:.1f} ms | Voxel: {(t2 - t1) * 1000:.1f} ms | RANSAC: {(t3 - t2) * 1000:.1f} ms | Render: {(t4 - t3) * 1000:.1f} ms"
            cv2.putText(img, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            if ransac_result is not None:
                hud2 = f"RANSAC inliers: {int(ransac_result['num_inliers'])}"
                cv2.putText(img, hud2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
            # Ayuda de teclas
            help1 = "W/S: pitch  A/D: yaw  Q/E: roll  =/-: zoom  J/L/I/K: pan  R: reset  [,]: voxel  [;'] minPts"
            cv2.putText(img, help1, (10, img.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1, cv2.LINE_AA)
            cv2.imshow('RANSAC PointCloud', img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            # Controles de navegación
            step_ang = 3.0
            step_pan = 0.05
            step_fov = 2.0
            if key in (ord('w'), ord('W')):
                pitch_deg += step_ang
            elif key in (ord('s'), ord('S')):
                pitch_deg -= step_ang
            elif key in (ord('a'), ord('A')):
                yaw_deg += step_ang
            elif key in (ord('d'), ord('D')):
                yaw_deg -= step_ang
            elif key in (ord('q'), ord('Q')):
                roll_deg -= step_ang
            elif key in (ord('e'), ord('E')):
                roll_deg += step_ang
            elif key in (ord('='), ord('+')):
                # zoom in (disminuir FOV o acercar tz)
                fov_deg = max(20.0, fov_deg - step_fov)
            elif key == ord('-'):
                # zoom out (aumentar FOV)
                fov_deg = min(100.0, fov_deg + step_fov)
            elif key in (ord('j'), ord('J')):
                tx -= step_pan
            elif key in (ord('l'), ord('L')):
                tx += step_pan
            elif key in (ord('i'), ord('I')):
                ty += step_pan
            elif key in (ord('k'), ord('K')):
                ty -= step_pan
            elif key in (ord('r'), ord('R')):
                yaw_deg, pitch_deg, roll_deg = -45.0, 25.0, 0.0
                tx, ty = 0.0, 0.0
                tz = None
                fov_deg = 60.0
            # Ajustes de voxel y min puntos rápidos
            elif key == ord('['):
                voxel_size = max(0.002, voxel_size - 0.002)
            elif key == ord(']'):
                voxel_size = min(0.05, voxel_size + 0.002)
            elif key == ord(';'):
                min_pts = max(1, min_pts - 1)
            elif key == ord('\''):
                min_pts = min(20, min_pts + 1)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()