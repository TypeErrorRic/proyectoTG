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
# Reutilizar objetos de RealSense para evitar costos por frame
_RS_PC = rs.pointcloud()
_RS_DEC = rs.decimation_filter()
_RS_DEC.set_option(rs.option.filter_magnitude, 2)

def extract_pointcloud(frames: rs.composite_frame) -> np.ndarray:
    """Extrae la nube de puntos (Nx3) del frame de RealSense con decimation.

    Devuelve NumPy (N,3) en metros, sólo Z>0.
    """
    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        return None
    # Decimation para reducir resolución del depth (acelera pointcloud + voxel)
    try:
        d2 = _RS_DEC.process(depth_frame)
        depth_frame = d2.as_depth_frame()
    except Exception:
        pass
    points = _RS_PC.calculate(depth_frame)
    verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    valid = verts[:, 2] > 0
    verts = verts[valid]
    return verts

def voxel_grid(points_xyz: np.ndarray, voxel_size: float = 0.01, min_points_per_voxel: int = 3) -> np.ndarray:
    """Filtro voxel en GPU (CuPy) con reducción por segmentos (rápido).

    - Cuantiza puntos a celdas de tamaño 'voxel_size'.
    - Ordena por clave de vóxel y usa reduceat para sumar por segmento.
    - Devuelve centros de vóxel con >= min_points_per_voxel.
    """
    if points_xyz is None or len(points_xyz) == 0:
        return points_xyz

    hash_factor = 100_000  # evitar colisiones; usar int64

    pts_gpu = cp.asarray(points_xyz, dtype=cp.float32)
    vox = cp.floor(pts_gpu / voxel_size).astype(cp.int64)
    voxel_hash = (vox[:, 0] + vox[:, 1] * hash_factor + vox[:, 2] * hash_factor * hash_factor).astype(cp.int64)

    # Ordenar por hash
    order = cp.argsort(voxel_hash)
    h_sorted = voxel_hash[order]
    pts_sorted = pts_gpu[order]

    # Inicios de segmento
    start_mask = cp.concatenate([cp.array([True]), h_sorted[1:] != h_sorted[:-1]])
    group_starts = cp.nonzero(start_mask)[0]

    # Sumas por segmento
    sums = cp.add.reduceat(pts_sorted, group_starts, axis=0)

    # Conteos por segmento
    group_ends = cp.concatenate([group_starts[1:], cp.array([pts_sorted.shape[0]])])
    counts = (group_ends - group_starts).astype(cp.int32)

    # Promedio y filtro
    means = sums / counts[:, None]
    keep_mask = counts >= int(max(1, min_points_per_voxel))
    means = means[keep_mask]
    return means

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
                      fov_deg: float = 60.0,
                      max_points: int = 150_000) -> np.ndarray:
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
    # Submuestreo para render si hay demasiados puntos
    N = int(pts.shape[0])
    if N > max_points:
        idx = cp.random.randint(0, N, size=(max_points,), dtype=cp.int32)
        pts = pts[idx]
    # Media en lugar de mediana (más rápido)
    center = cp.mean(pts, axis=0)
    pts_centered = pts - center
    R_cp = cp.asarray(R)
    pts_rot = pts_centered @ R_cp.T
    z = pts_rot[:, 2]
    if tz is None:
        # Aproximación rápida: usa min en vez de percentil (ahorra cómputo)
        tz = max(0.5, -float(cp.min(z)) + 1.5)
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
    try:
        cv2.namedWindow('RANSAC PointCloud', cv2.WINDOW_NORMAL)
        # Estado de cámara para navegación
        yaw_deg, pitch_deg, roll_deg = -45.0, 25.0, 0.0
        tx, ty = 0.0, 0.0
        tz = None  # auto al inicio
        fov_deg = 60.0
        voxel_size = 0.01
        min_pts = 3
        # Rendimiento / logging
        max_render_pts = 150_000
        frame_idx = 0
        last_log = time.perf_counter()
        log_interval = 1.0  # segundos
        while True:
            t0 = time.perf_counter()
            frames = pipeline.wait_for_frames()
            t1 = time.perf_counter()
            points_xyz = extract_pointcloud(frames)
            t2 = time.perf_counter()
            points_voxel = voxel_grid(points_xyz, voxel_size=voxel_size, min_points_per_voxel=min_pts) if points_xyz is not None else None
            t3 = time.perf_counter()
            frame_idx += 1
            img = render_pointcloud(points_voxel,
                                     out_size=(720, 720),
                                     yaw_deg=yaw_deg,
                                     pitch_deg=pitch_deg,
                                     roll_deg=roll_deg,
                                     tz=tz,
                                     tx=tx,
                                     ty=ty,
                                     fov_deg=fov_deg,
                                     max_points=max_render_pts)
            t4 = time.perf_counter()
            hud = f"Adquisición: {(t1 - t0) * 1000:.1f} ms | Extract: {(t2 - t1) * 1000:.1f} ms | Voxel: {(t3 - t2) * 1000:.1f} ms | Render: {(t4 - t3) * 1000:.1f} ms"
            cv2.putText(img, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            # Log a consola (rate-limited)
            now = time.perf_counter()
            if (now - last_log) >= log_interval:
                total_ms = (t4 - t0) * 1000.0
                fps = 1000.0 / max(total_ms, 1e-3)
                n_voxel = int(points_voxel.shape[0]) if points_voxel is not None else 0
                n_extract = int(points_xyz.shape[0]) if points_xyz is not None else 0
                print(
                    f"FPS: {fps:4.1f} | Acq: {(t1 - t0) * 1000:.1f} ms | Extract: {(t2 - t1) * 1000:.1f} ms | Voxel: {(t3 - t2) * 1000:.1f} ms | Render: {(t4 - t3) * 1000:.1f} ms | pts(raw): {n_extract} | pts(voxel): {n_voxel} | voxel: {voxel_size:.3f} | minPts: {min_pts} | maxPts: {max_render_pts}",
                    flush=True,
                )
                last_log = now
            # Ayuda de teclas
            help1 = "W/S: pitch  A/D: yaw  Q/E: roll  =/-: zoom  J/L/I/K: pan  R: reset  [,]: voxel  [;'] minPts  9/0: render pts"
            cv2.putText(img, help1, (10, img.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1, cv2.LINE_AA)
            cv2.imshow('RANSAC PointCloud', img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            # Controles de navegación
            step_ang = 3.0
            step_pan = 0.05
            step_fov = 2.0
            # Estado persistente adicional
            if 'max_render_pts' not in locals():
                max_render_pts = 150_000
            if 'frame_idx' not in locals():
                frame_idx = 0
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
            elif key == ord('9'):
                max_render_pts = max(20_000, int(max_render_pts * 0.8))
            elif key == ord('0'):
                max_render_pts = min(1_000_000, int(max_render_pts * 1.25))
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()