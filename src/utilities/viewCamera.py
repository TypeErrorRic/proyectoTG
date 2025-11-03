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

# Reutilizar objetos de RealSense para evitar costos por frame
_RS_PC = rs.pointcloud()
_RS_DEC = rs.decimation_filter()
_RS_DEC.set_option(rs.option.filter_magnitude, 3)  # 2=default, 3-4=más FPS, menos resolución

# Cache de parámetros para extracción acelerada
_DEPTH_SCALE = None  # metros por unidad en z16
_PIXGRID_CACHE = {}  # clave: (W,H) -> (Ugrid, Vgrid) en GPU

def _ensure_depth_scale() -> float:
    """Devuelve el depth_scale cacheado si existe; si no, retorna 0.001 por defecto.

    Importante: no intenta inicializar la cámara para evitar conflictos con pipelines existentes.
    """
    return _DEPTH_SCALE if _DEPTH_SCALE is not None else 0.001

def _update_depth_scale_from_profile(video_profile: rs.video_stream_profile):
    """Actualiza el depth_scale global a partir de un perfil de stream de profundidad."""
    global _DEPTH_SCALE
    try:
        depth_sensor = video_profile.get_device().first_depth_sensor()
        _DEPTH_SCALE = float(depth_sensor.get_depth_scale())
    except Exception:
        # Mantener valor actual o default si falla
        if _DEPTH_SCALE is None:
            _DEPTH_SCALE = 0.001

def _gpu_pixel_grids(width: int, height: int):
    """Devuelve mallas U,V en GPU cacheadas para (W,H)."""
    key = (width, height)
    grids = _PIXGRID_CACHE.get(key)
    if grids is None:
        u = cp.arange(width, dtype=cp.float32)
        v = cp.arange(height, dtype=cp.float32)
        U, V = cp.meshgrid(u, v)  # shape (H,W)
        _PIXGRID_CACHE[key] = (U, V)
        grids = (U, V)
    return grids

def extract_pointcloud_gpu(frames: rs.composite_frame,
                           stride: int = 1,
                           skip_top_ratio: float = 0.25,
                           max_distance_m: float = 3.5) -> cp.ndarray:
    """Extrae nube de puntos (Nx3) en GPU a partir del depth (z16) con intrínsecos.

    - Usa decimation (ya configurado globalmente) sobre el depth_frame.
    - Vectoriza la deproyección en CuPy y aplica muestreo por 'stride' previo.
    - skip_top_ratio: fracción superior de la imagen a ignorar (típicamente techo/cielo).
    - max_distance_m: filtra por distancia de profundidad (Z) en metros (<= max_distance_m).
    - Retorna un arreglo CuPy (N,3) en metros con Z>0.
    """
    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        return None

    # Aplicar decimation (reduce HxW y acelera la deproyección)
    try:
        d2 = _RS_DEC.process(depth_frame)
        depth_frame = d2.as_depth_frame()
    except Exception:
        pass

    # Intrínsecos del frame depth actual
    prof = depth_frame.get_profile().as_video_stream_profile()
    intr = prof.get_intrinsics()
    # Asegurar depth_scale desde el perfil actual (sin iniciar nuevos pipelines)
    _update_depth_scale_from_profile(prof)
    W, H = intr.width, intr.height
    fx, fy = float(intr.fx), float(intr.fy)
    ppx, ppy = float(intr.ppx), float(intr.ppy)

    # Profundidad a GPU (en metros) - optimizado: conversión directa a float32
    depth_np = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)
    depth_scale = cp.float32(_ensure_depth_scale())
    
    # ROI vertical: ignorar la parte superior para evitar procesar techo/cielo
    row_start = int(H * skip_top_ratio)
    if row_start > 0:
        depth_np = depth_np[row_start:, :]
        H = depth_np.shape[0]
        # Ajustar principal point en Y por el desplazamiento del ROI
        ppy = ppy - row_start
    
    # Muestreo por stride previo (opcional) para acelerar
    s = max(1, int(stride))
    if s > 1:
        depth_np = depth_np[::s, ::s]
        # Ajustar intrínsecos al muestreo
        fx /= s
        fy /= s
        ppx /= s
        ppy /= s
        W = int(np.ceil(W / s))
        H = int(np.ceil(H / s))

    # Transferencia a GPU y conversión a metros (fusionado)
    depth_m = cp.asarray(depth_np, dtype=cp.float32) * depth_scale

    # Grillas de pixeles en GPU (H,W) - cacheadas
    U, V = _gpu_pixel_grids(W, H)

    # Deproyección vectorizada fusionada (menos operaciones intermedias)
    if max_distance_m is not None and max_distance_m > 0:
        mask = (depth_m > 0) & (depth_m <= cp.float32(max_distance_m))
    else:
        mask = depth_m > 0
    if not cp.any(mask):
        return cp.empty((0, 3), dtype=cp.float32)
    
    # Pre-calcular inversos para multiplicación (más rápido que división)
    inv_fx = cp.float32(1.0 / fx)
    inv_fy = cp.float32(1.0 / fy)
    
    u = U[mask]
    v = V[mask]
    z = depth_m[mask]
    x = (u - ppx) * z * inv_fx
    y = (v - ppy) * z * inv_fy
    pts = cp.stack((x, y, z), axis=1)
    return pts

def voxel_grid(points_xyz: np.ndarray,
               voxel_size: float = 0.012,
               min_points_per_voxel: int = 1) -> np.ndarray:
    """Voxel super-rápido en GPU para denoise y preservación de planos.

    - Usa hashing de vóxeles y cp.unique para elegir 1 representante por vóxel.
    - Filtra vóxeles con menos de `min_points_per_voxel` puntos (ruido aislado).
    - Si voxel_size<=0 o None, devuelve los puntos tal cual (como CuPy).
    """
    if points_xyz is None:
        return None
    if len(points_xyz) == 0:
        return points_xyz
    if voxel_size is None or voxel_size <= 0:
        return cp.asarray(points_xyz, dtype=cp.float32)

    pts_gpu = cp.asarray(points_xyz, dtype=cp.float32)
    min_pts = int(max(1, min_points_per_voxel))

    # Hash de vóxeles (optimizado: operaciones fusionadas, int32 en vez de int64 para velocidad)
    inv_voxel = cp.float32(1.0 / voxel_size)
    vox = cp.floor(pts_gpu * inv_voxel).astype(cp.int32)
    
    # Hash con factor más pequeño (suficiente para escenas típicas, más rápido)
    hash_factor = cp.int32(73856)  # primo grande, evita colisiones
    voxel_hash = vox[:, 0] + vox[:, 1] * hash_factor + vox[:, 2] * (hash_factor * hash_factor)

    # Un representante por vóxel + conteos; filtra por soporte
    _, idx, counts = cp.unique(voxel_hash, return_index=True, return_counts=True)
    if min_pts > 1:
        keep = counts >= min_pts
        idx = idx[keep]
    return pts_gpu[idx]

def get_voxel_for_ransac(frames: Optional[rs.composite_frame] = None,
                         voxel_size: float = 0.012,
                         min_points_per_voxel: int = 1,
                         pipeline: Optional[rs.pipeline] = None,
                         extract_stride: int = 1,
                         skip_top_ratio: float = 0.25):
    """Extrae la nube de puntos, aplica filtro voxel y la retorna lista para RANSAC.

    - Si no se proveen frames pero hay pipeline, toma un frame de ese pipeline.
    - Si no hay ni frames ni pipeline, retorna None (no se inicializa cámara aquí).
    - Usa la ruta GPU para extracción y voxelizado.
    - skip_top_ratio: fracción superior de la imagen a ignorar (techo/cielo).
    """
    if frames is None:
        if pipeline is not None:
            frames = pipeline.wait_for_frames()
        else:
            return None

    points_xyz = extract_pointcloud_gpu(frames, stride=extract_stride, skip_top_ratio=skip_top_ratio)
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
        extract_stride = 1  # muestreo previo a la deproyección (1=full)
        skip_top_ratio = 0.25  # ignorar fracción superior de la imagen (techo/cielo)
        max_distance_m = 3.5   # limitar distancia de profundidad para acelerar
        # Parámetros de voxel establecidos
        voxel_size = 0.012
        min_pts = 1
        # Rendimiento / logging
        max_render_pts = 150_000
        frame_idx = 0
        last_log = time.perf_counter()
        log_interval = 1.0  # segundos
        # Limpieza de memoria GPU periódica (cada N frames)
        gpu_cleanup_interval = 300  # frames (~10 seg a 30 FPS)
        while True:
            t0 = time.perf_counter()
            frames = pipeline.wait_for_frames()
            t1 = time.perf_counter()
            points_xyz = extract_pointcloud_gpu(frames,
                                               stride=extract_stride,
                                               skip_top_ratio=skip_top_ratio,
                                               max_distance_m=max_distance_m)
            t2 = time.perf_counter()
            # Filtro voxel para reducir densidad manteniendo planos
            points_voxel = voxel_grid(points_xyz, voxel_size=voxel_size, min_points_per_voxel=min_pts) if points_xyz is not None else None
            t3 = time.perf_counter()
            frame_idx += 1
            
            # Limpieza periódica del pool de memoria de CuPy
            if frame_idx % gpu_cleanup_interval == 0:
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
            
            img = render_pointcloud(points_voxel if points_voxel is not None else points_xyz,
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
            hud = (
                f"Adquisición: {(t1 - t0) * 1000:.1f} ms | "
                f"Extract(GPU): {(t2 - t1) * 1000:.1f} ms | "
                f"Voxel: {(t3 - t2) * 1000:.1f} ms | "
                f"Render: {(t4 - t3) * 1000:.1f} ms | "
                f"stride: {extract_stride} | ROI: {skip_top_ratio:.2f} | maxD: {max_distance_m:.1f}m | vx:{voxel_size:.3f} | min:{min_pts}"
            )
            cv2.putText(img, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            # Log a consola (rate-limited)
            now = time.perf_counter()
            if (now - last_log) >= log_interval:
                total_ms = (t4 - t0) * 1000.0
                fps = 1000.0 / max(total_ms, 1e-3)
                n_extract = int(points_xyz.shape[0]) if points_xyz is not None else 0
                n_voxel = int(points_voxel.shape[0]) if points_voxel is not None else n_extract
                print(
                    f"FPS: {fps:4.1f} | Acq: {(t1 - t0) * 1000:.1f} ms | Extract(GPU): {(t2 - t1) * 1000:.1f} ms | Voxel: {(t3 - t2) * 1000:.1f} ms | Render: {(t4 - t3) * 1000:.1f} ms | pts(raw): {n_extract} | pts(voxel): {n_voxel} | maxPts: {max_render_pts} | stride: {extract_stride}",
                    flush=True,
                )
                last_log = now
            # Ayuda de teclas
            help1 = "W/S: pitch  A/D: yaw  Q/E: roll  =/-: zoom  J/L/I/K: pan  R: reset  9/0: render pts  O/P: stride  -/=: ROI"
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
            elif key == ord('9'):
                max_render_pts = max(20_000, int(max_render_pts * 0.8))
            elif key == ord('0'):
                max_render_pts = min(1_000_000, int(max_render_pts * 1.25))
            elif key in (ord('o'), ord('O')):  # aumentar stride (más rápido, menos puntos)
                extract_stride = min(8, extract_stride + 1)
            elif key in (ord('p'), ord('P')):  # disminuir stride (más denso)
                extract_stride = max(1, extract_stride - 1)
            elif key == ord('-'):  # disminuir ROI (procesar más de arriba)
                skip_top_ratio = max(0.0, skip_top_ratio - 0.05)
            elif key == ord('='):  # aumentar ROI (ignorar más de arriba)
                skip_top_ratio = min(0.5, skip_top_ratio + 0.05)
            # (Se elimina cambio de modo: sólo 'first' rápido)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()