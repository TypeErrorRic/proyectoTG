import numpy as np
import cv2
from typing import Optional, Tuple


# ==================================================
#  Utilidades de nubes para viewCamera.py
# ==================================================


def _rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """
    Construye una matriz de rotación R = Rz(roll) * Ry(yaw) * Rx(pitch) usando grados.
    """
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch),  np.cos(pitch)]], dtype=np.float32)
    Ry = np.array([[ np.cos(yaw), 0, np.sin(yaw)],
                   [ 0,          1, 0         ],
                   [-np.sin(yaw), 0, np.cos(yaw)]], dtype=np.float32)
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll),  np.cos(roll), 0],
                   [0,            0,             1]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


def points_from_rays_and_depth(rays: np.ndarray,
                               depth_m: np.ndarray,
                               stride: int = 4) -> np.ndarray:
    """
    Construye puntos 3D (N,3) en metros a partir de rayos (H,W,3) y profundidad (H,W).
    Aplica un submuestreo por 'stride' para reducir densidad.
    """
    if rays is None or depth_m is None:
        return np.empty((0, 3), dtype=np.float32)
    
    H, W = depth_m.shape[:2]
    if rays.shape[:2] != (H, W):
        # Si las dimensiones no coinciden, redimensionar depth_m
        depth_m = cv2.resize(depth_m, (rays.shape[1], rays.shape[0]), interpolation=cv2.INTER_NEAREST)
        H, W = rays.shape[:2]

    Dsub = depth_m[::stride, ::stride]
    Rsub = rays[::stride, ::stride]
    valid = (Dsub > 0)

    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    P = (Rsub * Dsub[..., None]).astype(np.float32)
    P = P[valid]
    return P


def render_pointcloud_numpy(points_xyz: np.ndarray,
                            colors_bgr: Optional[np.ndarray] = None,
                            out_size=(720, 720),
                            yaw_deg: float = -45.0,
                            pitch_deg: float = 25.0,
                            roll_deg: float = 0.0,
                            fov_deg: float = 60.0,
                            point_size: int = 1) -> np.ndarray:
    """
    Renderiza una nube de puntos como proyección en 2D (CPU, NumPy).
    - points_xyz: (N,3) en metros
    - colors_bgr: (N,3) uint8 opcional; si None usa gris
    """
    H, W = out_size
    img = np.zeros((H, W, 3), dtype=np.uint8)
    if points_xyz is None or len(points_xyz) == 0:
        return img

    pts = points_xyz.astype(np.float32)
    # Centrar para visualización estable
    center = np.median(pts, axis=0)
    pts_c = pts - center
    R = _rotation_matrix(yaw_deg, pitch_deg, roll_deg).astype(np.float32)
    pts_r = pts_c @ R.T

    # Desplazar en Z para asegurar z>0
    z = pts_r[:, 2]
    z_min = np.percentile(z, 5) if z.size > 0 else 0.0
    tz = max(0.5, -float(z_min) + 1.5)
    pts_cam = pts_r + np.array([0.0, 0.0, tz], dtype=np.float32)

    # Proyección perspectiva
    f = 0.5 * H / np.tan(np.deg2rad(fov_deg) * 0.5)
    Z = np.clip(pts_cam[:, 2], 1e-3, None)
    u = (W * 0.5 + (pts_cam[:, 0] * f) / Z).astype(np.int32)
    v = (H * 0.5 - (pts_cam[:, 1] * f) / Z).astype(np.int32)

    # Recortar a pantalla
    mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(mask):
        return img
    u, v = u[mask], v[mask]
    if colors_bgr is not None and len(colors_bgr) == len(points_xyz):
        col = colors_bgr[mask]
    else:
        col = None

    # Muestreo para evitar saturación
    max_pts = 150_000
    if u.size > max_pts:
        step = int(np.ceil(u.size / max_pts))
        u, v = u[::step], v[::step]
        if col is not None:
            col = col[::step]

    if point_size <= 1:
        if col is None:
            img[v, u] = (200, 200, 200)
        else:
            img[v, u] = col.astype(np.uint8)
    else:
        for i in range(u.size):
            c = (int(col[i, 0]), int(col[i, 1]), int(col[i, 2])) if col is not None else (200, 200, 200)
            cv2.circle(img, (int(u[i]), int(v[i])), point_size, c, -1, lineType=cv2.LINE_AA)
    return img