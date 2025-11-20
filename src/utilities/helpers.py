import os
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

#Por el momento no hay mas funciones

def apply_mask_to_rgb(rgb_image: np.ndarray, ground_mask: np.ndarray) -> np.ndarray:
    """
    Marca en verde la zona de suelo sobre la imagen.
    """
    if rgb_image is None:
        return None
    if ground_mask is None:
        ground_mask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)

    # Normalizar mask a 2D
    if ground_mask.ndim == 3 and ground_mask.shape[-1] == 3:
        ground_mask = cv2.cvtColor(ground_mask, cv2.COLOR_BGR2GRAY)
    if ground_mask.shape[:2] != rgb_image.shape[:2]:
        ground_mask = cv2.resize(
            ground_mask,
            (rgb_image.shape[1], rgb_image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    mask = ground_mask > 0
    result = rgb_image.copy()
    result[mask] = (0, 255, 0)
    return result

_DATASET_IMAGE_FILES = None
_DATASET_INDEX = 0


def load_dataset_frame() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Carga un par RGB + depth desde src/data/{images,depths}.

    Asume depth en PNG uint16 (mm) y lo convierte a metros (float32).
    """
    global _DATASET_IMAGE_FILES, _DATASET_INDEX
    try:
        base_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
        )
        images_dir = os.path.join(base_dir, "images")
        depths_dir = os.path.join(base_dir, "depths")

        if _DATASET_IMAGE_FILES is None:
            if not os.path.isdir(images_dir):
                print(f"[helpers] Carpeta de imágenes no encontrada: {images_dir}")
                return None, None
            _DATASET_IMAGE_FILES = sorted(
                f
                for f in os.listdir(images_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            )
            _DATASET_INDEX = 0

        if not _DATASET_IMAGE_FILES:
            print(f"[helpers] No se encontraron imágenes en {images_dir}")
            return None, None

        if _DATASET_INDEX >= len(_DATASET_IMAGE_FILES):
            _DATASET_INDEX = 0
        filename = _DATASET_IMAGE_FILES[_DATASET_INDEX]
        _DATASET_INDEX += 1

        image_path = os.path.join(images_dir, filename)
        depth_path = os.path.join(depths_dir, filename)

        if not os.path.exists(depth_path):
            print(
                f"[helpers] No se encontró mapa de profundidad para "
                f"{filename} en {depths_dir}"
            )
            return None, None

        imagen_rgb = cv2.imread(image_path, cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if imagen_rgb is None or depth_raw is None:
            print(
                f"[helpers] Error al leer archivos:\n"
                f"  RGB: {image_path}\n"
                f"  Depth: {depth_path}"
            )
            return None, None

        if depth_raw.dtype == np.uint16:
            mapa_profundidad = depth_raw.astype(np.float32) / 1000.0
        else:
            mapa_profundidad = depth_raw.astype(np.float32)

        return imagen_rgb, mapa_profundidad
    except Exception as exc:
        print(f"[helpers] Error cargando datos desde src/data: {exc}")
        return None, None


