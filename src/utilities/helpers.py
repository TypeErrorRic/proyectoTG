"""
Helper utilities for point clouds, GPU overlays, and dataset loading.

Used by viewCamera, ransacCellingGround, and segmentar to prepare geometry,
apply masks, and stream sample data.
"""
import os
import numpy as np
import cv2
from typing import Optional, Tuple


"""
Helper utilities for point clouds, GPU overlays, and dataset loading.

Used by viewCamera, ransacCellingGround, and segmentar to prepare geometry,
apply masks, and stream sample data.
"""

# ==================================================
#  Point cloud utilities for viewCamera.py
# ==================================================


def _rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """
    Build rotation matrix R = Rz(roll) * Ry(yaw) * Rx(pitch) in degrees.
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
    Build 3D points (N,3) in meters from rays (H,W,3) and depth (H,W).
    Applies subsampling by 'stride' to reduce density.
    """
    if rays is None or depth_m is None:
        return np.empty((0, 3), dtype=np.float32)
    
    H, W = depth_m.shape[:2]
    if rays.shape[:2] != (H, W):
        # If shapes differ, resize depth_m to match rays
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
    Render a point cloud as a 2D projection (CPU, NumPy).
    - points_xyz: (N,3) in meters
    - colors_bgr: optional (N,3) uint8; if None uses gray
    """
    H, W = out_size
    img = np.zeros((H, W, 3), dtype=np.uint8)
    if points_xyz is None or len(points_xyz) == 0:
        return img

    pts = points_xyz.astype(np.float32)
    # Center for stable visualization
    center = np.median(pts, axis=0)
    pts_c = pts - center
    R = _rotation_matrix(yaw_deg, pitch_deg, roll_deg).astype(np.float32)
    pts_r = pts_c @ R.T

    # Offset in Z to ensure z > 0
    z = pts_r[:, 2]
    z_min = np.percentile(z, 5) if z.size > 0 else 0.0
    tz = max(0.5, -float(z_min) + 1.5)
    pts_cam = pts_r + np.array([0.0, 0.0, tz], dtype=np.float32)

    # Perspective projection
    f = 0.5 * H / np.tan(np.deg2rad(fov_deg) * 0.5)
    Z = np.clip(pts_cam[:, 2], 1e-3, None)
    u = (W * 0.5 + (pts_cam[:, 0] * f) / Z).astype(np.int32)
    v = (H * 0.5 - (pts_cam[:, 1] * f) / Z).astype(np.int32)

    # Clip to viewport
    mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(mask):
        return img
    u, v = u[mask], v[mask]
    if colors_bgr is not None and len(colors_bgr) == len(points_xyz):
        col = colors_bgr[mask]
    else:
        col = None

    # Subsample to avoid saturation
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

# Mask overlay toggles (shared with GUI controls)
MASK_SHOW_GROUND = True
MASK_SHOW_WALL = True
MASK_SHOW_DOOR = True


def set_mask_visibility(
    ground: Optional[bool] = None,
    wall: Optional[bool] = None,
    door: Optional[bool] = None,
) -> Tuple[bool, bool, bool]:
    """
    Update mask overlay visibility flags and return the current states.
    """
    global MASK_SHOW_GROUND, MASK_SHOW_WALL, MASK_SHOW_DOOR
    if ground is not None:
        MASK_SHOW_GROUND = bool(ground)
    if wall is not None:
        MASK_SHOW_WALL = bool(wall)
    if door is not None:
        MASK_SHOW_DOOR = bool(door)
    return MASK_SHOW_GROUND, MASK_SHOW_WALL, MASK_SHOW_DOOR


def toggle_mask_visibility(name: str) -> bool:
    """
    Toggle one mask overlay flag by name and return the new state.
    """
    global MASK_SHOW_GROUND, MASK_SHOW_WALL, MASK_SHOW_DOOR
    key = name.strip().lower()
    if key in ("ground", "suelo"):
        MASK_SHOW_GROUND = not MASK_SHOW_GROUND
        return MASK_SHOW_GROUND
    if key in ("wall", "muro"):
        MASK_SHOW_WALL = not MASK_SHOW_WALL
        return MASK_SHOW_WALL
    if key in ("door", "puerta"):
        MASK_SHOW_DOOR = not MASK_SHOW_DOOR
        return MASK_SHOW_DOOR
    raise ValueError(f"Nombre de mascara desconocido: {name}")


# No additional helpers at the moment

def apply_mask_to_rgb(
    rgb_image: np.ndarray,
    ground_mask: np.ndarray,
    wall_mask: Optional[np.ndarray] = None,
    door_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Paint floor, wall, and door regions with different colors over the RGB image.

    Args:
        rgb_image: RGB image (H, W, 3)
        ground_mask: Floor mask (H, W) - painted in GREEN
        wall_mask: Wall mask (H, W) - painted in BLUE (optional)
        door_mask: Door mask (H, W) - painted in RED (optional)

    Returns:
        RGB image with colored masks overlaid
    """
    if rgb_image is None:
        return None
    if ground_mask is None:
        ground_mask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)

    def _prepare_mask(mask, target_shape):
        """Normalize, resize, and threshold a mask using GPU"""
        if mask is None:
            return None

        mask = np.asarray(mask)
        mask_h, mask_w = mask.shape[:2]

        # Upload to GPU
        mask_gpu = cv2.cuda_GpuMat()
        mask_gpu.upload(mask)

        # Convert to grayscale if needed
        if mask.ndim == 3 and mask.shape[-1] == 3:
            mask_gpu = cv2.cuda.cvtColor(mask_gpu, cv2.COLOR_BGR2GRAY)

        # Resize if needed
        if (mask_h, mask_w) != target_shape:
            mask_gpu = cv2.cuda.resize(
                mask_gpu,
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Binary threshold
        _, mask_gpu = cv2.cuda.threshold(mask_gpu, 0, 255, cv2.THRESH_BINARY)
        return mask_gpu

    # Prepare all masks
    target_shape = (rgb_image.shape[0], rgb_image.shape[1])
    ground_gpu = _prepare_mask(ground_mask, target_shape)
    wall_gpu = _prepare_mask(wall_mask, target_shape) if wall_mask is not None else None
    door_gpu = _prepare_mask(door_mask, target_shape) if door_mask is not None else None

    # Respect visibility toggles from GUI.
    if not MASK_SHOW_GROUND:
        ground_gpu = None
    if not MASK_SHOW_WALL:
        wall_gpu = None
    if not MASK_SHOW_DOOR:
        door_gpu = None

    # Upload RGB to GPU
    rgb_gpu = cv2.cuda_GpuMat()
    rgb_gpu.upload(rgb_image)

    # Create color overlays
    def _create_color_overlay(color_bgr):
        """Create a solid color image matching RGB size"""
        color_cpu = np.zeros_like(rgb_image, dtype=rgb_image.dtype)
        color_cpu[:, :, 0] = color_bgr[0]  # B
        color_cpu[:, :, 1] = color_bgr[1]  # G
        color_cpu[:, :, 2] = color_bgr[2]  # R
        color_gpu = cv2.cuda_GpuMat()
        color_gpu.upload(color_cpu)
        return color_gpu

    # Green for ground, Blue for wall, Red for door
    green_gpu = _create_color_overlay((0, 255, 0))    # Green (BGR)
    blue_gpu = _create_color_overlay((255, 0, 0))     # Blue (BGR)
    red_gpu = _create_color_overlay((0, 0, 255))      # Red (BGR)

    # Start with the original RGB
    result_gpu = rgb_gpu.clone()

    # Apply masks in order: ground -> wall -> door (door has priority)
    def _apply_overlay(base_gpu, overlay_gpu, mask_gpu):
        """Apply colored overlay using mask"""
        if mask_gpu is None:
            return base_gpu

        # Convert mask to 3 channels
        mask3_gpu = cv2.cuda.cvtColor(mask_gpu, cv2.COLOR_GRAY2BGR)
        mask_inv_gpu = cv2.cuda.bitwise_not(mask_gpu)
        mask_inv3_gpu = cv2.cuda.cvtColor(mask_inv_gpu, cv2.COLOR_GRAY2BGR)

        # Apply overlay: fg = overlay & mask, bg = base & ~mask
        fg_gpu = cv2.cuda.bitwise_and(overlay_gpu, mask3_gpu)
        bg_gpu = cv2.cuda.bitwise_and(base_gpu, mask_inv3_gpu)
        return cv2.cuda.bitwise_or(bg_gpu, fg_gpu)

    # Apply ground (green)
    result_gpu = _apply_overlay(result_gpu, green_gpu, ground_gpu)

    # Apply wall (blue)
    if wall_gpu is not None:
        result_gpu = _apply_overlay(result_gpu, blue_gpu, wall_gpu)

    # Apply door (red) - has highest priority
    if door_gpu is not None:
        result_gpu = _apply_overlay(result_gpu, red_gpu, door_gpu)

    return result_gpu.download()

_DATASET_IMAGE_FILES = None
_DATASET_INDEX = 0


def load_dataset_frame(index: Optional[int] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load an RGB + depth pair from src/data/{images,depths}.

    Assumes depth in PNG uint16 (mm) and converts to meters (float32).
    If 'index' is provided, loads that specific item (0-based, wraps modulo dataset size).
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
                print(f"[helpers] Images folder not found: {images_dir}")
                return None, None
            _DATASET_IMAGE_FILES = sorted(
                f
                for f in os.listdir(images_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            )
            _DATASET_INDEX = 0

        if not _DATASET_IMAGE_FILES:
            print(f"[helpers] No images found in {images_dir}")
            return None, None

        if index is not None:
            try:
                idx = int(index)
            except Exception:
                print(f"[helpers] Invalid dataset index: {index}")
                return None, None
            _DATASET_INDEX = idx % len(_DATASET_IMAGE_FILES)
        elif _DATASET_INDEX >= len(_DATASET_IMAGE_FILES):
            _DATASET_INDEX = 0

        filename = _DATASET_IMAGE_FILES[_DATASET_INDEX]
        _DATASET_INDEX = (_DATASET_INDEX + 1) % len(_DATASET_IMAGE_FILES)

        image_path = os.path.join(images_dir, filename)
        depth_path = os.path.join(depths_dir, filename)

        if not os.path.exists(depth_path):
            print(
                f"[helpers] Depth map not found for {filename} in {depths_dir}"
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

        # Handle different depth formats
        if depth_raw.dtype == np.uint16:
            # RealSense format: depth in mm, convert to meters
            mapa_profundidad = depth_raw.astype(np.float32) / 1000.0
        elif depth_raw.dtype == np.uint8:
            # NYU dataset format: normalized depth [0, 255] → keep as-is
            # doorDetection will handle the normalization
            mapa_profundidad = depth_raw.astype(np.float32)
        else:
            # Already float32, assume in meters
            mapa_profundidad = depth_raw.astype(np.float32)

        return imagen_rgb, mapa_profundidad
    except Exception as exc:
        print(f"[helpers] Error cargando datos desde src/data: {exc}")
        return None, None

