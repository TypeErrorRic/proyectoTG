"""
Visualización en vivo de la cámara Intel RealSense (solo color o color + profundidad)
Compatible con Jetson Nano, Ubuntu y Windows.

Utilidades añadidas:
- get_depth_scale(pipeline=None): lee el scale del sensor de profundidad
- extract_rgb(frames): devuelve imagen BGR (np.uint8)
- extract_depth_raw(frames): devuelve depth en uint16 (unidades nativas)
- extract_depth_meters(frames, depth_scale=None): depth en metros (float32)
- extract_pointcloud(frames, with_colors=True, filter_invalid=True, organized=False):
    -> nube de puntos Nx3 (y opcional Nx3 de colores) o en forma HxWx3 si organized=True
- depth_to_colormap(depth_m, max_m=4.0): convierte profundidad en metros a una imagen BGR
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import cupy as cp

# =========================================================
# ===============  U T I L I D A D E S  ===================
# =========================================================

_DEPTH_SCALE_CACHE = None

def get_depth_scale(pipeline: rs.pipeline = None) -> float:
    """
    Obtiene y cachea el factor 'depth_scale' del sensor de profundidad.
    Si no se pasa 'pipeline', intenta leer el primer dispositivo conectado.
    Devuelve 0.001 (1 mm) como último recurso.
    """
    global _DEPTH_SCALE_CACHE
    if _DEPTH_SCALE_CACHE is not None:
        return _DEPTH_SCALE_CACHE

    try:
        if pipeline is not None:
            dev = pipeline.get_active_profile().get_device()
            depth_sensor = dev.first_depth_sensor()
            _DEPTH_SCALE_CACHE = float(depth_sensor.get_depth_scale())
            return _DEPTH_SCALE_CACHE
    except Exception:
        pass

    # Fallback: intenta vía contexto global
    try:
        ctx = rs.context()
        devs = ctx.query_devices()
        if len(devs) > 0:
            depth_sensor = devs[0].first_depth_sensor()
            _DEPTH_SCALE_CACHE = float(depth_sensor.get_depth_scale())
            return _DEPTH_SCALE_CACHE
    except Exception:
        pass

    # Último recurso (común en D435/D415): 1 mm
    _DEPTH_SCALE_CACHE = 0.001
    return _DEPTH_SCALE_CACHE


def extract_rgb(frames: rs.composite_frame, copy: bool = False) -> np.ndarray:
    """
    Devuelve la imagen de color (BGR, uint8) como np.ndarray de forma (H, W, 3).
    Retorna None si no hay frame de color.
    """
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    img = np.asanyarray(color_frame.get_data())
    return img.copy() if copy else img


def extract_depth_raw(frames: rs.composite_frame) -> np.ndarray:
    """
    Devuelve el mapa de profundidad en bruto (uint16) como np.ndarray (H, W).
    Estas unidades deben multiplicarse por 'depth_scale' para obtener metros.
    Retorna None si no hay frame de profundidad.
    """
    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        return None
    depth_raw = np.asanyarray(depth_frame.get_data())
    return depth_raw


def extract_depth_meters(frames: rs.composite_frame, depth_scale: float = None) -> np.ndarray:
    """
    Devuelve el mapa de profundidad en metros (float32) como np.ndarray (H, W).
    Si no se entrega 'depth_scale', se intentará consultarlo (y cachearlo).
    """
    depth_raw = extract_depth_raw(frames)
    if depth_raw is None:
        return None
    if depth_scale is None:
        depth_scale = get_depth_scale()
    depth_m = depth_raw.astype(np.float32) * float(depth_scale)
    return depth_m


def _map_texture_to_colors(color_image: np.ndarray, texcoords: np.ndarray) -> np.ndarray:
    """
    Convierte coordenadas de textura (u,v) en colores BGR tomando muestras de 'color_image'.
    - color_image: np.uint8 (H, W, 3) en BGR
    - texcoords: float32 (N, 2) en [0,1]
    Retorna np.uint8 (N, 3) con el color por punto. Los 'texcoords' fuera de rango se clipean.
    """
    H, W = color_image.shape[:2]
    u = np.clip((texcoords[:, 0] * W).astype(np.int32), 0, W - 1)
    v = np.clip((texcoords[:, 1] * H).astype(np.int32), 0, H - 1)
    return color_image[v, u, :]


def extract_pointcloud(
    frames: rs.composite_frame,
    with_colors: bool = True,
    filter_invalid: bool = True,
    organized: bool = False,
):
    """
    Genera una nube de puntos a partir de los frames actuales.

    Parámetros:
      - with_colors: si True, intenta mapear colores desde el frame RGB.
      - filter_invalid: si True, filtra puntos con z <= 0 (inválidos).
      - organized: si True, devuelve puntos (y colores) en forma (H, W, 3)
                   en lugar de (N, 3) plano.

    Retorna:
      - Si organized == False:
          points_xyz: np.float32 (N, 3)
          colors_bgr (opcional): np.uint8 (N, 3) o None
      - Si organized == True:
          points_xyz_img: np.float32 (H, W, 3)
          colors_bgr_img (opcional): np.uint8 (H, W, 3) o None
    """
    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        return (None, None) if not organized else (None, None)

    color_frame = frames.get_color_frame() if with_colors else None
    color_image = np.asanyarray(color_frame.get_data()) if color_frame else None

    pc = rs.pointcloud()
    if color_frame is not None:
        pc.map_to(color_frame)

    points = pc.calculate(depth_frame)

    # Vértices (x, y, z)
    # Nota: get_vertices() devuelve un array estructurado; no castear directamente con dtype
    verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

    # Colores desde texcoords si se pidió
    colors = None
    if with_colors and color_image is not None:
        texcoords = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
        colors = _map_texture_to_colors(color_image, texcoords)

    # Filtrado de puntos inválidos
    if filter_invalid:
        valid = verts[:, 2] > 0
        verts = verts[valid]
        if colors is not None:
            colors = colors[valid]

    if organized:
        vsprof = depth_frame.get_profile().as_video_stream_profile()
        W, H = vsprof.width(), vsprof.height()
        if filter_invalid:
            verts_full = np.asanyarray(points.get_vertices()).view(np.float32).reshape(H, W, 3)
            if with_colors and color_image is not None:
                tex_full = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(H*W, 2)
                colors_full = _map_texture_to_colors(color_image, tex_full).reshape(H, W, 3)
            else:
                colors_full = None
            return verts_full, colors_full
        else:
            verts_img = verts.reshape(H, W, 3)
            colors_img = colors.reshape(H, W, 3) if colors is not None else None
            return verts_img, colors_img

    return verts, colors

# =========================================================
# ==========  I N I C I A L I Z A C I Ó N  C Á M A R A  ===
# =========================================================
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
    _depth_scale = get_depth_scale(pipeline)
    return pipeline

# =========================================================
# ============  M A I N   D E   V I S U A L I Z A C I Ó N
# (Ahora usando las utilidades)
# =========================================================

def _rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """
    Construye una matriz de rotación R = Rz(roll) * Ry(yaw) * Rx(pitch)
    usando grados. Convención mano derecha.
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


def render_pointcloud(points_xyz: np.ndarray,
                      colors_bgr: np.ndarray = None,
                      out_size=(720, 720),
                      yaw_deg: float = -45.0,
                      pitch_deg: float = 25.0,
                      roll_deg: float = 0.0,
                      fov_deg: float = 60.0,
                      point_size: int = 1,
                      add_tz: float = 0.0,
                      tx: float = 0.0,
                      ty: float = 0.0) -> np.ndarray:
    """
    Renderiza una nube de puntos como proyección en 2D con una rotación fija
    para percibir profundidad.

    - points_xyz: (N,3) en metros.
    - colors_bgr: (N,3) uint8 opcional. Si None, se usa un color fijo.
    - out_size: (alto, ancho) de la imagen de salida.
    - yaw/pitch/roll: ángulos en grados para orientar la nube.
    - fov_deg: campo de visión vertical aproximado.
    - point_size: tamaño (radio) para dibujar cada punto.
    - add_tz: desplazamiento extra de la cámara en z (m) para acercar/alejar.
    - tx/ty: desplazamientos en x/y (m) para paneo.
    """
    if points_xyz is None or len(points_xyz) == 0:
        return np.zeros((out_size[0], out_size[1], 3), dtype=np.uint8)

    H, W = out_size
    img = np.zeros((H, W, 3), dtype=np.uint8)

    # Computación pesada en GPU con CuPy (obligatoria)
    pts = cp.asarray(points_xyz, dtype=cp.float32)
    # Centrar la nube para visualización más estable
    center = cp.median(pts, axis=0)
    pts_centered = pts - center

    # Rotación
    R = _rotation_matrix(yaw_deg, pitch_deg, roll_deg).astype(np.float32)
    R_cp = cp.asarray(R)
    pts_rot = pts_centered @ R_cp.T

    # Aleja la 'cámara' un poco para que todo quede delante (z>0)
    z = pts_rot[:, 2]
    # percentil en CuPy (si la versión no lo soporta, usar camino manual)
    try:
        z_min = float(cp.percentile(z, 5))
    except Exception:
        zs = cp.sort(z)
        idx = int(0.05 * (zs.size - 1))
        z_min = float(zs[idx].get())
    tz = max(0.5, -z_min + 1.5)  # offset para asegurar z positiva
    pts_cam = pts_rot + cp.asarray([tx, ty, tz + add_tz], dtype=cp.float32)

    # Proyección perspectiva
    f = 0.5 * H / np.tan(np.deg2rad(fov_deg) * 0.5)
    Z = cp.clip(pts_cam[:, 2], 1e-3, None)
    x_proj = (pts_cam[:, 0] * f) / Z
    y_proj = (pts_cam[:, 1] * f) / Z
    u = (W * 0.5 + x_proj).astype(cp.int32)
    v = (H * 0.5 - y_proj).astype(cp.int32)

    # Filtro: puntos dentro de la pantalla
    mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if int(mask.size - cp.count_nonzero(mask)) == mask.size:
        return img

    u, v, Z = u[mask], v[mask], Z[mask]
    col = None
    if colors_bgr is not None and len(colors_bgr) == len(points_xyz):
        col = (cp.asarray(colors_bgr, dtype=cp.uint8))[mask]

    # Para rendimiento, muestreamos si hay demasiados puntos
    max_pts = 120000
    if int(u.size) > max_pts:
        step = int(np.ceil(int(u.size) / max_pts))
        u, v, Z = u[::step], v[::step], Z[::step]
        if col is not None:
            col = col[::step]

    # Z-buffer básico: pintamos del fondo al frente para que el frente quede visible
    order = cp.argsort(-Z)  # de mayor Z a menor Z, así el frente (menor Z) se dibuja encima
    u, v = u[order], v[order]
    if col is not None:
        col = col[order]

    # Descargar índices/colores a NumPy para el dibujado final
    u = cp.asnumpy(u)
    v = cp.asnumpy(v)
    if col is not None:
        col = cp.asnumpy(col)
    # Dibujado
    if point_size <= 1:
        if col is None:
            img[v, u] = (200, 200, 200)
        else:
            img[v, u] = col
    else:
        for i in range(u.size):
            c = (int(col[i, 0]), int(col[i, 1]), int(col[i, 2])) if col is not None else (200, 200, 200)
            cv2.circle(img, (int(u[i]), int(v[i])), point_size, c, -1, lineType=cv2.LINE_AA)
    return img


if __name__ == "__main__":
    pipeline = init_camera(640, 480, 640, 480, 30)

    print("Presiona ESC para salir...")

    try:
        # Estado interactivo
        yaw, pitch, roll = -45.0, 25.0, 0.0
        fov = 60.0
        point_size = 1
        add_tz = 0.0
        pan_tx, pan_ty = 0.0, 0.0

        step_angle = 5.0
        step_zoom = 0.2   # metros
        step_pan = 0.05   # metros
        step_fov = 5.0

        cv2.namedWindow('RealSense PointCloud', cv2.WINDOW_NORMAL)

        while True:
            # Espera frames sincronizados
            frames = pipeline.wait_for_frames()

            # Nube de puntos y render 2D con ángulo/posición ajustables
            points_xyz, colors_bgr = extract_pointcloud(frames, with_colors=True, filter_invalid=True, organized=False)
            if points_xyz is None:
                continue

            pc_img = render_pointcloud(points_xyz, colors_bgr,
                                    out_size=(720, 720),
                                    yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
                                    fov_deg=fov, point_size=point_size,
                                    add_tz=add_tz, tx=pan_tx, ty=pan_ty)

            # HUD
            hud = f"Yaw:{yaw:.0f}  Pitch:{pitch:.0f}  Roll:{roll:.0f}  FOV:{fov:.0f}  Size:{point_size}  Zoff:{add_tz:+.2f}  Pan({pan_tx:+.2f},{pan_ty:+.2f})"
            cv2.putText(pc_img, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(pc_img, "Controles: WASD rotar | Q/E roll | Z/X FOV | +/- tamaño | I/K/J/L paneo | [/ ] z-off | R reset | ESC salir", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA)

            # Muestra
            cv2.imshow('RealSense PointCloud', pc_img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

            # Controles
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
        # Limpieza
        pipeline.stop()
        cv2.destroyAllWindows()
