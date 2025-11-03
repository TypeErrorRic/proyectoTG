"""
Utilidades de cámara Intel RealSense necesarias para el flujo de RANSAC en ransacCellingGround:
- get_depth_scale(pipeline=None): lee el "depth scale" del sensor de profundidad.
- extract_rgb(frames): imagen BGR (np.uint8).
- extract_depth_raw(frames): depth en uint16 (unidades nativas).
- extract_depth_meters(frames, depth_scale=None): depth en metros (float32).
- compute_rays_from_intrinsics(intr): calcula los rayos por píxel a coords de cámara.
- precompute_rays_for_stream(pipeline, stream): obtiene rayos (H, W, 3) para el stream indicado (depth/color).
- init_camera(...): inicializa la cámara.

Incluye un main ligero que muestra qué se le entregaría a la función de RANSAC:
    - dimensiones de rays (H, W, 3), depth (H, W) y un conteo de puntos válidos tras un submuestreo.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time
from typing import Tuple, Callable, Optional

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


def compute_rays_from_intrinsics(intr) -> np.ndarray:
    """
    Precalcula los rayos de back-proyección por píxel a coords de cámara.
    Devuelve array (H, W, 3) con [x/z, y/z, 1].
    """
    W, H = intr.width, intr.height
    fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)  # (H,W)
    x = (uu - cx) / float(fx)
    y = (vv - cy) / float(fy)
    ones = np.ones_like(x, dtype=np.float32)
    rays = np.stack([x, y, ones], axis=-1).astype(np.float32)
    return rays

###############################################
# Alineación DEPTH -> COLOR (GPU opcional)
###############################################

# Kernel CUDA para alineación (idéntico al usado previamente)
_ALIGN_KERNEL_SRC = r"""
extern "C" __global__
void align_depth_to_color(
    const float* __restrict__ depth,    // (Hd*Wd)
    const float* __restrict__ A,        // (Hd*Wd*3) precomputado: R_cd * ray_d
    const float* __restrict__ t,        // (3) traslación depth->color
    const float fx, const float fy,
    const float cx, const float cy,
    const int Hd, const int Wd,
    const int Hc, const int Wc,
    unsigned int* __restrict__ out_bits // (Hc*Wc) inicializado a +inf
){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int N = Hd * Wd;
    if (idx >= N) return;

    float z = depth[idx];
    if (!(z > 0.0f) || !isfinite(z)) return;

    // A indexado linealmente (x3)
    int aidx = idx * 3;
    float ax = A[aidx + 0];
    float ay = A[aidx + 1];
    float az = A[aidx + 2];

    // Transformar punto de depth->color
    float Xcx = ax * z + t[0];
    float Xcy = ay * z + t[1];
    float Xcz = az * z + t[2];
    if (!(Xcz > 0.0f) || !isfinite(Xcz)) return;

    // Proyectar a píxel de color
    float u = fx * (Xcx / Xcz) + cx;
    float v = fy * (Xcy / Xcz) + cy;

    int ui = (int)roundf(u);
    int vi = (int)roundf(v);
    if (ui < 0 || ui >= Wc || vi < 0 || vi >= Hc) return;

    // Z-buffer: quedarse con el punto más cercano (menor Z en cámara color)
    unsigned int zbits = __float_as_uint(Xcz);
    int o = vi * Wc + ui;
    atomicMin(&out_bits[o], zbits);
}
"""


class DepthToColorAlignerGPU:
    """
    Alineador rápido de DEPTH->COLOR usando GPU (CuPy + kernel CUDA).

    - Precalcula A = R_cd @ ray_d(x,y) por píxel (depth) y usa t_cd.
    - Por frame: hace un pase sobre depth y "salpica" al plano de color
      con z-buffer (atómico) para quedarse con la muestra más cercana.
    """

    def __init__(self, pipeline: rs.pipeline) -> None:
        import cupy as cp  # import local para no requerir CuPy si no se usa
        prof = pipeline.get_active_profile()
        depth_prof = prof.get_stream(rs.stream.depth).as_video_stream_profile()
        color_prof = prof.get_stream(rs.stream.color).as_video_stream_profile()

        # Intrínsecos
        intr_d = depth_prof.get_intrinsics()
        intr_c = color_prof.get_intrinsics()
        self.Wd, self.Hd = intr_d.width, intr_d.height
        self.Wc, self.Hc = intr_c.width, intr_c.height
        self.fx_c = float(intr_c.fx)
        self.fy_c = float(intr_c.fy)
        self.cx_c = float(intr_c.ppx)
        self.cy_c = float(intr_c.ppy)

        # Extrínsecos depth->color (rotación y traslación)
        extr = depth_prof.get_extrinsics_to(color_prof)
        R = np.asarray(extr.rotation, dtype=np.float32).reshape(3, 3)
        t = np.asarray(extr.translation, dtype=np.float32).reshape(3)
        self._cp = cp
        self.t_cp = cp.asarray(t, dtype=cp.float32)

        # Rayos del stream de profundidad
        rays_d = compute_rays_from_intrinsics(intr_d).astype(np.float32)  # (Hd,Wd,3)

        # A(x,y) = R_cd * ray_d(x,y)
        A = np.tensordot(rays_d, R.T, axes=(2, 0))  # (Hd,Wd,3)
        self.A_cp = cp.asarray(A, dtype=cp.float32)

        # Compilar kernel
        self._kernel = cp.RawKernel(_ALIGN_KERNEL_SRC, 'align_depth_to_color')

    def align(self, depth_m: np.ndarray) -> np.ndarray:
        """
        Devuelve depth alineado al espacio de COLOR (Hc,Wc) en metros (float32).
        """
        cp = self._cp
        if depth_m is None:
            return None
        # Ajustar tamaño de entrada si hiciera falta
        if depth_m.shape != (self.Hd, self.Wd):
            depth_m = cv2.resize(depth_m, (self.Wd, self.Hd), interpolation=cv2.INTER_NEAREST)

        depth_cp = cp.asarray(depth_m, dtype=cp.float32)
        # Salida inicializada a +inf (como uint32 para atomicMin bit a bit)
        out_bits = cp.full((self.Hc, self.Wc), np.float32(np.inf).view(np.uint32), dtype=cp.uint32)

        # Lanzar kernel
        N = int(self.Hd * self.Wd)
        threads = 256
        blocks = (N + threads - 1) // threads
        self._kernel((blocks,), (threads,), (
            depth_cp,
            self.A_cp.ravel(),
            self.t_cp,
            np.float32(self.fx_c), np.float32(self.fy_c),
            np.float32(self.cx_c), np.float32(self.cy_c),
            np.int32(self.Hd), np.int32(self.Wd),
            np.int32(self.Hc), np.int32(self.Wc),
            out_bits.ravel()
        ))

        # Convertir a float32 y poner 0.0 donde quedó +inf (no asignado)
        depth_c = out_bits.view(cp.float32)
        mask_inf = cp.isinf(depth_c)
        if mask_inf.any():
            depth_c = depth_c.copy()
            depth_c[mask_inf] = 0.0
        return depth_c.get()


def make_depth_to_color_aligner(pipeline: rs.pipeline) -> Callable[[rs.composite_frame], Optional[np.ndarray]]:
    """
    Crea una función align(frames)->depth_en_color (float32, Hc x Wc) que
    usa GPU (CuPy) si está disponible; si no, usa rs.align como respaldo.
    """
    try:
        # Intentar GPU
        aligner_gpu = DepthToColorAlignerGPU(pipeline)

        def _align(frames: rs.composite_frame) -> Optional[np.ndarray]:
            depth_native = extract_depth_meters(frames)
            if depth_native is None:
                return None
            return aligner_gpu.align(depth_native)

        return _align
    except Exception:
        # Respaldo con rs.align
        align_to_color = rs.align(rs.stream.color)

        def _align_cpu(frames: rs.composite_frame) -> Optional[np.ndarray]:
            aligned_frames = align_to_color.process(frames)
            return extract_depth_meters(aligned_frames)

        return _align_cpu


def precompute_rays_for_stream(pipeline: rs.pipeline, stream: rs.stream = rs.stream.depth) -> Tuple[np.ndarray, int, int, Optional[Callable[[rs.composite_frame], Optional[np.ndarray]]]]:
    """
    Igual que precompute_rays_from_pipeline, pero permitiendo elegir el stream
    (p.ej. rs.stream.color) para que los rayos estén en el mismo espacio de píxeles
    que el frame con el que se quiera correlacionar por píxel.

    Retorna (rays_np, H, W, align_fn):
      - align_fn(frames) devuelve depth alineado al espacio de COLOR si stream=color,
        o None si no aplica.
    """
    prof = pipeline.get_active_profile().get_stream(stream).as_video_stream_profile()
    intr = prof.get_intrinsics()
    rays_np = compute_rays_from_intrinsics(intr)
    H, W = intr.height, intr.width

    align_fn: Optional[Callable[[rs.composite_frame], Optional[np.ndarray]]] = None
    if stream == rs.stream.color:
        # Preparar función de alineación depth->color
        align_fn = make_depth_to_color_aligner(pipeline)

    return rays_np, H, W, align_fn

# =========================================================
# ==========  I N I C I A L I Z A C I Ó N  C Á M A R A  ===
# =========================================================
def init_camera(
    color_width: int = 640,
    color_height: int = 480,
    depth_width: int = 640,
    depth_height: int = 480,
    fps: int = 30,
    stride: int = 1,
    yaw: float = -45.0,
    pitch: float = 25.0,
    roll: float = 0.0,
    fov: float = 60.0,
    point_size: int = 1,
):
    """
    Inicializa el pipeline de Intel RealSense con streams de color y profundidad.

    Parámetros:
      - color_width/height: resolución para el stream de color.
      - depth_width/height: resolución para el stream de profundidad.
      - fps: cuadros por segundo para ambos streams.
      - stride: submuestreo para la nube de puntos (1=máxima densidad, >1=menos puntos).
      - yaw/pitch/roll: ángulos iniciales de visualización de la nube de puntos (grados).
      - fov: campo de visión para la proyección de la nube (grados).
      - point_size: tamaño de los puntos en el render (píxeles).

    Retorna:
      - pipeline inicializado y en ejecución.
      - diccionario con parámetros de configuración.
    """
    pipeline = rs.pipeline()
    config = rs.config()

    # Habilita streams (ajusta resolución según tu modelo D435/D415/L515)
    config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, fps)

    pipeline.start(config)
    _depth_scale = get_depth_scale(pipeline)
    
    # Parámetros de procesamiento
    params = {
        'stride': stride,
        'yaw': yaw,
        'pitch': pitch,
        'roll': roll,
        'fov': fov,
        'point_size': point_size,
    }
    
    return pipeline, params

# =========================================================
# ============  M A I N   D E   V I S U A L I Z A C I Ó N
# =========================================================

if __name__ == "__main__":
    from helpers import *
    
    # Configuración centralizada en init_camera
    pipeline, params = init_camera(
        color_width=640,
        color_height=480,
        depth_width=640,
        depth_height=480,
        fps=30,
        stride=2,        # submuestreo: 1=máxima densidad, 2-4=menos puntos
        yaw=-45.0,       # rotación horizontal
        pitch=25.0,      # rotación vertical
        roll=0.0,        # rotación en eje Z
        fov=60.0,        # campo de visión
        point_size=1     # tamaño de puntos en render
    )
    
    print("Demo: Nube de puntos desde rayos y profundidad (ESC para salir)")
    try:
        # Usar la versión basada en stream COLOR para demostrar alineación depth->color
        # De esta forma, los rayos (u,v)->rayo están en el espacio de la imagen RGB
        rays_np, H, W, align_depth_fn = precompute_rays_for_stream(pipeline, rs.stream.color)
        
        # Extraer parámetros de configuración
        stride_demo = params['stride']
        yaw = params['yaw']
        pitch = params['pitch']
        roll = params['roll']
        fov = params['fov']
        point_size = params['point_size']
        
        cv2.namedWindow('RGB', cv2.WINDOW_NORMAL)
        cv2.namedWindow('PointCloud', cv2.WINDOW_NORMAL)
        
        # Variables para medición de tiempos
        frame_count = 0
        
        while True:
            t_start_frame = time.perf_counter()
            
            # 1. Captura de frames
            t1 = time.perf_counter()
            frames = pipeline.wait_for_frames()
            t2 = time.perf_counter()
            
            # 2. Extracción RGB y alineación DEPTH->COLOR
            rgb = extract_rgb(frames)
            t2a = time.perf_counter()
            depth_m = align_depth_fn(frames) if align_depth_fn is not None else extract_depth_meters(frames)
            t3 = time.perf_counter()
            
            if rgb is None or depth_m is None:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                continue

            # 3. Ajuste de dimensiones (por seguridad)
            if depth_m.shape != (H, W):
                depth_m = cv2.resize(depth_m, (W, H), interpolation=cv2.INTER_NEAREST)
            t4 = time.perf_counter()

            # 4. Construir nube de puntos a partir de rayos y profundidad
            points_xyz = points_from_rays_and_depth(rays_np, depth_m, stride=stride_demo)
            t5 = time.perf_counter()
            
            # 5. Extracción de colores
            rgb_sub = rgb[::stride_demo, ::stride_demo]
            depth_sub = depth_m[::stride_demo, ::stride_demo]
            valid = (depth_sub > 0)
            
            if points_xyz is not None and points_xyz.size > 0:
                colors = rgb_sub[valid]  # Indexación directa con máscara 2D
            else:
                colors = None
            t6 = time.perf_counter()

            # 6. Render 2D de la nube
            pc_img = render_pointcloud_numpy(points_xyz, colors,
                                             out_size=(720, 720),
                                             yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
                                             fov_deg=fov, point_size=point_size)
            t7 = time.perf_counter()

            # 7. HUD y visualización
            t_end_frame = time.perf_counter()
            
            vis = rgb.copy()
            hud1 = f"rays: {H}x{W}x3  stride:{stride_demo}  pts:{0 if points_xyz is None else len(points_xyz)}"
            hud2 = f"Yaw:{yaw:.0f} Pitch:{pitch:.0f} Roll:{roll:.0f} FOV:{fov:.0f} Size:{point_size}"
            cv2.putText(vis, hud1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(vis, hud2, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
            
            cv2.putText(pc_img, "Controles: A/D yaw  W/S pitch  Q/E roll  Z/X FOV  +/- tamaño  ESC salir",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 1, cv2.LINE_AA)
            cv2.imshow('RGB', vis)
            cv2.imshow('PointCloud', pc_img)
            
            # Imprimir tiempos cada 60 frames (incluyendo alineación)
            frame_count += 1
            if frame_count % 60 == 0:
                print(f"\n=== Tiempos Frame #{frame_count} ===")
                print(f"  Captura frames:     {(t2-t1)*1000:6.2f} ms")
                print(f"  Extracción RGB:     {(t2a-t2)*1000:6.2f} ms")
                print(f"  Alineación D->C:    {(t3-t2a)*1000:6.2f} ms")
                print(f"  Resize depth:       {(t4-t3)*1000:6.2f} ms")
                print(f"  Rayos -> PointCloud:{(t5-t4)*1000:6.2f} ms")
                print(f"  Extracción colores: {(t6-t5)*1000:6.2f} ms")
                print(f"  Render PointCloud:  {(t7-t6)*1000:6.2f} ms")
                print(f"  TOTAL:              {(t_end_frame-t_start_frame)*1000:6.2f} ms ({1.0/(t_end_frame-t_start_frame):.1f} FPS)")

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            # Controles simples
            if key == ord('a'):
                yaw -= 5.0
            elif key == ord('d'):
                yaw += 5.0
            elif key == ord('w'):
                pitch += 5.0
            elif key == ord('s'):
                pitch -= 5.0
            elif key == ord('q'):
                roll -= 5.0
            elif key == ord('e'):
                roll += 5.0
            elif key == ord('z'):
                fov = max(20.0, fov - 5.0)
            elif key == ord('x'):
                fov = min(120.0, fov + 5.0)
            elif key in (ord('+'), ord('=')):
                point_size = min(6, point_size + 1)
            elif key in (ord('-'), ord('_')):
                point_size = max(1, point_size - 1)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()