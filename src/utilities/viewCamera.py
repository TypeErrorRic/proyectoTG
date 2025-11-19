"""
Utilidades RealSense para obtener RGB, profundidad (m) y rayos por píxel, con opción
de alinear DEPTH→COLOR en GPU o CPU. Incluye un demo que genera una nube de puntos
desde rayos + profundidad y la renderiza en 2D.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time
from typing import Tuple, Callable, Optional
import math

# =========================================================
# ===============  U T I L I D A D E S  ===================
# =========================================================

_DEPTH_SCALE_CACHE = None

def get_depth_scale(pipeline: rs.pipeline = None) -> float:
    """
    Lee y cachea el depth_scale del sensor. Si falta pipeline, intenta el primer
    dispositivo disponible. Fallback: 0.001 (1 mm).
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
    Devuelve RGB (BGR, uint8) de forma (H, W, 3) o None si no hay color.
    """
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    img = np.asanyarray(color_frame.get_data())
    return img.copy() if copy else img


def extract_depth_raw(frames: rs.composite_frame) -> np.ndarray:
    """
    Devuelve profundidad nativa (uint16, H×W) o None si falta.
    """
    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        return None
    depth_raw = np.asanyarray(depth_frame.get_data())
    return depth_raw


def extract_depth_meters(frames: rs.composite_frame, depth_scale: float = None) -> np.ndarray:
    """
    Profundidad en metros (float32, H×W). Usa depth_scale cacheado si no se pasa.
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
    Calcula rayos por píxel en coords de cámara. Retorna (H, W, 3) con [x/z, y/z, 1].
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


def compute_normalized_rays(H: int, W: int) -> np.ndarray:
    """
    Construye rayos \"normalizados\" para una imagen de tamaño (H, W)
    cuando no se conocen intrínsecas reales de cámara.

    Devuelve un array (H, W, 3) con vectores [x, y, 1] donde x,y están
    en el rango aproximado [-1, 1]. Esto es suficiente para RANSAC sobre
    un mapa de profundidad de dataset.
    """
    u = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)  # (H,W)
    ones = np.ones_like(uu, dtype=np.float32)
    rays = np.stack([uu, vv, ones], axis=-1).astype(np.float32)
    return rays

###############################################
# Alineación DEPTH→COLOR (GPU opcional)
###############################################

# Parámetros de rendimiento básicos:
# - ALIGN_DOWNSAMPLE: submuestreo previo del depth (1=sin DS).
# - ALIGN_MAX_DEPTH_M: recorte por distancia (<=0 desactiva).
ALIGN_DOWNSAMPLE: int = 2
ALIGN_MAX_DEPTH_M: Optional[float] = 6

# Kernel CUDA para alineación (z-buffer por mínimo Z en cámara color)
_ALIGN_KERNEL_SRC = r"""
extern "C" __global__
void align_depth_to_color(
    const float* __restrict__ depth,    // (Hd*Wd)
    const float* __restrict__ A,        // (Hd*Wd*3) precomputado: R_cd * ray_d
    const float* __restrict__ t,        // (3) traslación depth->color
    const float fx, const float fy,
    const float cx, const float cy,
    const float max_depth,              // <=0: desactivado
    const int Hd, const int Wd,
    const int Hc, const int Wc,
    unsigned int* __restrict__ out_bits // (Hc*Wc) inicializado a +inf
){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int N = Hd * Wd;
    if (idx >= N) return;

    float z = depth[idx];
    if (!(z > 0.0f) || !isfinite(z)) return;
    if (max_depth > 0.0f && z > max_depth) return;

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
        Alineador DEPTH→COLOR en GPU (CuPy + CUDA): precalcula A = R_cd @ ray_d y
        proyecta cada píxel de depth al plano color con z-buffer atómico.
    """

    def __init__(self, pipeline: rs.pipeline, downsample: int = 1, max_depth_m: Optional[float] = None, reuse_output: bool = True) -> None:
        import cupy as cp  # import local: solo si se usa GPU
        prof = pipeline.get_active_profile()
        depth_prof = prof.get_stream(rs.stream.depth).as_video_stream_profile()
        color_prof = prof.get_stream(rs.stream.color).as_video_stream_profile()

        # Intrínsecos
        intr_d = depth_prof.get_intrinsics()
        intr_c = color_prof.get_intrinsics()
        # Downsample del depth para reducir coste
        self._ds = max(1, int(downsample))
        self.Wd_native, self.Hd_native = intr_d.width, intr_d.height
        self.Wd = (self.Wd_native + (self._ds - 1)) // self._ds
        self.Hd = (self.Hd_native + (self._ds - 1)) // self._ds
        self.Wc, self.Hc = intr_c.width, intr_c.height
        self.fx_c = float(intr_c.fx)
        self.fy_c = float(intr_c.fy)
        self.cx_c = float(intr_c.ppx)
        self.cy_c = float(intr_c.ppy)
        self._max_depth = float(max_depth_m) if (max_depth_m is not None) else -1.0

        # Extrínsecos depth→color
        extr = depth_prof.get_extrinsics_to(color_prof)
        R = np.asarray(extr.rotation, dtype=np.float32).reshape(3, 3)
        t = np.asarray(extr.translation, dtype=np.float32).reshape(3)
        self._cp = cp
        self.t_cp = cp.asarray(t, dtype=cp.float32)

        # Rayos de profundidad (con DS si aplica)
        rays_d_full = compute_rays_from_intrinsics(intr_d).astype(np.float32)  # (Hd_native,Wd_native,3)
        if self._ds > 1:
            rays_d = rays_d_full[::self._ds, ::self._ds]
        else:
            rays_d = rays_d_full

        # A(x,y) = R_cd * ray_d(x,y)
        A = np.tensordot(rays_d, R.T, axes=(2, 0))  # (Hd,Wd,3)
        self.A_cp = cp.asarray(A, dtype=cp.float32)

        # Compilar kernel
        self._kernel = cp.RawKernel(_ALIGN_KERNEL_SRC, 'align_depth_to_color')

        # Salida reutilizable (evita malloc/fill por frame)
        self._inf_bits = np.float32(np.inf).view(np.uint32)
        self._out_bits = cp.empty((self.Hc, self.Wc), dtype=cp.uint32) if reuse_output else None

    def align(self, depth_m: np.ndarray) -> np.ndarray:
        """
        Depth alineado al espacio COLOR (Hc×Wc) en metros (float32).
        """
        cp = self._cp
        if depth_m is None:
            return None
        # Ajustar tamaño a (Hd, Wd) si es necesario
        if depth_m.shape != (self.Hd, self.Wd):
            depth_m = cv2.resize(depth_m, (self.Wd, self.Hd), interpolation=cv2.INTER_NEAREST)

        depth_cp = cp.asarray(depth_m, dtype=cp.float32)
        # Salida a +inf (uint32) para atomicMin bit a bit
        if self._out_bits is None:
            out_bits = cp.full((self.Hc, self.Wc), self._inf_bits, dtype=cp.uint32)
        else:
            out_bits = self._out_bits
            out_bits.fill(self._inf_bits)

        # Ejecutar kernel
        N = int(self.Hd * self.Wd)
        threads = 256
        blocks = (N + threads - 1) // threads
        self._kernel((blocks,), (threads,), (
            depth_cp,
            self.A_cp.ravel(),
            self.t_cp,
            np.float32(self.fx_c), np.float32(self.fy_c),
            np.float32(self.cx_c), np.float32(self.cy_c),
            np.float32(self._max_depth),
            np.int32(self.Hd), np.int32(self.Wd),
            np.int32(self.Hc), np.int32(self.Wc),
            out_bits.ravel()
        ))

        # Convertir a float32; 0.0 donde quede +inf (sin asignación)
        depth_c = out_bits.view(cp.float32)
        mask_inf = cp.isinf(depth_c)
        if mask_inf.any():
            depth_c = depth_c.copy()
            depth_c[mask_inf] = 0.0
        return depth_c.get()


def make_depth_to_color_aligner(pipeline: rs.pipeline) -> Callable[[rs.composite_frame], Optional[np.ndarray]]:
    """
    Devuelve una función align(frames)-> depth (float32, Hc×Wc) en espacio COLOR.
    Prefiere GPU (CuPy); fallback a rs.align en CPU.
    """
    try:
        # Intentar GPU
        ds = max(1, int(ALIGN_DOWNSAMPLE))
        max_d = ALIGN_MAX_DEPTH_M if (ALIGN_MAX_DEPTH_M is not None and ALIGN_MAX_DEPTH_M > 0) else None
        aligner_gpu = DepthToColorAlignerGPU(pipeline, downsample=ds, max_depth_m=max_d, reuse_output=True)

        def _align(frames: rs.composite_frame) -> Optional[np.ndarray]:
            depth_native = extract_depth_meters(frames)
            if depth_native is None:
                return None
            return aligner_gpu.align(depth_native)

        return _align
    except Exception:
        # Fallback: rs.align en CPU
        align_to_color = rs.align(rs.stream.color)

        def _align_cpu(frames: rs.composite_frame) -> Optional[np.ndarray]:
            aligned_frames = align_to_color.process(frames)
            return extract_depth_meters(aligned_frames)

        return _align_cpu


def precompute_rays_for_stream(pipeline: rs.pipeline, stream: rs.stream = rs.stream.depth) -> Tuple[np.ndarray, int, int, Optional[Callable[[rs.composite_frame], Optional[np.ndarray]]]]:
    """
        Precalcula rayos (H, W, 3) para el stream dado. Si stream=color, también
        retorna una función para alinear depth→color.
    """
    prof = pipeline.get_active_profile().get_stream(stream).as_video_stream_profile()
    intr = prof.get_intrinsics()
    rays_np = compute_rays_from_intrinsics(intr)
    H, W = intr.height, intr.width

    align_fn: Optional[Callable[[rs.composite_frame], Optional[np.ndarray]]] = None
    if stream == rs.stream.color:
        # Preparar función de alineación depth→color
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
    stride: int = 2,
    yaw: float = -45.0,
    pitch: float = 25.0,
    roll: float = 0.0,
    fov: float = 60.0,
    point_size: int = 1,
):
    """
        Inicia RealSense con streams de color y depth.
        Devuelve: (pipeline, params dict para visualización de nube de puntos).
    """
    pipeline = rs.pipeline()
    config = rs.config()

    # Streams (ajusta resolución según modelo)
    config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, fps)

    pipeline.start(config)
    _depth_scale = get_depth_scale(pipeline)
    
    # Parámetros de visualización
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
    
    # Config en init_camera
    pipeline, params = init_camera(
        color_width=640,
        color_height=480,
        depth_width=640,
        depth_height=480,
        fps=30,
        stride=2,        # submuestreo para nube
        yaw=-45.0,
        pitch=25.0,
        roll=0.0,
        fov=60.0,
        point_size=1
    )
    
    print("Demo: Nube de puntos desde rayos + profundidad (ESC para salir)")
    try:
        # Usar stream COLOR para tener rayos en el mismo espacio que RGB
        rays_np, H, W, align_depth_fn = precompute_rays_for_stream(pipeline, rs.stream.color)
        
        # Parámetros de visualización
        stride_demo = params['stride']
        yaw = params['yaw']
        pitch = params['pitch']
        roll = params['roll']
        fov = params['fov']
        point_size = params['point_size']
        
        cv2.namedWindow('RGB', cv2.WINDOW_NORMAL)
        cv2.namedWindow('PointCloud', cv2.WINDOW_NORMAL)
        
        # Métrica simple de tiempos
        frame_count = 0
        
        while True:
            t_start_frame = time.perf_counter()
            
            # 1) Captura
            t1 = time.perf_counter()
            frames = pipeline.wait_for_frames()
            t2 = time.perf_counter()
            
            # 2) RGB y DEPTH→COLOR
            rgb = extract_rgb(frames)
            t2a = time.perf_counter()
            depth_m = align_depth_fn(frames) if align_depth_fn is not None else extract_depth_meters(frames)
            t3 = time.perf_counter()
            
            if rgb is None or depth_m is None:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                continue

            # 3) Ajuste de dimensiones
            if depth_m.shape != (H, W):
                depth_m = cv2.resize(depth_m, (W, H), interpolation=cv2.INTER_NEAREST)
            t4 = time.perf_counter()

            # 4) Nube de puntos (rayos + depth)
            points_xyz = points_from_rays_and_depth(rays_np, depth_m, stride=stride_demo)
            t5 = time.perf_counter()
            
            # 5) Colores
            rgb_sub = rgb[::stride_demo, ::stride_demo]
            depth_sub = depth_m[::stride_demo, ::stride_demo]
            valid = (depth_sub > 0)
            
            if points_xyz is not None and points_xyz.size > 0:
                colors = rgb_sub[valid]  # Indexación directa con máscara 2D
            else:
                colors = None
            t6 = time.perf_counter()

            # 6) Render 2D
            pc_img = render_pointcloud_numpy(points_xyz, colors,
                                             out_size=(720, 720),
                                             yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
                                             fov_deg=fov, point_size=point_size)
            t7 = time.perf_counter()

            # 7) HUD y visualización
            t_end_frame = time.perf_counter()
            
            vis = rgb.copy()
            hud1 = f"rays: {H}x{W}x3  stride:{stride_demo}  pts:{0 if points_xyz is None else len(points_xyz)}"
            hud2 = f"Yaw:{yaw:.0f} Pitch:{pitch:.0f} Roll:{roll:.0f} FOV:{fov:.0f} Size:{point_size}"
            cv2.putText(vis, hud1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(vis, hud2, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
            
            cv2.putText(pc_img, "Controles: A/D yaw  W/S pitch  Q/E roll  Z/X FOV  +/- tamaño  ESC",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 1, cv2.LINE_AA)
            cv2.imshow('RGB', vis)
            cv2.imshow('PointCloud', pc_img)
            
            # Tiempos cada 60 frames
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
            # Controles
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
