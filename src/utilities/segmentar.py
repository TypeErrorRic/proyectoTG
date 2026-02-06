"""
Thread module for Jetson Nano.

Provides a background worker thread that executes a segmentation task
and shares its result with the main thread through a small queue.
"""

# Project libraries
import src.utilities.viewCamera as viewCamera
from utilities.GroundDetection import get_ground, get_last_ransac_ms
import utilities.GroundDetection as ground_utils
from src.utilities.helpers import (
    apply_mask_to_rgb,
    load_dataset_frame,
    mejorar_mascara_pared,
    mejorar_mascara_suelo,
)
from utilities.WallPlaneDetection import get_wall_planes
from src.models.doorDetection import doorDetection

# Runtime libraries
import numpy as np
import cupy as cp
import cv2
import threading
import time
import queue
from typing import Optional, Callable, Any, Tuple, Dict


# Centralized state dictionary to avoid loose globals
_runtime: Dict[str, Any] = {
    "initialized": False,
    "mode": None,
    "pipeline": None,
    "rays_cp": None,
    "H": None,
    "W": None,
    "align_depth_fn": None,
    "imagenRGB": None,
    "mapaProfundidad": None,
    "last_ransac_ms": None,
    "groundParams": {
        "dist_thresh": 0.04,
        "max_iters": 300,
        "min_inliers": 400,
        "subsample_stride": 2,
        "up_axis": (0.0, -1.0, 0.0),
        "max_angle_deg": 60.0,
        "seed": 42,
        "score_subset": 2048,
        "orientation": "ground",
        "early_stop_ratio": 0.90,
        "batch_size": 512,
        # Extra controls for quality vs velocidad
        "low_height_pct": 25.0,          # usar percentil inferior en altura
        "roi_bottom_fraction": 0.34,     # arranca con este porcentaje inferior
        "roi_expand_step": 0.2,          # expande ROI hacia arriba si faltan puntos
        "max_agg_points": 150000,        # límite de puntos usados en RANSAC
        "refine_full_res": True,         # refinar plano con inliers full-res
        "refine_max_points": 200000,     # límite puntos en refinamiento
        "refine_dist_mult": 1.6,         # tolerancia para recolectar inliers al refinar
        "ground_mask_refine": True,     # mejora opcional mascara de suelo
    },
    "wallParams": {
        "max_up_dot": 0.35,              # |dot(normal, up)| maximo para paredes
        "ground_perp_deg": 20.0,
        "wall_ortho_deg": 20.0,
        "wall_parallel_deg": 10.0,
        "wall_parallel_distance_m": 0.60,
        "wall_mask_refine": True,       # mejora opcional mascara de pared
    },
    "doorParams": {
        "door_hue_tol": 18,              # tolerancia HSV (H) para puerta
        "door_min_s": 30,                # saturacion minima HSV
        "door_min_v": 20,                # valor minimo HSV
        "door_glare_s_max": 35,          # max S para considerar glare
        "door_glare_v_min": 210,         # min V para considerar glare
        "door_glare_v_clip": 200,        # V usado al recortar glare
    },
}

GROUND_PARAM_KEYS = {
    "dist_thresh",
    "max_iters",
    "min_inliers",
    "subsample_stride",
    "up_axis",
    "max_angle_deg",
    "seed",
    "score_subset",
    "orientation",
    "early_stop_ratio",
    "batch_size",
    "low_height_pct",
    "roi_bottom_fraction",
    "roi_expand_step",
    "max_agg_points",
    "refine_full_res",
    "refine_max_points",
    "refine_dist_mult",
    "ground_mask_refine",
}

WALL_PARAM_KEYS = {
    "max_up_dot",
    "ground_perp_deg",
    "wall_ortho_deg",
    "wall_parallel_deg",
    "wall_parallel_distance_m",
    "wall_mask_refine",
}

DOOR_PARAM_KEYS = {
    "door_hue_tol",
    "door_min_s",
    "door_min_v",
    "door_glare_s_max",
    "door_glare_v_min",
    "door_glare_v_clip",
}

# Wall-plane overrides applied on top of ground parameters.
WALL_PARAMS_OVERRIDES: Dict[str, Any] = {
    "max_angle_deg": 20.0,
    "max_planes": 3,
    "enforce_vertical": True,
    "max_up_dot": 0.35,
    "refine": True,
    "ground_perp_deg": 20.0,
    "wall_ortho_deg": 20.0,
    "wall_parallel_deg": 10.0,
    "wall_parallel_distance_m": 0.60,
}

# Protect shared runtime parameters updated from the GUI while the worker runs.
_runtime_lock = threading.Lock()

# Event used to stop the worker thread cleanly
_detener_evento = threading.Event()
_hilo_trabajador: Optional[threading.Thread] = None

# Result queue shared with the main thread (buffer of 1 element)
_resultados: "queue.Queue[Any]" = queue.Queue(maxsize=1)

# Type of the function that will be executed in the worker thread
TareaFuncion = Callable[..., Any]

# Reference to the function that will run in the worker
_tarea_funcion: Optional[TareaFuncion] = None
_tarea_args: Tuple[Any, ...] = ()
_tarea_kwargs: Dict[str, Any] = {}


def _snapshot_param_groups() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Return copies of ground, wall, and door parameters.
    """
    with _runtime_lock:
        ground_params = dict(_runtime.get("groundParams", {}) or {})
        wall_params = dict(_runtime.get("wallParams", {}) or {})
        door_params = dict(_runtime.get("doorParams", {}) or {})
    return ground_params, wall_params, door_params


def obtener_parametros_ground(copy: bool = True) -> Dict[str, Any]:
    """
    Snapshot of the current segmentation parameters (ground + wall + door).
    """
    ground_params, wall_params, door_params = _snapshot_param_groups()
    merged = {**ground_params, **wall_params, **door_params}
    return merged.copy() if copy else merged


def actualizar_parametros_ground(nuevos_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge new parameter values into the runtime dictionary.

    Returns the updated parameter dict.
    """
    if not nuevos_params:
        return obtener_parametros_ground()

    with _runtime_lock:
        ground_params = dict(_runtime.get("groundParams", {}) or {})
        wall_params = dict(_runtime.get("wallParams", {}) or {})
        door_params = dict(_runtime.get("doorParams", {}) or {})
        for key, value in nuevos_params.items():
            if value is None:
                continue
            if key in DOOR_PARAM_KEYS:
                door_params[key] = value
            elif key in WALL_PARAM_KEYS:
                wall_params[key] = value
            elif key in GROUND_PARAM_KEYS:
                ground_params[key] = value
            else:
                ground_params[key] = value
        _runtime["groundParams"] = ground_params
        _runtime["wallParams"] = wall_params
        _runtime["doorParams"] = door_params
        merged = {**ground_params, **wall_params, **door_params}
        return merged.copy()


def obtener_metricas(copy: bool = True) -> Dict[str, Any]:
    """
    Snapshot of runtime metrics (currently RANSAC duration in ms).
    """
    with _runtime_lock:
        last_ms = _runtime.get("last_ransac_ms")
        metrics = {"last_ransac_ms": last_ms}
    return metrics.copy() if copy else metrics


def segmentar() -> Any:
    """
    Segmentation worker.

    Uses the current frame and parameters stored in _runtime to detect
    the ground, wall and door planes and returns the RGB image with all masks
    overlaid.
    """
    imagenRGB = _runtime.get("imagenRGB")
    mapaProfundidad = _runtime.get("mapaProfundidad")
    rays_cp = _runtime.get("rays_cp")
    H = _runtime.get("H")
    W = _runtime.get("W")
    use_realsense = _runtime.get("mode", "camera") == "camera"

    # Bail out gracefully if data is missing (e.g., right after mode switch)
    if imagenRGB is None or mapaProfundidad is None or rays_cp is None or H is None or W is None:
        with _runtime_lock:
            _runtime["last_ransac_ms"] = None
        return imagenRGB

    depth_cp = None
    try:
        depth_cp = cp.asarray(mapaProfundidad, dtype=cp.float32)
    except Exception:
        depth_cp = None

    # Get ground mask
    ground_params, wall_cfg, door_params = _snapshot_param_groups()
    ground_mask = get_ground(
        mapaProfundidad,
        rays_cp,
        H,
        W,
        ground_params,
        depth_cp=depth_cp,
    )
    with _runtime_lock:
        _runtime["last_ransac_ms"] = get_last_ransac_ms()

    # Reuse common ground params for wall RANSAC, then override wall-specific values
    wall_params = {
        "subsample_stride": ground_params.get("subsample_stride"),
        "min_points": ground_params.get("min_inliers"),
        "max_points": ground_params.get("max_agg_points"),
        "dist_thresh": ground_params.get("dist_thresh"),
        "max_iters": ground_params.get("max_iters"),
        "score_subset": ground_params.get("score_subset"),
        "batch_size": ground_params.get("batch_size"),
        "early_stop_ratio": ground_params.get("early_stop_ratio"),
        "up_axis": ground_params.get("up_axis"),
        "refine_dist_mult": ground_params.get("refine_dist_mult"),
    }
    wall_params.update(WALL_PARAMS_OVERRIDES)
    wall_params["max_up_dot"] = wall_cfg.get(
        "max_up_dot",
        WALL_PARAMS_OVERRIDES.get("max_up_dot", 0.35),
    )
    for key in (
        "ground_perp_deg",
        "wall_ortho_deg",
        "wall_parallel_deg",
        "wall_parallel_distance_m",
    ):
        if key in wall_cfg:
            wall_params[key] = wall_cfg[key]
    try:
        if ground_utils.last_n_cp is not None:
            wall_params["ground_normal"] = ground_utils.last_n_cp
    except Exception:
        pass

    # Get wall planes using the fast plane fitter (no TensorRT)
    wall_mask = None
    door_mask = None
    wall_res = {}
    try:
        wall_res = get_wall_planes(
            mapaProfundidad,
            rays_cp,
            H,
            W,
            wallParams=wall_params,
            ground_mask=ground_mask,
            depth_cp=depth_cp,
        )
        wall_mask = wall_res.get("wall_mask")
    except Exception as e:
        print(f"[segmentar] Wall plane extraction failed: {e}")
        # Continue with only ground mask if wall extraction fails
        wall_mask = np.zeros(ground_mask.shape, dtype=np.uint8)

    # Door detection using TensorRT model (bisenetv2)
    try:
        door_mask = doorDetection(
            imagenRGB,
            hue_tol=door_params.get("door_hue_tol"),
            min_s=door_params.get("door_min_s"),
            min_v=door_params.get("door_min_v"),
            glare_s_max=door_params.get("door_glare_s_max"),
            glare_v_min=door_params.get("door_glare_v_min"),
            glare_v_clip=door_params.get("door_glare_v_clip"),
            depth_m=mapaProfundidad,
            rays=rays_cp,
            visualize_points=True,
        )
    except Exception as e:
        print(f"[segmentar] Door detection failed: {e}")
        door_mask = np.zeros(ground_mask.shape, dtype=np.uint8)

    # Optional: refine/improve masks (after door segmentation)
    if ground_params.get("ground_mask_refine") and ground_mask is not None and np.any(ground_mask):
        try:
            ground_mask = mejorar_mascara_suelo(
                ground_mask,
                imagen_rgb=imagenRGB,
                depth_m=mapaProfundidad,
                rays=rays_cp,
                plane_n=ground_utils.last_n_cp,
                plane_d=ground_utils.last_d_cp,
                dist_thresh=ground_params.get("dist_thresh"),
                use_realsense=use_realsense,
            )
        except Exception as e:
            print(f"[segmentar] Ground mask refinement failed: {e}")

    if wall_cfg.get("wall_mask_refine") and wall_mask is not None and np.any(wall_mask):
        try:
            wall_mask = mejorar_mascara_pared(
                wall_mask,
                imagen_rgb=imagenRGB,
                depth_m=mapaProfundidad,
                rays=rays_cp,
                planes=wall_res.get("planes"),
                dist_thresh=wall_params.get("dist_thresh"),
                use_realsense=use_realsense,
            )
        except Exception as e:
            print(f"[segmentar] Wall mask refinement failed: {e}")

    return apply_mask_to_rgb(imagenRGB, ground_mask, wall_mask, door_mask)


def configurar_tarea(funcion: TareaFuncion, *args: Any, **kwargs: Any) -> None:
    """
    Configure the function that will be executed in the worker thread.
    """
    global _tarea_funcion, _tarea_args, _tarea_kwargs
    _tarea_funcion = funcion
    _tarea_args = args
    _tarea_kwargs = kwargs


def _bucle_hilo() -> None:
    """
    Run the configured task once in a background thread.

    When the task finishes, its result is placed in the shared queue
    (if there is space) and the thread exits.
    """
    global _hilo_trabajador
    if _tarea_funcion is None:
        _hilo_trabajador = None
        return
    try:
        resultado = _tarea_funcion(*_tarea_args, **_tarea_kwargs)
        if resultado is not None:
            while not _detener_evento.is_set():
                try:
                    _resultados.put(resultado, timeout=0.1)
                    break
                except queue.Full:
                    continue
    except Exception as exc:
        # Keep this message in Spanish as it is user-facing debug output
        print(f"[thread] Error en tarea de fondo: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        _hilo_trabajador = None


def obtener_resultado(
    bloqueante: bool = False, timeout: Optional[float] = None
) -> Any:
    """
    Return a result produced by the background task.

    - If bloqueante=False, return immediately:
      * a result, or
      * None if nothing is available.
    - If bloqueante=True, block until a result is available or
      until timeout expires. If the timeout elapses, return None.
    """
    try:
        if bloqueante:
            return _resultados.get(block=True, timeout=timeout)
        return _resultados.get(block=False)
    except queue.Empty:
        return None


def iniciar_hilo_secundario(daemon: bool = True) -> None:
    """
    Create and start the worker thread (if it is not already running).
    """
    global _hilo_trabajador
    if _hilo_trabajador is not None and _hilo_trabajador.is_alive():
        return

    _detener_evento.clear()
    _hilo_trabajador = threading.Thread(
        target=_bucle_hilo,
        name="hilo_tarea_secundaria",
        daemon=daemon,
    )
    _hilo_trabajador.start()


def detener_hilo_secundario(timeout: Optional[float] = 2.0) -> None:
    """
    Request the worker thread to stop and wait for it to finish.
    """
    global _hilo_trabajador
    if _hilo_trabajador is None:
        return

    _detener_evento.set()
    if _hilo_trabajador.is_alive():
        _hilo_trabajador.join(timeout)
    _hilo_trabajador = None


def _lazy_init(
    color_width: int = 640,
    color_height: int = 480,
    depth_width: int = 640,
    depth_height: int = 480,
    fps: int = 30,
    stride: int = 2,
    mode: str = "camera",
) -> None:
    """
    Initialize camera or dataset-related state.

    It is safe to call this repeatedly; if the mode changes
    (camera <-> prueba), the initialization state is reset
    so that the new mode is configured correctly. When switching
    modes the worker thread is stopped (if any), but the camera
    pipeline is not explicitly shut down.
    """
    last_mode = _runtime.get("mode")
    if last_mode != mode:
        # Stop any running worker thread and clear pending results
        detener_hilo_secundario()
        try:
            while not _resultados.empty():
                _resultados.get_nowait()
        except queue.Empty:
            pass
        _runtime["initialized"] = False
        _runtime["last_ransac_ms"] = None
    _runtime["mode"] = mode

    if mode == "prueba":
        # Dataset mode: do not touch the RealSense pipeline; only image/depth/rays.
        _runtime.setdefault("mascara", None)
        # Force recomputation of rays when entering dataset mode
        _runtime["rays_cp"] = None
        return

    # Camera mode: reuse pipeline if already created (do not stop camera).
    pipeline = _runtime.get("pipeline")
    if pipeline is None:
        print("Initializing RealSense camera...")
        pipeline, _ = viewCamera.init_camera(
            color_width,
            color_height,
            depth_width,
            depth_height,
            fps,
            stride,  # subsampling for point cloud
            yaw=-45.0,
            pitch=25.0,
            roll=0.0,
            fov=60.0,
            point_size=1,
        )
        _runtime["pipeline"] = pipeline

    # If coming from dataset mode or rays are not ready, recompute rays
    # and the depth->color aligner for the camera.
    if last_mode != "camera" or _runtime.get("rays_cp") is None:
        rays_np, H, W, align_depth_fn = viewCamera.precompute_rays_for_stream(
            pipeline, viewCamera.rs.stream.color
        )
        _runtime["rays_cp"] = cp.asarray(rays_np)
        _runtime["H"] = H
        _runtime["W"] = W
        _runtime["align_depth_fn"] = align_depth_fn
    _runtime.setdefault("mascara", None)


def _resize_gpu(img, size, interpolation=cv2.INTER_NEAREST):
    """
    Resize helper using cv2.cuda to offload work to GPU.
    """
    gpu = cv2.cuda_GpuMat()
    gpu.upload(img)
    return cv2.cuda.resize(gpu, size, interpolation=interpolation).download()


def preprocesar(
    pipeline=None, mode: str = "camera", dataset_index: Optional[int] = None
) -> bool:
    """
    Extract and store in _runtime the data needed for RANSAC.

    When mode == "prueba", RGB and depth are loaded from
    src/data/{images,depths} instead of the RealSense camera.

    Returns True on success and False if the current frame is invalid.
    """
    if mode == "prueba":
        # Offline mode: load RGB and depth from disk.
        imagenRGB, mapaProfundidad = load_dataset_frame(index=dataset_index)
        if imagenRGB is None or mapaProfundidad is None:
            _runtime["imagenRGB"] = None
            _runtime["mapaProfundidad"] = None
            return False

        # Set H, W from the actual image dimensions.
        H, W = imagenRGB.shape[:2]
        _runtime["H"] = H
        _runtime["W"] = W

        # Ensure depth matches the RGB size.
        if mapaProfundidad.shape[0] != H or mapaProfundidad.shape[1] != W:
            mapaProfundidad = _resize_gpu(
                mapaProfundidad, (W, H), interpolation=cv2.INTER_NEAREST
            )

        # In dataset mode we always use "normalized" rays that are independent
        # of real intrinsics and consistent with a simple Z-up camera model.
        rays_np = viewCamera.compute_normalized_rays(H, W)
        _runtime["rays_cp"] = cp.asarray(rays_np)

    else:
        # Camera mode: use RealSense pipeline and precomputed rays.
        H = _runtime["H"]
        W = _runtime["W"]

        if pipeline is None:
            return False
        try:
            frames = pipeline.wait_for_frames(500)  # timeout ms to avoid blocking forever
        except Exception:
            return False
        align_depth_fn = _runtime["align_depth_fn"]

        # Extract native RGB and depth from the camera
        imagenRGB = viewCamera.extract_rgb(frames)
        mapaProfundidad = (
            align_depth_fn(frames)
            if align_depth_fn is not None
            else viewCamera.extract_depth_meters(frames)
        )

        if imagenRGB is None or mapaProfundidad is None:
            _runtime["imagenRGB"] = None
            _runtime["mapaProfundidad"] = None
            return False

        # Aplicar CLAHE a la imagen RGB (mejora contraste adaptativo)
        lab = cv2.cvtColor(imagenRGB, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        imagenRGB = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Filtro morfológico: dilatar valores válidos hacia inválidos
        invalid_mask = (mapaProfundidad < 0.3) | (mapaProfundidad == 0)
        if np.any(invalid_mask):
            kernel = np.ones((3, 3), dtype=np.uint8)
            depth_temp = mapaProfundidad.copy()
            depth_temp[invalid_mask] = 0
            depth_dilated = cv2.dilate(depth_temp, kernel, iterations=1)
            valid_from_dilation = depth_dilated > 0.3
            mapaProfundidad[invalid_mask & valid_from_dilation] = depth_dilated[invalid_mask & valid_from_dilation]

        # Guided filter: suaviza ruido preservando bordes usando RGB como guía
        mapaProfundidad = cv2.ximgproc.guidedFilter(
            guide=imagenRGB,
            src=mapaProfundidad.astype(np.float32),
            radius=8,
            eps=0.01
        )

        # Ensure shapes match expected HxW (from camera intrinsics).
        if mapaProfundidad.shape[0] != H or mapaProfundidad.shape[1] != W:
            mapaProfundidad = _resize_gpu(
                mapaProfundidad, (W, H), interpolation=cv2.INTER_NEAREST
            )
        if imagenRGB.shape[0] != H or imagenRGB.shape[1] != W:
            imagenRGB = _resize_gpu(
                imagenRGB, (W, H), interpolation=cv2.INTER_AREA
            )

    # Persist current frame data in the runtime dictionary
    _runtime["imagenRGB"] = imagenRGB
    _runtime["mapaProfundidad"] = mapaProfundidad

    return True


def AlgoritmosSegmentacion(
    color_width: int = 640,
    color_height: int = 480,
    depth_width: int = 640,
    depth_height: int = 480,
    fps: int = 30,
    stride: int = 2,
    mode: str = "camera",
    ground_params: Optional[Dict[str, Any]] = None,
    dataset_index: Optional[int] = None,
) -> Any:
    """
    Entry point used by the main loop.

    Reads the latest result from the worker thread, performs preprocessing
    and schedules the next ground-segmentation task in the background.

    mode:
        - "camera" (default): use the RealSense camera.
        - "prueba": use RGB and depth frames from src/data.

    ground_params:
        Optional dictionary to override the current RANSAC/segmentation parameters.
    dataset_index:
        Optional index (0-based) of the dataset frame to load when mode == "prueba".
        If None, frames cycle sequentially as before.
    """

    if ground_params:
        actualizar_parametros_ground(ground_params)

    _lazy_init(color_width, color_height, depth_width, depth_height, fps, stride, mode=mode)

    # Try to obtain a recent result from the worker thread
    resultado = obtener_resultado()
    if resultado is not None or not _runtime["initialized"]:
        # The worker has finished; consume the result and launch a new one
        if resultado is not None:
            _runtime["mascara"] = resultado

        # Ensure that the ground segmentation task is scheduled.
        # If there is any error (invalid frame or exception), retry
        # but bail out after a short budget to avoid blocking the GUI.
        for _ in range(60):  # ~0.6 s budget (sleep 0.01 each)
            try:
                # Get new data for the next task and store it in _runtime
                ok = preprocesar(_runtime["pipeline"], mode=mode, dataset_index=dataset_index)
                imagenRGB = _runtime.get("imagenRGB")
                mapaProfundidad = _runtime.get("mapaProfundidad")
                rays_cp = _runtime.get("rays_cp")

                # On the first frame, use the raw RGB image as a fallback mask
                if _runtime.get("mascara") is None and imagenRGB is not None:
                    _runtime["mascara"] = imagenRGB

                # Start ground segmentation task if we have valid frame data
                if ok and imagenRGB is not None and mapaProfundidad is not None and rays_cp is not None:
                    configurar_tarea(segmentar)
                    iniciar_hilo_secundario()
                    break

                # If data is not valid yet, wait a bit and retry
                time.sleep(0.01)
            except Exception:
                # Retry until it works, but don't spin forever
                time.sleep(0.01)
        else:
            # If we exhausted retries, return last known mask or None
            print("[segmentar] No se pudo preparar frame (modo:", mode, ")")
            return _runtime.get("mascara")

        # Mark initialization as complete
        _runtime["initialized"] = True
        # If there is no new data, keep the last mask
        return _runtime["mascara"]

    # If there is no result, show the last known mask or the current RGB image
    if _runtime["mascara"] is not None:
        return _runtime["mascara"]

    if mode == "prueba":
        # In dataset mode, just return the latest RGB frame from disk
        ok = preprocesar(_runtime["pipeline"], mode=mode, dataset_index=dataset_index)
        if ok and _runtime.get("imagenRGB") is not None:
            return _runtime["imagenRGB"]
    else:
        if _runtime["pipeline"] is not None:
            imagenRGB = viewCamera.extract_rgb(_runtime["pipeline"].wait_for_frames())
            if imagenRGB is not None:
                return imagenRGB

    return None

def liberar_recursos() -> None:
    """
    Stop worker thread and camera pipeline (if any) for a clean shutdown.
    """
    detener_hilo_secundario()

    pipeline = _runtime.get("pipeline")
    if pipeline is not None:
        try:
            pipeline.stop()
        except Exception as exc:
            print(f"[segmentar] Error al detener pipeline: {exc}")

    # Reset basic state for the next run
    _runtime["pipeline"] = None
    _runtime["initialized"] = False
    _runtime["align_depth_fn"] = None
    _runtime["rays_cp"] = None
    _runtime["last_ransac_ms"] = None

    try:
        while not _resultados.empty():
            _resultados.get_nowait()
    except queue.Empty:
        pass
