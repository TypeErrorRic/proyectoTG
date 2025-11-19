"""
Thread module for Jetson Nano.

Provides a background worker thread that executes a segmentation task
and shares its result with the main thread through a small queue.
"""

# Personal libraries
import src.utilities.viewCamera as viewCamera
from src.utilities.ransacCellingGround import get_ground
from src.utilities.helpers import apply_mask_to_rgb, load_dataset_frame

# Runtime libraries
import cupy as cp
import numpy as np
import cv2
import threading
import time
import queue
from typing import Optional, Callable, Any, Tuple, Dict

# Centralized state dictionary to avoid many loose globals
_runtime: Dict[str, Any] = {
    "initialized": False,
    "mode": None,
    "pipeline": None,
    "rays_cp": None,
    "H": None,
    "W": None,
    "align_depth_fn": None,
    "params": None,
    "result_dict": {},
    "imagenRGB": None,
    "mapaProfundidad": None,
    "groundParams": {
        "dist_thresh": 0.03,
        "max_iters": 500,
        "min_inliers": 600,
        "subsample_stride": 4,
        "time_budget_ms": 50,
        "up_axis": (0.0, -1.0, 0.0),
        "max_angle_deg": 45.0,
        "seed": 42,
        "score_subset": 2048,
        "orientation": "ground",
        "early_stop_ratio": 0.92,
        "batch_size": 256,
    },
}

# Event to stop the worker thread cleanly
_detener_evento = threading.Event()
_hilo_trabajador: Optional[threading.Thread] = None

# Result queue shared with the main thread (buffer of 1 element).
_resultados: "queue.Queue[Any]" = queue.Queue(maxsize=1)

# Type of the function that will be executed in the worker thread.
TareaFuncion = Callable[..., Any]

# Reference to the function that will run in the worker
_tarea_funcion: Optional[TareaFuncion] = None
_tarea_args: Tuple[Any, ...] = ()
_tarea_kwargs: Dict[str, Any] = {}


def segmentar() -> Any:
    """
    Ground segmentation worker.

    Uses the current frame and parameters stored in _runtime to detect
    the ground plane and returns the RGB image with the ground mask
    overlaid.
    """
    ground = get_ground(
        _runtime["mapaProfundidad"],
        _runtime["rays_cp"],
        _runtime["H"],
        _runtime["W"],
        _runtime["groundParams"],
    )
    return apply_mask_to_rgb(_runtime["imagenRGB"], ground)


def configurar_tarea(funcion: TareaFuncion, *args: Any, **kwargs: Any) -> None:
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
        print(f"[thread] Error en tarea de fondo: {exc}")
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
    _runtime["mode"] = mode

    if mode == "prueba":
        # Dataset mode: do not touch the RealSense pipeline; H/W and rays
        # are derived later in `preprocesar` from the dataset images.
        _runtime["params"] = {"stride": stride, "mode": mode}
        _runtime.setdefault("mascara", None)
        return

    # Camera mode: reuse pipeline if already created (do not stop camera).
    pipeline = _runtime.get("pipeline")
    if pipeline is None:
        print("Initializing RealSense camera...")
        pipeline, params = viewCamera.init_camera(
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
        _runtime["params"] = params

    # If coming from dataset mode or rays are not ready, recompute rays
    # and depth->color aligner for the camera.
    if last_mode != "camera" or _runtime.get("rays_cp") is None:
        rays_np, H, W, align_depth_fn = viewCamera.precompute_rays_for_stream(
            pipeline, viewCamera.rs.stream.color
        )
        _runtime["rays_cp"] = cp.asarray(rays_np)
        _runtime["H"] = H
        _runtime["W"] = W
        _runtime["align_depth_fn"] = align_depth_fn
    _runtime.setdefault("mascara", None)


def preprocesar(pipeline=None, mode: str = "camera") -> bool:
    """
    Extract and store in _runtime the data needed for RANSAC.

    When mode == "prueba", RGB and depth are loaded from
    src/data/{images,depths} instead of the RealSense camera.

    Returns True on success and False if the current frame is invalid.
    """
    if mode == "prueba":
        # Offline mode: load RGB and depth from disk.
        imagenRGB, mapaProfundidad = load_dataset_frame()
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
            mapaProfundidad = cv2.resize(
                mapaProfundidad, (W, H), interpolation=cv2.INTER_NEAREST
            )

        # Compute rays based on image size if needed (no alignment required).
        rays_cp = _runtime.get("rays_cp")
        if rays_cp is None or rays_cp.shape[0] != H or rays_cp.shape[1] != W:
            rays_np = viewCamera.compute_normalized_rays(H, W)
            _runtime["rays_cp"] = cp.asarray(rays_np)

    else:
        # Camera mode: use RealSense pipeline and precomputed rays.
        H = _runtime["H"]
        W = _runtime["W"]

        if pipeline is None:
            return False
        frames = pipeline.wait_for_frames()
        align_depth_fn = _runtime["align_depth_fn"]
        # Extract native RGB and depth from camera
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

        # Ensure shapes match expected HxW (from camera intrinsics)
        if mapaProfundidad.shape[0] != H or mapaProfundidad.shape[1] != W:
            mapaProfundidad = cv2.resize(
                mapaProfundidad, (W, H), interpolation=cv2.INTER_NEAREST
            )
        if imagenRGB.shape[0] != H or imagenRGB.shape[1] != W:
            imagenRGB = cv2.resize(imagenRGB, (W, H), interpolation=cv2.INTER_AREA)

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
) -> Any:
    """
    Read the result from the worker thread, perform preprocessing
    and schedule the next segmentation algorithm (ground) for
    the worker thread.

    mode:
        - "camera" (por defecto): usa la RealSense.
        - "prueba": usa imágenes y depth desde src/data.
    """

    _lazy_init(color_width, color_height, depth_width, depth_height, fps, stride, mode=mode)

    # Try to obtain a recent result from the worker thread
    resultado = obtener_resultado()
    if resultado is not None or not _runtime["initialized"]:
        # The worker has finished; consume the result and launch a new one
        if resultado is not None:
            _runtime["mascara"] = resultado

        # Ensure that the floor segmentation task is scheduled.
        # If there is any error (invalid frame or exception), retry
        # until it succeeds.
        while True:
            try:
                # Get new data for the next task and store it in _runtime
                ok = preprocesar(_runtime["pipeline"], mode=mode)
                imagenRGB = _runtime.get("imagenRGB")
                mapaProfundidad = _runtime.get("mapaProfundidad")
                rays_cp = _runtime.get("rays_cp")

                # On the first frame, use the raw RGB image as a fallback mask
                if _runtime.get("mascara") is None and imagenRGB is not None:
                    _runtime["mascara"] = imagenRGB

                # Start floor segmentation task if we have valid frame data
                if ok and imagenRGB is not None and mapaProfundidad is not None and rays_cp is not None:
                    configurar_tarea(segmentar)
                    iniciar_hilo_secundario()
                    break

                # If data is not valid yet, wait a bit and retry
                time.sleep(0.01)
            except Exception:
                # Retry until it works
                time.sleep(0.01)

        # Mark initialization as complete
        _runtime["initialized"] = True
        # If there is no new data, keep the last mask
        return _runtime["mascara"]

    # If there is no result, show the last known mask or the current RGB image
    if _runtime["mascara"] is not None:
        return _runtime["mascara"]

    if mode == "prueba":
        # In dataset mode, just return the latest RGB frame from disk
        ok = preprocesar(_runtime["pipeline"], mode=mode)
        if ok and _runtime.get("imagenRGB") is not None:
            return _runtime["imagenRGB"]
    else:
        if _runtime["pipeline"] is not None:
            imagenRGB = viewCamera.extract_rgb(_runtime["pipeline"].wait_for_frames())
            if imagenRGB is not None:
                return imagenRGB

    return None

