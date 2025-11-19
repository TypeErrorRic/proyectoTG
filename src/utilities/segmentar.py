"""
Thread module for Jetson Nano.

Provides a background worker thread that periodically executes a user-defined
function (the "task") and shares its results with the main thread through a
bounded queue.
"""

# Personal libraries
import src.utilities.viewCamera as viewCamera
from src.utilities.ransacCellingGround import get_ground
from src.utilities.helpers import ColocarMascara

# Runtime libraries
import cupy as cp
import cv2
import threading
import time
import queue
from typing import Optional, Callable, Any, Tuple, Dict

# =======================
# Runtime state and parameters (camera and rays)
# =======================

# Centralized state dictionary to avoid many loose globals
_runtime = {
    "initialized": False,
    "pipeline": None,
    "rays_cp": None,
    "H": None,
    "W": None,
    "align_depth_fn": None,
    "params": None,
    "result_dict": {},
    "fps_t0": None,
    "subsample_stride": None,
    "imagenRGB": None,
    "mapaProfundidad": None,
}

# Default parameters to keep an acceptable FPS
SUBSAMPLE_STRIDE = 4  # 1/stride^2 subsampling for RANSAC

# Event to stop the worker thread cleanly
_detener_evento = threading.Event()
_hilo_trabajador: Optional[threading.Thread] = None

# Result queue shared with the main thread (buffer of 1 element).
# The worker thread blocks when there is already a pending result and
# only runs the task again after the main thread has consumed it.
_resultados: "queue.Queue[Any]" = queue.Queue(maxsize=1)

# Type of the function that will be executed in the worker thread.
# It must return the "result" that will be shared with the main thread.
TareaFuncion = Callable[..., Any]

# Reference to the function that will run in the worker
_tarea_funcion: Optional[TareaFuncion] = None
# Arguments with which the function will be called in the worker
_tarea_args: Tuple[Any, ...] = ()
_tarea_kwargs: Dict[str, Any] = {}


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
        # Only the worker thread inserts results into the queue.
        # If there is already a pending result, it blocks until
        # the main thread consumes it.
        if resultado is not None:
            while not _detener_evento.is_set():
                try:
                    _resultados.put(resultado, timeout=0.1)
                    break
                except queue.Full:
                    # Wait until the main thread consumes the data
                    continue
    except Exception as exc:
        # You can replace this print with logging if you prefer
        print(f"[thread] Error en tarea de fondo: {exc}")
    finally:
        # Mark thread as finished
        _hilo_trabajador = None


def obtener_resultado(
    bloqueante: bool = False, timeout: Optional[float] = None
) -> Any:
    """
    Return a result produced by the background task.

    - If `bloqueante=False` (default), return immediately:
      * a result, or
      * None if nothing is available.
    - If `bloqueante=True`, block until a result is available or
      until `timeout` expires (if provided). If the timeout elapses,
      return None.
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
) -> None:
    """Initialize camera and rays on the first call."""
    if _runtime["initialized"]:
        return
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
    rays_np, H, W, align_depth_fn = viewCamera.precompute_rays_for_stream(
        pipeline, viewCamera.rs.stream.color
    )
    _runtime["pipeline"] = pipeline
    _runtime["rays_cp"] = cp.asarray(rays_np)
    _runtime["H"] = H
    _runtime["W"] = W
    _runtime["align_depth_fn"] = align_depth_fn
    _runtime["params"] = params
    _runtime["fps_t0"] = time.time()
    _runtime.setdefault("algoritmo", 1)
    _runtime.setdefault("mascara", None)
    # Use the stride from params for RANSAC subsampling if it is provided
    try:
        _runtime["subsample_stride"] = int(params.get("stride", SUBSAMPLE_STRIDE))
    except Exception:
        _runtime["subsample_stride"] = SUBSAMPLE_STRIDE


def preprocesar(pipeline=None) -> bool:
    """
    Extract and store in _runtime the data needed for RANSAC.

    Returns True on success and False if the current frame is invalid.
    """
    frames = pipeline.wait_for_frames()
    H = _runtime["H"]
    W = _runtime["W"]
    align_depth_fn = _runtime["align_depth_fn"]

    # Extract native RGB and depth
    imagenRGB = viewCamera.extract_rgb(frames)
    mapaProfundidad = (
        align_depth_fn(frames)
        if align_depth_fn is not None
        else viewCamera.extract_depth_meters(frames)
    )
    if imagenRGB is None or mapaProfundidad is None:
        # Clear stored frame data on invalid capture
        _runtime["imagenRGB"] = None
        _runtime["mapaProfundidad"] = None
        return False

    # Ensure depth shape matches COLOR size
    if mapaProfundidad.shape[0] != H or mapaProfundidad.shape[1] != W:
        mapaProfundidad = cv2.resize(
            mapaProfundidad, (W, H), interpolation=cv2.INTER_NEAREST
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
) -> Any:
    """
    Read the result from the worker thread, perform preprocessing
    and schedule the next segmentation algorithm (1 floor,
    2 wall, 3 door) for the worker thread.
    """

    _lazy_init(color_width, color_height, depth_width, depth_height, fps, stride)

    # Try to obtain a recent result from the worker thread
    resultado = obtener_resultado()
    if resultado is not None or not _runtime["initialized"]:
        # The worker has finished; consume the result and launch a new one
        if resultado is not None:
            _runtime["mascara"] = resultado
        algoritmo = _runtime.get("algoritmo", 1)
        if algoritmo == 1:
            # Ensure that the floor segmentation task is scheduled.
            # If there is any error (invalid frame or exception), retry
            # until it succeeds.
            while True:
                try:
                    # Get new data for the next task and store it in _runtime
                    ok = preprocesar(_runtime["pipeline"])
                    imagenRGB = _runtime.get("imagenRGB")
                    mapaProfundidad = _runtime.get("mapaProfundidad")
                    rays_cp = _runtime.get("rays_cp")
                    H = _runtime.get("H")
                    W = _runtime.get("W")

                    # On the first frame, use the raw RGB image as a fallback mask
                    if _runtime.get("mascara") is None and imagenRGB is not None:
                        _runtime["mascara"] = imagenRGB

                    # Start floor segmentation task if we have valid frame data
                    if ok and imagenRGB is not None and mapaProfundidad is not None and rays_cp is not None:
                        configurar_tarea(
                            get_ground,
                            imagenRGB,
                            mapaProfundidad,
                            rays_cp,
                            H,
                            W,
                            _runtime["subsample_stride"],
                        )
                        iniciar_hilo_secundario()
                        # Change to the next algorithm
                        _runtime["algoritmo"] = 2
                        break

                    # If data is not valid yet, wait a bit and retry
                    time.sleep(0.01)
                except Exception as exc:
                    # Retry until it works
                    time.sleep(0.01)
        elif algoritmo == 2:
            # Ensure that the secondary segmentation task is scheduled.
            # If there is any error (invalid frame or exception), retry
            # until it succeeds.
            while True:
                try:
                    configurar_tarea(ColocarMascara, _runtime["mascara"])
                    iniciar_hilo_secundario()
                    # Change back to the first algorithm
                    _runtime["algoritmo"] = 1
                    break
                except Exception as exc:
                    # Retry until it works
                    time.sleep(0.01)
        # Mark initialization as complete
        _runtime["initialized"] = True
        # If there is no new data, keep the last mask
        return ColocarMascara(_runtime["mascara"])
    else:
        # If there is no result, show the last known mask or the current RGB image
        if _runtime["mascara"] is not None:
            return ColocarMascara(_runtime["mascara"])
        imagenRGB = viewCamera.extract_rgb(_runtime["pipeline"].wait_for_frames())
        if imagenRGB is not None:
            return ColocarMascara(imagenRGB)
        return None
