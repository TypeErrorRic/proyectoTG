"""
Thread module for Jetson Nano.

Provides a background worker thread that executes a segmentation task
and shares its result with the main thread through a small queue.
"""

# Project libraries
import src.utilities.viewCamera as viewCamera
from utilities.GroundDetection import (
    get_ground,
    get_last_ransac_ms,
    camino_transitable as camino_transitable_detector,
)
import utilities.GroundDetection as ground_utils
from src.utilities.helpers import (
    apply_mask_to_rgb,
    load_dataset_frame,
    mejorar_mascara_pared,
    mejorar_mascara_suelo,
)
from utilities.WallPlaneDetection import (
    get_wall_planes,
    muro as muro_detector,
)
from src.models.doorDetection import (
    doorDetection,
    puerta as puerta_detector,
)
import src.models.doorDetection as door_detection_module

# Runtime libraries
import json
import numpy as np
import cupy as cp
import cv2
import threading
import time
import queue
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Callable, Any, Tuple, Dict, List


_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "segmentar_defaults.json"
_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_DATASET_IMAGES_DIR = _DATA_DIR / "images"
_LABEL_DIRS = {
    "ground": _DATA_DIR / "labels" / "floorGroundTruth",
    "wall": _DATA_DIR / "labels" / "wallGroundTruth",
    "door": _DATA_DIR / "labels" / "doorGrounTruth",
}
_DATASET_EXTS = (".png", ".jpg", ".jpeg")
_GT_CACHE_MAX = 64
_REQUIRED_CONFIG_SECTIONS = (
    "groundParams",
    "wallParams",
    "doorParams",
    "wallParamsOverrides",
)


def _load_segmentation_config() -> Dict[str, Dict[str, Any]]:
    """
    Load segmentation defaults from config/segmentar_defaults.json.
    """
    try:
        with _CONFIG_PATH.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError("la raiz del archivo debe ser un objeto JSON")
    except FileNotFoundError:
        raise RuntimeError(f"[segmentar] No existe archivo de configuracion: {_CONFIG_PATH}") from None
    except Exception as exc:
        raise RuntimeError(f"[segmentar] No se pudo leer config '{_CONFIG_PATH}': {exc}") from exc

    cfg: Dict[str, Dict[str, Any]] = {}
    for section in _REQUIRED_CONFIG_SECTIONS:
        section_data = loaded.get(section)
        if not isinstance(section_data, dict):
            raise RuntimeError(
                f"[segmentar] Seccion requerida faltante o invalida en config: '{section}'"
            )
        cfg[section] = dict(section_data)

    # JSON stores arrays, but the ground detector expects a tuple-like up axis.
    up_axis = cfg["groundParams"].get("up_axis")
    if isinstance(up_axis, (list, tuple)) and len(up_axis) == 3:
        try:
            cfg["groundParams"]["up_axis"] = tuple(float(x) for x in up_axis)
        except Exception as exc:
            raise RuntimeError(
                "[segmentar] groundParams.up_axis debe tener 3 valores numericos"
            ) from exc
    else:
        raise RuntimeError(
            "[segmentar] groundParams.up_axis debe ser una lista de 3 elementos"
        )

    return cfg


_loaded_config = _load_segmentation_config()

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
_gt_mask_cache: "OrderedDict[str, Optional[np.ndarray]]" = OrderedDict()


def _snapshot_param_groups() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Return copies of ground, wall, and door parameters.
    """
    with _runtime_lock:
        ground_params = dict(_runtime.ground_params)
        wall_params = dict(_runtime.wall_params)
        door_params = dict(_runtime.door_params)
    return ground_params, wall_params, door_params


def _obtener_parametros_impl(copy: bool = True) -> Dict[str, Any]:
    """
    Snapshot of the current segmentation parameters (ground + wall + door).
    """
    ground_params, wall_params, door_params = _snapshot_param_groups()
    merged = {**ground_params, **wall_params, **door_params}
    return merged.copy() if copy else merged


def _actualizar_parametros_impl(nuevos_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge new parameter values into the runtime dictionary.

    Returns the updated parameter dict.
    """
    if not nuevos_params:
        return _obtener_parametros_impl()

    with _runtime_lock:
        ground_params = dict(_runtime.ground_params)
        wall_params = dict(_runtime.wall_params)
        door_params = dict(_runtime.door_params)
        for key, value in nuevos_params.items():
            if value is None:
                continue
            if key in _runtime.door_param_keys or key.startswith("door_"):
                door_params[key] = value
            elif (
                key in _runtime.wall_param_keys
                or key.startswith("wall_")
                or key in ("max_up_dot", "ground_perp_deg")
            ):
                wall_params[key] = value
            elif key in _runtime.ground_param_keys:
                ground_params[key] = value
            else:
                ground_params[key] = value
        _runtime.ground_params = ground_params
        _runtime.wall_params = wall_params
        _runtime.door_params = door_params
        merged = {**ground_params, **wall_params, **door_params}
        return merged.copy()


def _dataset_files() -> List[str]:
    files = _runtime.get("dataset_files")
    if files is not None:
        return files
    try:
        files = sorted(
            f.name
            for f in _DATASET_IMAGES_DIR.iterdir()
            if f.is_file() and f.suffix.lower() in _DATASET_EXTS
        )
    except Exception:
        files = []
    _runtime["dataset_files"] = files
    return files


def _resolve_dataset_filename(index: Optional[int]) -> Optional[str]:
    files = _dataset_files()
    if not files:
        return None
    if index is None:
        return _runtime.get("dataset_filename")
    try:
        idx = int(index) % len(files)
    except Exception:
        return None
    return files[idx]


def _resolve_label_mask_path(mask_dir: Path, filename: str) -> Optional[Path]:
    if not filename:
        return None
    direct = mask_dir / filename
    if direct.exists():
        return direct
    stem = Path(filename).stem
    for ext in _DATASET_EXTS:
        candidate = mask_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _load_gt_mask_cached(label_key: str, filename: str) -> Optional[np.ndarray]:
    cache_key = f"{label_key}:{filename}"
    cached = _gt_mask_cache.get(cache_key)
    if cache_key in _gt_mask_cache:
        _gt_mask_cache.move_to_end(cache_key)
        return cached

    folder = _LABEL_DIRS.get(label_key)
    if folder is None:
        return None
    mask_path = _resolve_label_mask_path(folder, filename)
    if mask_path is None:
        _gt_mask_cache[cache_key] = None
    else:
        gt_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        _gt_mask_cache[cache_key] = None if gt_img is None else (gt_img > 0)
    _gt_mask_cache.move_to_end(cache_key)
    while len(_gt_mask_cache) > _GT_CACHE_MAX:
        _gt_mask_cache.popitem(last=False)
    return _gt_mask_cache.get(cache_key)


def _to_bool_mask(mask: Any, shape_hw: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        return None
    if shape_hw is not None and arr.shape[:2] != shape_hw:
        arr = cv2.resize(arr, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return arr > 0


def _empty_class_metrics() -> Dict[str, Dict[str, Optional[float]]]:
    return {
        key: {"iou": None, "dice": None, "precision": None}
        for key in _LABEL_DIRS
    }


def _exclusive_masks_bool(masks: Dict[str, Any], shape_hw: Tuple[int, int]) -> Dict[str, np.ndarray]:
    ground_raw = _to_bool_mask(masks.get("ground"), shape_hw=shape_hw)
    wall_raw = _to_bool_mask(masks.get("wall"), shape_hw=shape_hw)
    door_raw = _to_bool_mask(masks.get("door"), shape_hw=shape_hw)
    if ground_raw is None:
        ground_raw = np.zeros(shape_hw, dtype=bool)
    if wall_raw is None:
        wall_raw = np.zeros(shape_hw, dtype=bool)
    if door_raw is None:
        door_raw = np.zeros(shape_hw, dtype=bool)

    # Same visual priority as overlay: ground > door > wall.
    ground = ground_raw
    door = np.logical_and(door_raw, ~ground_raw)
    wall = np.logical_and(wall_raw, ~(np.logical_or(door_raw, ground_raw)))
    return {"ground": ground, "wall": wall, "door": door}


def calcular_iou(pred_mask: Any, gt_mask: Any) -> Optional[float]:
    gt = _to_bool_mask(gt_mask)
    pred = _to_bool_mask(pred_mask, shape_hw=gt.shape if gt is not None else None)
    if gt is None or pred is None:
        return None
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def calcular_dice(pred_mask: Any, gt_mask: Any) -> Optional[float]:
    gt = _to_bool_mask(gt_mask)
    pred = _to_bool_mask(pred_mask, shape_hw=gt.shape if gt is not None else None)
    if gt is None or pred is None:
        return None
    inter = int(np.logical_and(pred, gt).sum())
    denom = int(pred.sum()) + int(gt.sum())
    if denom == 0:
        return 1.0
    return float(2 * inter) / float(denom)


def calcular_precision(pred_mask: Any, gt_mask: Any) -> Optional[float]:
    gt = _to_bool_mask(gt_mask)
    pred = _to_bool_mask(pred_mask, shape_hw=gt.shape if gt is not None else None)
    if gt is None or pred is None:
        return None
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    denom = tp + fp
    if denom == 0:
        return 1.0 if int(gt.sum()) == 0 else 0.0
    return float(tp) / float(denom)


def _compute_dataset_stats(filename: Optional[str], masks: Dict[str, Any]) -> Dict[str, Any]:
    class_metrics = _empty_class_metrics()
    if not filename:
        return {
            "iou": None,
            "dice": None,
            "precision": None,
            "class_metrics": class_metrics,
        }

    gt_raw: Dict[str, np.ndarray] = {}
    shape_hw: Optional[Tuple[int, int]] = None

    for key in _LABEL_DIRS:
        gt_bool = _load_gt_mask_cached(key, filename)
        if gt_bool is None:
            continue
        gt_raw[key] = gt_bool
        if shape_hw is None:
            shape_hw = gt_bool.shape

    if shape_hw is None:
        return {
            "iou": None,
            "dice": None,
            "precision": None,
            "class_metrics": class_metrics,
        }

    pred_ex = _exclusive_masks_bool(masks, shape_hw=shape_hw)
    gt_ex = _exclusive_masks_bool(gt_raw, shape_hw=shape_hw)

    iou_vals: List[float] = []
    dice_vals: List[float] = []
    prec_vals: List[float] = []
    for key in _LABEL_DIRS:
        pred_mask = pred_ex.get(key)
        gt_mask = gt_ex.get(key)
        iou = calcular_iou(pred_mask, gt_mask)
        dice = calcular_dice(pred_mask, gt_mask)
        prec = calcular_precision(pred_mask, gt_mask)
        if (
            key != "door"
            and iou == 1.0
            and dice == 1.0
            and prec == 1.0
            and (not np.any(gt_mask))
            and (not np.any(pred_mask))
        ):
            iou = None
            dice = None
            prec = None
        class_metrics[key] = {
            "iou": float(iou) if iou is not None else None,
            "dice": float(dice) if dice is not None else None,
            "precision": float(prec) if prec is not None else None,
        }
        if iou is not None:
            iou_vals.append(float(iou))
        if dice is not None:
            dice_vals.append(float(dice))
        if prec is not None:
            prec_vals.append(float(prec))

    if not iou_vals:
        return {
            "iou": None,
            "dice": None,
            "precision": None,
            "class_metrics": class_metrics,
        }

    return {
        "iou": float(sum(iou_vals) / len(iou_vals)),
        "dice": float(sum(dice_vals) / len(dice_vals)),
        "precision": float(sum(prec_vals) / len(prec_vals)),
        "class_metrics": class_metrics,
    }


def _obtener_metricas_impl(copy: bool = True) -> Dict[str, Any]:
    """
    Snapshot of runtime metrics for mode "prueba" and camera.
    """
    with _runtime_lock:
        last_ms = _runtime.get("last_ransac_ms")
        class_metrics = _runtime.get("last_class_metrics") or _empty_class_metrics()
        metrics = {
            "last_ransac_ms": last_ms,
            "last_frame_ms": _runtime.get("last_frame_ms"),
            "iou": _runtime.get("last_iou"),
            "dice": _runtime.get("last_dice"),
            "precision": _runtime.get("last_precision"),
            "dataset_filename": _runtime.get("dataset_filename"),
            "class_metrics": {
                key: dict(values) for key, values in class_metrics.items()
            },
        }
    return metrics.copy() if copy else metrics


def _obtener_mascaras_impl(copy: bool = True) -> Dict[str, Any]:
    """
    Snapshot of the latest raw masks (ground, wall, door).

    Returns a dict with keys: "ground", "wall", "door".
    """
    with _runtime_lock:
        masks = _runtime.get("last_masks") or {}
    if not copy:
        return masks
    out: Dict[str, Any] = {}
    for key, value in masks.items():
        if value is None:
            out[key] = None
        else:
            try:
                out[key] = value.copy()
            except Exception:
                out[key] = np.asarray(value).copy()
    return out


def _segmentar_impl(frame_started_at: Optional[float] = None) -> Any:
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
            _runtime["last_frame_ms"] = None
            _runtime["last_iou"] = None
            _runtime["last_dice"] = None
            _runtime["last_precision"] = None
            _runtime["last_class_metrics"] = _empty_class_metrics()
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

    wall_params = _runtime.construir_parametros_muro(ground_params, wall_cfg)

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
            use_hsv_filter=bool(door_params.get("door_hsv_enabled", True)),
            hue_tol=door_params.get("door_hue_tol"),
            min_s=door_params.get("door_min_s"),
            min_v=door_params.get("door_min_v"),
            glare_s_max=door_params.get("door_glare_s_max"),
            glare_v_min=door_params.get("door_glare_v_min"),
            glare_v_clip=door_params.get("door_glare_v_clip"),
            depth_m=mapaProfundidad,
            rays=rays_cp,
            ground_normal=ground_utils.last_n_cp,
            ground_parallel_deg=door_params.get("door_ground_parallel_deg"),
            plane_inlier_ratio=door_params.get("door_plane_inlier_ratio", 0.70),
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

    stats = {
        "iou": None,
        "dice": None,
        "precision": None,
        "class_metrics": _empty_class_metrics(),
    }
    if _runtime.get("mode") == "prueba":
        stats = _compute_dataset_stats(
            _runtime.get("dataset_filename"),
            {
                "ground": ground_mask,
                "wall": wall_mask,
                "door": door_mask,
            },
        )

    with _runtime_lock:
        _runtime["last_masks"] = {
            "ground": ground_mask,
            "wall": wall_mask,
            "door": door_mask,
        }
        _runtime["last_iou"] = stats.get("iou")
        _runtime["last_dice"] = stats.get("dice")
        _runtime["last_precision"] = stats.get("precision")
        _runtime["last_class_metrics"] = stats.get("class_metrics") or _empty_class_metrics()

    frame_out = apply_mask_to_rgb(imagenRGB, ground_mask, wall_mask, door_mask)
    if frame_started_at is None:
        return frame_out
    return (frame_out, frame_started_at)


def _configurar_tarea_impl(funcion: TareaFuncion, *args: Any, **kwargs: Any) -> None:
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
        frame_started_at = None
        payload = resultado
        if isinstance(resultado, tuple) and len(resultado) == 2:
            payload, frame_started_at = resultado
        if payload is not None:
            while not _detener_evento.is_set():
                try:
                    if frame_started_at is not None:
                        payload_to_queue = (payload, frame_started_at, time.perf_counter())
                    else:
                        payload_to_queue = payload
                    _resultados.put(payload_to_queue, timeout=0.1)
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


def _obtener_resultado_impl(
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


def _iniciar_hilo_secundario_impl(daemon: bool = True) -> None:
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


def _detener_hilo_secundario_impl(timeout: Optional[float] = 2.0) -> None:
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
        _detener_hilo_secundario_impl()
        try:
            while not _resultados.empty():
                _resultados.get_nowait()
        except queue.Empty:
            pass
        _runtime["initialized"] = False
        _runtime["last_ransac_ms"] = None
        _runtime["last_frame_ms"] = None
        _runtime["last_iou"] = None
        _runtime["last_dice"] = None
        _runtime["last_precision"] = None
        _runtime["last_class_metrics"] = _empty_class_metrics()
        _runtime["dataset_filename"] = None
        _gt_mask_cache.clear()
    _runtime["mode"] = mode

    if mode == "prueba":
        # Dataset mode: do not touch the RealSense pipeline; only image/depth/rays.
        _runtime.setdefault("mascara", None)
        # Force recomputation of rays when entering dataset mode
        _runtime["rays_cp"] = None
        return
    _runtime["dataset_filename"] = None

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


def _preprocesar_impl(
    pipeline=None, mode: str = "camera", dataset_index: Optional[int] = None
) -> bool:
    """
    Extract and store in _runtime the data needed for RANSAC.

    When mode == "prueba", RGB and depth are loaded from
    src/data/{images,depths} instead of the RealSense camera.

    Returns True on success and False if the current frame is invalid.
    """
    if mode == "prueba":
        _runtime["dataset_filename"] = _resolve_dataset_filename(dataset_index)
        # Offline mode: load RGB and depth from disk.
        imagenRGB, mapaProfundidad = load_dataset_frame(index=dataset_index)
        if imagenRGB is None or mapaProfundidad is None:
            _runtime["imagenRGB"] = None
            _runtime["mapaProfundidad"] = None
            _runtime["last_iou"] = None
            _runtime["last_dice"] = None
            _runtime["last_precision"] = None
            _runtime["last_class_metrics"] = _empty_class_metrics()
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
        _runtime["dataset_filename"] = None
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

        # Filtro morfologico: dilatar valores validos hacia invalidos
        invalid_mask = (mapaProfundidad < 0.3) | (mapaProfundidad == 0)
        if np.any(invalid_mask):
            kernel = np.ones((3, 3), dtype=np.uint8)
            depth_temp = mapaProfundidad.copy()
            depth_temp[invalid_mask] = 0
            depth_dilated = cv2.dilate(depth_temp, kernel, iterations=1)
            valid_from_dilation = depth_dilated > 0.3
            mapaProfundidad[invalid_mask & valid_from_dilation] = depth_dilated[invalid_mask & valid_from_dilation]

        # Guided filter: suaviza ruido preservando bordes usando RGB como guia
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


def _algoritmos_segmentacion_impl(
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
        _runtime.actualizar_parametros(ground_params)

    _runtime.inicializar(color_width, color_height, depth_width, depth_height, fps, stride, mode=mode)

    # Try to obtain a recent result from the worker thread
    resultado = _runtime.obtener_resultado()
    if resultado is not None or not _runtime["initialized"]:
        # The worker has finished; consume the result and launch a new one
        if resultado is not None:
            frame_with_masks = resultado
            frame_started_at = None
            frame_sent_at = None
            if isinstance(resultado, tuple) and len(resultado) == 3:
                frame_with_masks, frame_started_at, frame_sent_at = resultado
            elif isinstance(resultado, tuple) and len(resultado) == 2:
                frame_with_masks, frame_started_at = resultado
            _runtime["mascara"] = frame_with_masks
            if frame_started_at is not None and frame_sent_at is not None:
                try:
                    _runtime["last_frame_ms"] = max(
                        0.0, (float(frame_sent_at) - float(frame_started_at)) * 1000.0
                    )
                except Exception:
                    _runtime["last_frame_ms"] = None
            elif frame_started_at is not None:
                try:
                    _runtime["last_frame_ms"] = max(
                        0.0, (time.perf_counter() - float(frame_started_at)) * 1000.0
                    )
                except Exception:
                    _runtime["last_frame_ms"] = None
            else:
                _runtime["last_frame_ms"] = None

        # Ensure that the ground segmentation task is scheduled.
        # If there is any error (invalid frame or exception), retry
        # but bail out after a short budget to avoid blocking the GUI.
        for _ in range(60):  # ~0.6 s budget (sleep 0.01 each)
            try:
                # Get new data for the next task and store it in _runtime
                t_preprocess_start = time.perf_counter()
                ok = _runtime.preprocesar(_runtime["pipeline"], mode=mode, dataset_index=dataset_index)
                imagenRGB = _runtime.get("imagenRGB")
                mapaProfundidad = _runtime.get("mapaProfundidad")
                rays_cp = _runtime.get("rays_cp")

                # On the first frame, use the raw RGB image as a fallback mask
                if _runtime.get("mascara") is None and imagenRGB is not None:
                    _runtime["mascara"] = imagenRGB

                # Start ground segmentation task if we have valid frame data
                if ok and imagenRGB is not None and mapaProfundidad is not None and rays_cp is not None:
                    _runtime.configurar_tarea(_runtime.segmentar, t_preprocess_start)
                    _runtime.iniciar_hilo_secundario()
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
        ok = _runtime.preprocesar(_runtime["pipeline"], mode=mode, dataset_index=dataset_index)
        if ok and _runtime.get("imagenRGB") is not None:
            return _runtime["imagenRGB"]
    else:
        if _runtime["pipeline"] is not None:
            imagenRGB = viewCamera.extract_rgb(_runtime["pipeline"].wait_for_frames())
            if imagenRGB is not None:
                return imagenRGB

    return None

def _liberar_recursos_impl() -> None:
    """
    Stop worker thread and camera pipeline (if any) for a clean shutdown.
    """
    _detener_hilo_secundario_impl()

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
    _runtime["last_frame_ms"] = None
    _runtime["last_iou"] = None
    _runtime["last_dice"] = None
    _runtime["last_precision"] = None
    _runtime["last_class_metrics"] = _empty_class_metrics()
    _runtime["dataset_filename"] = None
    _gt_mask_cache.clear()

    try:
        while not _resultados.empty():
            _resultados.get_nowait()
    except queue.Empty:
        pass


class Segmentacion:
    """
    Orquestador de segmentacion y contenedor del estado de ejecucion.
    """

    def __init__(self) -> None:
        self.puerta = puerta_detector
        self.camino_transitable = camino_transitable_detector
        self.muro = muro_detector
        self.loaded_config = _loaded_config
        self.ground_params = dict(self.loaded_config["groundParams"])
        self.wall_params = dict(self.loaded_config["wallParams"])
        self.door_params = dict(self.loaded_config["doorParams"])
        self.ground_param_keys = set(self.loaded_config["groundParams"].keys())
        self.wall_param_keys = set(self.loaded_config["wallParams"].keys())
        self.door_param_keys = set(self.loaded_config["doorParams"].keys())
        self.wall_params_overrides = dict(self.loaded_config["wallParamsOverrides"])
        self.initialized = False
        self.mode = None
        self.pipeline = None
        self.rays_cp = None
        self.H = None
        self.W = None
        self.align_depth_fn = None
        self.imagenRGB = None
        self.mapaProfundidad = None
        self.last_ransac_ms = None
        self.last_frame_ms = None
        self.dataset_filename = None
        self.last_iou = None
        self.last_dice = None
        self.last_precision = None
        self.last_class_metrics = None
        self.dataset_files = None
        self.mascara = None
        self.last_masks = {}

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def setdefault(self, key: str, default: Any = None) -> Any:
        if not hasattr(self, key):
            setattr(self, key, default)
        return getattr(self, key)

    def obtener_estado_global(self) -> Dict[str, Any]:
        return {
            "segment_runtime": self,
            "segment_runtime_lock": _runtime_lock,
            "segment_loaded_config": self.loaded_config,
            "segment_wall_overrides": self.wall_params_overrides,
            "ground_module": ground_utils,
            "door_module_runtime": getattr(door_detection_module, "_runtime", None),
            "view_camera_module": viewCamera,
        }

    def algoritmos_segmentacion(self, *args: Any, **kwargs: Any) -> Any:
        return _algoritmos_segmentacion_impl(*args, **kwargs)

    def segmentar(self, frame_started_at: Optional[float] = None) -> Any:
        return _segmentar_impl(frame_started_at=frame_started_at)

    def preprocesar(
        self,
        pipeline=None,
        mode: str = "camera",
        dataset_index: Optional[int] = None,
    ) -> bool:
        return _preprocesar_impl(pipeline=pipeline, mode=mode, dataset_index=dataset_index)

    def configurar_tarea(self, funcion: TareaFuncion, *args: Any, **kwargs: Any) -> None:
        _configurar_tarea_impl(funcion, *args, **kwargs)

    def obtener_resultado(
        self,
        bloqueante: bool = False,
        timeout: Optional[float] = None,
    ) -> Any:
        return _obtener_resultado_impl(bloqueante=bloqueante, timeout=timeout)

    def iniciar_hilo_secundario(self, daemon: bool = True) -> None:
        _iniciar_hilo_secundario_impl(daemon=daemon)

    def detener_hilo_secundario(self, timeout: Optional[float] = 2.0) -> None:
        _detener_hilo_secundario_impl(timeout=timeout)

    def inicializar(
        self,
        color_width: int = 640,
        color_height: int = 480,
        depth_width: int = 640,
        depth_height: int = 480,
        fps: int = 30,
        stride: int = 2,
        mode: str = "camera",
    ) -> None:
        _lazy_init(
            color_width=color_width,
            color_height=color_height,
            depth_width=depth_width,
            depth_height=depth_height,
            fps=fps,
            stride=stride,
            mode=mode,
        )

    def actualizar_parametros(self, nuevos_params: Dict[str, Any]) -> Dict[str, Any]:
        return _actualizar_parametros_impl(nuevos_params)

    def obtener_parametros(self, copy: bool = True) -> Dict[str, Any]:
        return _obtener_parametros_impl(copy=copy)

    def obtener_metricas(self, copy: bool = True) -> Dict[str, Any]:
        return _obtener_metricas_impl(copy=copy)

    def obtener_mascaras(self, copy: bool = True) -> Dict[str, Any]:
        return _obtener_mascaras_impl(copy=copy)

    def liberar_recursos(self) -> None:
        _liberar_recursos_impl()

    def esta_cargando_modelo_puerta(self) -> bool:
        return self.puerta.modelo_cargando()

    def construir_parametros_muro(
        self,
        ground_params: Dict[str, Any],
        wall_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        wall_defaults = dict(self.loaded_config.get("wallParams", {}) or {})
        wall_params = {
            "subsample_stride": wall_cfg.get(
                "wall_subsample_stride",
                wall_defaults.get("wall_subsample_stride", 2),
            ),
            "min_points": wall_cfg.get(
                "wall_min_inliers",
                wall_defaults.get("wall_min_inliers", 400),
            ),
            "max_points": ground_params.get("max_agg_points"),
            "dist_thresh": wall_cfg.get(
                "wall_dist_thresh",
                wall_defaults.get("wall_dist_thresh", 0.03),
            ),
            "max_iters": wall_cfg.get(
                "wall_max_iters",
                wall_defaults.get("wall_max_iters", 300),
            ),
            "score_subset": wall_cfg.get(
                "wall_score_subset",
                wall_defaults.get("wall_score_subset", 2048),
            ),
            "batch_size": wall_cfg.get(
                "wall_batch_size",
                wall_defaults.get("wall_batch_size", 512),
            ),
            "early_stop_ratio": wall_cfg.get(
                "wall_early_stop_ratio",
                wall_defaults.get("wall_early_stop_ratio", 0.9),
            ),
            "up_axis": ground_params.get("up_axis"),
            "refine_dist_mult": wall_cfg.get(
                "wall_refine_dist_mult",
                wall_defaults.get("wall_refine_dist_mult", 1.6),
            ),
        }
        wall_params.update(self.wall_params_overrides)
        if "wall_max_angle_deg" in wall_cfg:
            wall_params["max_angle_deg"] = wall_cfg["wall_max_angle_deg"]
        if "wall_refine_dist_mult" in wall_cfg:
            wall_params["refine_dist_mult"] = wall_cfg["wall_refine_dist_mult"]
        wall_params["max_up_dot"] = wall_cfg.get(
            "max_up_dot",
            self.wall_params_overrides.get("max_up_dot", 0.35),
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
        return wall_params


_runtime = Segmentacion()
segmentacion = _runtime

