"""
Lightweight segmentation facade for opening the GUI without running algorithms.

This module intentionally avoids importing the real segmentation stack
(camera, CuPy, TensorRT, RealSense helpers). It exposes the same methods used
by GUI.py, but returns placeholder state so the interface can be opened and
configured safely on machines without the embedded runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _PROJECT_ROOT / "config" / "segmentar_defaults.json"


def _flatten_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert config/segmentar_defaults.json sections into one flat parameter dict.
    """
    params: Dict[str, Any] = {}
    for section in ("groundParams", "wallParams", "doorParams", "wallParamsOverrides"):
        section_data = config.get(section, {})
        if isinstance(section_data, dict):
            params.update(section_data)
    return params


def _load_default_params() -> Dict[str, Any]:
    """
    Load GUI defaults without importing the real segmentation implementation.
    """
    try:
        with _CONFIG_PATH.open("r", encoding="utf-8") as file:
            loaded = json.load(file)
        if isinstance(loaded, dict):
            return _flatten_config(loaded)
    except Exception as exc:
        print(f"[segment_v2] no se pudieron leer defaults: {exc}")
    return {
        "subsample_stride": 2,
        "dist_thresh": 0.03,
        "max_iters": 300,
        "min_inliers": 400,
        "max_angle_deg": 60.0,
        "score_subset": 2048,
        "early_stop_ratio": 0.9,
        "batch_size": 512,
        "wall_subsample_stride": 2,
        "wall_dist_thresh": 0.03,
        "wall_max_iters": 300,
        "wall_min_inliers": 400,
        "wall_max_angle_deg": 20.0,
        "wall_score_subset": 2048,
        "wall_early_stop_ratio": 0.9,
        "wall_batch_size": 512,
        "door_hsv_enabled": True,
        "door_hue_tol": 18,
        "door_min_s": 50,
        "door_min_v": 20,
    }


class SegmentacionDummy:
    """
    API-compatible no-op segmentation object for GUI-only execution.
    """

    def __init__(self) -> None:
        self._params: Dict[str, Any] = _load_default_params()
        self._metrics: Dict[str, Any] = {
            "last_frame_ms": None,
            "last_ransac_ms": None,
            "dataset_filename": None,
            "class_metrics": {
                "ground": {"iou": None, "dice": None, "precision": None},
                "wall": {"iou": None, "dice": None, "precision": None},
                "door": {"iou": None, "dice": None, "precision": None},
            },
            "segmentation_enabled": False,
        }

    def algoritmos_segmentacion(
        self,
        color_width: int = 640,
        color_height: int = 480,
        depth_width: int = 640,
        depth_height: int = 480,
        fps: int = 30,
        stride: int = 2,
        mode: str = "camera",
        ground_params: Optional[Dict[str, Any]] = None,
        dataset_index: Optional[int] = None,
        **_: Any,
    ) -> Any:
        """
        Keep GUI compatibility but do not capture frames or run segmentation.
        """
        _ = (
            color_width,
            color_height,
            depth_width,
            depth_height,
            fps,
            stride,
            mode,
            ground_params,
            dataset_index,
        )
        self._metrics["segmentation_enabled"] = False
        self._metrics["last_frame_ms"] = None
        return None

    def segmentar(self, frame_started_at: Optional[float] = None) -> Any:
        _ = frame_started_at
        return None

    def preprocesar(
        self,
        pipeline: Any = None,
        mode: str = "camera",
        dataset_index: Optional[int] = None,
    ) -> bool:
        _ = (pipeline, mode, dataset_index)
        return False

    def inicializar(self, *args: Any, **kwargs: Any) -> None:
        _ = (args, kwargs)
        return None

    def configurar_tarea(self, *args: Any, **kwargs: Any) -> None:
        _ = (args, kwargs)
        return None

    def obtener_resultado(
        self,
        bloqueante: bool = False,
        timeout: Optional[float] = None,
    ) -> Any:
        _ = (bloqueante, timeout)
        return None

    def iniciar_hilo_secundario(self, daemon: bool = True) -> None:
        _ = daemon
        return None

    def detener_hilo_secundario(self, timeout: Optional[float] = 2.0) -> None:
        _ = timeout
        return None

    def liberar_recursos(self) -> None:
        return None

    def esta_cargando_modelo_puerta(self) -> bool:
        return False

    def obtener_parametros(self, copy: bool = True) -> Dict[str, Any]:
        return dict(self._params) if copy else self._params

    def actualizar_parametros(self, nuevos_params: Dict[str, Any]) -> Dict[str, Any]:
        if nuevos_params:
            for key, value in nuevos_params.items():
                if value is not None:
                    self._params[key] = value
        return self.obtener_parametros()

    def obtener_metricas(self, copy: bool = True) -> Dict[str, Any]:
        return dict(self._metrics) if copy else self._metrics

    def obtener_mascaras(self, copy: bool = True) -> Dict[str, Any]:
        _ = copy
        return {"ground": None, "wall": None, "door": None}

    def obtener_estado_global(self) -> Dict[str, Any]:
        return {
            "enabled": False,
            "mode": "gui-only",
            "params": self.obtener_parametros(),
            "metrics": self.obtener_metricas(),
        }

    def construir_parametros_muro(
        self,
        ground_params: Dict[str, Any],
        wall_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        params = dict(ground_params or {})
        params.update(wall_cfg or {})
        return params


segmentacion = SegmentacionDummy()


def AlgoritmosSegmentacion(*args: Any, **kwargs: Any) -> Any:
    return segmentacion.algoritmos_segmentacion(*args, **kwargs)


def liberar_recursos() -> None:
    return segmentacion.liberar_recursos()


def actualizar_parametros_ground(nuevos_params: Dict[str, Any]) -> Dict[str, Any]:
    return segmentacion.actualizar_parametros(nuevos_params)


def obtener_parametros_ground() -> Dict[str, Any]:
    return segmentacion.obtener_parametros()


def obtener_metricas(copy: bool = True) -> Dict[str, Any]:
    return segmentacion.obtener_metricas(copy=copy)
