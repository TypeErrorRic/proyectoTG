#!/usr/bin/env python3
"""
Class facade for door detection.
"""
from typing import Any, Dict, Optional

import numpy as np

from src.application.detector import DetectorClase


class Puerta(DetectorClase):
    """Encapsula deteccion de puertas y su estado de runtime."""

    def __init__(self) -> None:
        super().__init__()
        self._helper = None
        self.img_mean = (0.485, 0.456, 0.406)
        self.img_std = (0.229, 0.224, 0.225)
        self.model = None
        self.engine_path = None
        self.input_size = (256, 256)
        self.min_area = 300
        self.model_loading = True

    def _obtener_helper(self):
        if self._helper is None:
            from src.infrastructure.inference.helpers import doorDetection as door_helpers
            self._helper = door_helpers
            self.img_mean = door_helpers.IMG_MEAN
            self.img_std = door_helpers.IMG_STD
            self._sincronizar_estado()
        return self._helper

    def _sincronizar_estado(self) -> None:
        if self._helper is None:
            return
        runtime = self._helper._runtime
        self.model = runtime.get("model")
        self.engine_path = runtime.get("engine_path")
        self.input_size = runtime.get("input_size")
        self.min_area = runtime.get("min_area")
        self.model_loading = runtime.get("model_loading")

    def inicializar(self, engine_path: Optional[str] = None) -> bool:
        door_helpers = self._obtener_helper()
        listo = door_helpers._lazy_init(engine_path=engine_path)
        self._sincronizar_estado()
        return listo

    def preprocesar(self, rgb_image: np.ndarray) -> np.ndarray:
        door_helpers = self._obtener_helper()
        return door_helpers._preprocess_inputs(rgb_image)

    def postprocesar(self, output: np.ndarray, original_size: Any) -> np.ndarray:
        door_helpers = self._obtener_helper()
        return door_helpers._postprocess_outputs(output, original_size)

    def detectar(self, *args: Any, **kwargs: Any) -> np.ndarray:
        door_helpers = self._obtener_helper()
        if len(args) >= 3:
            imagen_rgb, mapa_profundidad, rayos = args[:3]
            parametros = args[3] if len(args) > 3 else kwargs.pop("parametros", None)
            if isinstance(parametros, dict):
                self.actualizar_parametros(parametros)
                kwargs = {**self.parametros, **kwargs}
            kwargs.setdefault("depth_m", mapa_profundidad)
            kwargs.setdefault("rays", rayos)
            call_args = (imagen_rgb,)
        else:
            call_args = args
        resultado = door_helpers.doorDetection(*call_args, **kwargs)
        self.mascara = resultado
        self._sincronizar_estado()
        return resultado

    def modelo_cargando(self) -> bool:
        self._sincronizar_estado()
        return bool(self.model_loading)

    def algoritmo_segmentacion_puerta(self, rgb: Any, profundidad: Any, parametros: Dict[str, Any]) -> Any:
        rayos = parametros.get("rayos") if isinstance(parametros, dict) else None
        return self.detectar(rgb, profundidad, rayos, parametros)

    def obtener_estado_global(self) -> Dict[str, Any]:
        self._sincronizar_estado()
        return {
            "img_mean": self.img_mean,
            "img_std": self.img_std,
            "model": self.model,
            "engine_path": self.engine_path,
            "input_size": self.input_size,
            "min_area": self.min_area,
            "model_loading": self.model_loading,
        }
