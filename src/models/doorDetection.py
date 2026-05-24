#!/usr/bin/env python3
"""
Class facade for door detection.
"""
from typing import Any, Dict, Optional

import numpy as np


class Puerta:
    """Encapsula deteccion de puertas y su estado de runtime."""

    def __init__(self) -> None:
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
            from src.models.helpers import doorDetection as door_helpers
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
        resultado = door_helpers.doorDetection(*args, **kwargs)
        self._sincronizar_estado()
        return resultado

    def modelo_cargando(self) -> bool:
        self._sincronizar_estado()
        return bool(self.model_loading)

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
