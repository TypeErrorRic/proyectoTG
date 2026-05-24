#!/usr/bin/env python3
"""
Class facade for door detection.
"""
from typing import Any, Dict, Optional

import numpy as np

from src.models.helpers import doorDetection as door_helpers


class Puerta:
    """Encapsula deteccion de puertas y su estado de runtime."""

    def __init__(self) -> None:
        self.img_mean = door_helpers.IMG_MEAN
        self.img_std = door_helpers.IMG_STD
        self._sincronizar_estado()

    def _sincronizar_estado(self) -> None:
        runtime = door_helpers._runtime
        self.model = runtime.get("model")
        self.engine_path = runtime.get("engine_path")
        self.input_size = runtime.get("input_size")
        self.min_area = runtime.get("min_area")
        self.model_loading = runtime.get("model_loading")

    def inicializar(self, engine_path: Optional[str] = None) -> bool:
        listo = door_helpers._lazy_init(engine_path=engine_path)
        self._sincronizar_estado()
        return listo

    def preprocesar(self, rgb_image: np.ndarray) -> np.ndarray:
        return door_helpers._preprocess_inputs(rgb_image)

    def postprocesar(self, output: np.ndarray, original_size: Any) -> np.ndarray:
        return door_helpers._postprocess_outputs(output, original_size)

    def detectar(self, *args: Any, **kwargs: Any) -> np.ndarray:
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
