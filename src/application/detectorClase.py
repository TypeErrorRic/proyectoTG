"""
Contrato comun para detectores de clases semanticas.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict


class DetectorClase(ABC):
    """Interfaz base para detectores de suelo, muro y puerta."""

    def __init__(self) -> None:
        self.parametros: Dict[str, Any] = {}
        self.mascara: Any = None
        self.metricas: Dict[str, Any] = {}

    @abstractmethod
    def detectar(self, imagen_rgb: Any, mapa_profundidad: Any, rayos: Any, *args: Any, **kwargs: Any) -> Any:
        """Ejecuta la deteccion y devuelve la mascara o resultado del detector."""

    def actualizar_parametros(self, parametros: Dict[str, Any]) -> None:
        if parametros:
            self.parametros.update(parametros)

    def obtener_mascara(self) -> Any:
        return self.mascara

    def obtener_metricas(self) -> Dict[str, Any]:
        return dict(self.metricas)
