"""
Validacion geometrica de mascaras usando profundidad.
"""
from typing import Any, Dict

from src.infrastructure.inference.door_deep import PuertaDeep


class ValidacionProfundidad:
    """Fachada de validacion 3D para regiones candidatas de puerta."""

    def __init__(self) -> None:
        self.distancia_plano = 0.02
        self.porcentaje_inliers = 0.30
        self._deep = PuertaDeep()

    def validar_por_profundidad(self, mascara: Any, mapa_profundidad: Any, rayos: Any) -> Dict[str, Any]:
        return self._deep.puntos_desde_mascaras(
            mascara,
            mascara,
            mapa_profundidad,
            rayos,
            plane_inlier_dist=self.distancia_plano,
            plane_inlier_ratio=self.porcentaje_inliers,
        )

    def generar_nube_puntos(self, mapa_profundidad: Any, rayos: Any) -> Any:
        from src.application.pipeline_utils import geometria

        return geometria.points_from_rays_and_depth(rayos, mapa_profundidad)
