"""
Class facade for HSV-based door mask refinement.
"""
from typing import Any, Dict, Optional

import numpy as np


class PuertaHSV:
    """Encapsula los parametros y metodos HSV para refinar mascaras de puerta."""

    def __init__(self) -> None:
        self._helper = None
        self.hue_tol = 18
        self.min_s = 30
        self.min_v = 20
        self.glare_s_max = 35
        self.glare_v_min = 210
        self.glare_v_clip = 200

    def _obtener_helper(self):
        if self._helper is None:
            from application.implementations import refinamientoHSV as hsv_helpers

            self._helper = hsv_helpers
            self.hue_tol = hsv_helpers._HUE_TOL
            self.min_s = hsv_helpers._MIN_S
            self.min_v = hsv_helpers._MIN_V
            self.glare_s_max = hsv_helpers._GLARE_S_MAX
            self.glare_v_min = hsv_helpers._GLARE_V_MIN
            self.glare_v_clip = hsv_helpers._GLARE_V_CLIP
        return self._helper

    def resolver_parametros(
        self,
        hue_tol: Optional[int] = None,
        min_s: Optional[int] = None,
        min_v: Optional[int] = None,
        glare_s_max: Optional[int] = None,
        glare_v_min: Optional[int] = None,
        glare_v_clip: Optional[int] = None,
    ) -> Dict[str, int]:
        hsv_helpers = self._obtener_helper()
        return hsv_helpers._resolve_hsv_params(
            hue_tol,
            min_s,
            min_v,
            glare_s_max,
            glare_v_min,
            glare_v_clip,
        )

    def filtrar_componentes_pequenos(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        hsv_helpers = self._obtener_helper()
        return hsv_helpers._filter_small_components(mask, min_area)

    def rellenar_huecos(
        self,
        mask: np.ndarray,
        kernel_size: int = 5,
        bbox: Any = None,
    ) -> np.ndarray:
        hsv_helpers = self._obtener_helper()
        return hsv_helpers._fill_holes(mask, kernel_size, bbox)

    def refinar_mascara(
        self,
        bgr_image: np.ndarray,
        door_mask: np.ndarray,
        min_area: int,
        use_roi: bool = True,
        reduce_glare: bool = True,
        hue_tol: Optional[int] = None,
        min_s: Optional[int] = None,
        min_v: Optional[int] = None,
        glare_s_max: Optional[int] = None,
        glare_v_min: Optional[int] = None,
        glare_v_clip: Optional[int] = None,
    ) -> np.ndarray:
        hsv_helpers = self._obtener_helper()
        return hsv_helpers.refine_door_mask_hsv(
            bgr_image,
            door_mask,
            min_area,
            use_roi=use_roi,
            reduce_glare=reduce_glare,
            hue_tol=hue_tol,
            min_s=min_s,
            min_v=min_v,
            glare_s_max=glare_s_max,
            glare_v_min=glare_v_min,
            glare_v_clip=glare_v_clip,
        )

    def obtener_estado_global(self) -> Dict[str, int]:
        return {
            "hue_tol": self.hue_tol,
            "min_s": self.min_s,
            "min_v": self.min_v,
            "glare_s_max": self.glare_s_max,
            "glare_v_min": self.glare_v_min,
            "glare_v_clip": self.glare_v_clip,
        }


class RefinamientoHSV(PuertaHSV):
    """Nombre de dominio para el refinamiento por color de puertas."""

    @property
    def tolerancia_tono(self) -> int:
        return self.hue_tol

    @property
    def saturacion_minima(self) -> int:
        return self.min_s

    @property
    def valor_minimo(self) -> int:
        return self.min_v

    def refinar_por_color(self, imagen_rgb: Any, mascara: Any) -> Any:
        return self.refinar_mascara(imagen_rgb, mascara, min_area=0)
