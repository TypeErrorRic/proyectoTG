"""
Class facade for ground/caminotransitable detection.
"""
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional

from application.detectorClase import DetectorClase


class CaminoTransitable(DetectorClase):
    """Encapsula deteccion de suelo/camino transitable y su estado."""

    def __init__(self) -> None:
        super().__init__()
        self._helper = None
        self.debug_timing = False
        self.last_ransac_ms = None
        self.last_n_cp = None
        self.last_d_cp = None
        self.debug_ransac_times = []
        self.debug_ransac_counter = 0

    def _obtener_helper(self):
        if self._helper is None:
            helper_path = Path(__file__).resolve().parent / "helpers" / "caminoTransitable.py"
            spec = importlib.util.spec_from_file_location(
                "src.application.helpers_ground_detection",
                helper_path,
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"No se pudo cargar helper de camino transitable: {helper_path}")
            helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helper)
            self._helper = helper
            self._sincronizar_estado()
        return self._helper

    def _sincronizar_estado(self) -> None:
        if self._helper is None:
            return
        self.debug_timing = self._helper.DEBUG_TIMING
        self.last_ransac_ms = self._helper._last_ransac_ms
        self.last_n_cp = self._helper.last_n_cp
        self.last_d_cp = self._helper.last_d_cp
        self.debug_ransac_times = self._helper._debug_ransac_times
        self.debug_ransac_counter = self._helper._debug_ransac_counter
        self.mascara = getattr(self._helper, "last_mask", self.mascara)
        self.metricas = {
            "last_ransac_ms": self.last_ransac_ms,
            "last_n_cp": self.last_n_cp,
            "last_d_cp": self.last_d_cp,
        }

    def detectar(self, *args: Any, **kwargs: Any) -> Any:
        ground_helpers = self._obtener_helper()
        if len(args) >= 5 or (len(args) >= 4 and not isinstance(args[3], dict)):
            call_args = args
        elif len(args) >= 3:
            _imagen_rgb, mapa_profundidad, rayos = args[:3]
            parametros = args[3] if len(args) > 3 else kwargs.pop("parametros", None)
            if parametros is not None:
                self.actualizar_parametros(parametros)
            h, w = mapa_profundidad.shape[:2]
            call_args = (mapa_profundidad, rayos, h, w, self.parametros)
        else:
            call_args = args
        resultado = ground_helpers.get_ground(*call_args, **kwargs)
        self.mascara = resultado
        self._sincronizar_estado()
        return resultado

    def estimar_plano_suelo(self, *args: Any, **kwargs: Any) -> Any:
        return self.ajustar_plano_ransac(*args, **kwargs)

    def ajustar_plano_ransac(self, *args: Any, **kwargs: Any) -> Any:
        ground_helpers = self._obtener_helper()
        resultado = ground_helpers.ransac_plane_gpu(*args, **kwargs)
        self._sincronizar_estado()
        return resultado

    def refinar_plano(self, *args: Any, **kwargs: Any) -> Any:
        ground_helpers = self._obtener_helper()
        resultado = ground_helpers._refine_plane(*args, **kwargs)
        self._sincronizar_estado()
        return resultado

    def obtener_tiempo_ransac_ms(self) -> Optional[float]:
        self._sincronizar_estado()
        return None if self.last_ransac_ms is None else float(self.last_ransac_ms)

    def algoritmo_segmentacion_suelo(self, rgb: Any, profundidad: Any, parametros: Dict[str, Any]) -> Any:
        rayos = parametros.get("rayos") if isinstance(parametros, dict) else None
        return self.detectar(rgb, profundidad, rayos, parametros)

    def obtener_estado_global(self) -> Dict[str, Any]:
        self._sincronizar_estado()
        return {
            "debug_timing": self.debug_timing,
            "last_ransac_ms": self.last_ransac_ms,
            "last_n_cp": self.last_n_cp,
            "last_d_cp": self.last_d_cp,
            "debug_ransac_times": self.debug_ransac_times,
            "debug_ransac_counter": self.debug_ransac_counter,
        }
