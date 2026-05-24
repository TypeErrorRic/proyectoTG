"""
Class facade for ground/caminotransitable detection.
"""
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional


_HELPER_PATH = Path(__file__).resolve().parent / "helpers" / "GroundDetection.py"
_HELPER_SPEC = importlib.util.spec_from_file_location(
    "src.utilities.helpers_ground_detection",
    _HELPER_PATH,
)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:
    raise ImportError(f"No se pudo cargar helper de camino transitable: {_HELPER_PATH}")
ground_helpers = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(ground_helpers)


class CaminoTransitable:
    """Encapsula deteccion de suelo/camino transitable y su estado."""

    def __init__(self) -> None:
        self._sincronizar_estado()

    def _sincronizar_estado(self) -> None:
        self.debug_timing = ground_helpers.DEBUG_TIMING
        self.last_ransac_ms = ground_helpers._last_ransac_ms
        self.last_n_cp = ground_helpers.last_n_cp
        self.last_d_cp = ground_helpers.last_d_cp
        self.debug_ransac_times = ground_helpers._debug_ransac_times
        self.debug_ransac_counter = ground_helpers._debug_ransac_counter

    def detectar(self, *args: Any, **kwargs: Any) -> Any:
        resultado = ground_helpers.get_ground(*args, **kwargs)
        self._sincronizar_estado()
        return resultado

    def ajustar_plano_ransac(self, *args: Any, **kwargs: Any) -> Any:
        resultado = ground_helpers.ransac_plane_gpu(*args, **kwargs)
        self._sincronizar_estado()
        return resultado

    def refinar_plano(self, *args: Any, **kwargs: Any) -> Any:
        resultado = ground_helpers._refine_plane(*args, **kwargs)
        self._sincronizar_estado()
        return resultado

    def obtener_tiempo_ransac_ms(self) -> Optional[float]:
        self._sincronizar_estado()
        return None if self.last_ransac_ms is None else float(self.last_ransac_ms)

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
