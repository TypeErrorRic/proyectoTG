"""
Class facade for wall-plane detection.
"""
import importlib.util
from pathlib import Path
from typing import Any, Dict

from application.detectorClase import DetectorClase


class Muro(DetectorClase):
    """Encapsula deteccion de muros y su estado."""

    def __init__(self) -> None:
        super().__init__()
        self._helper = None
        self.debug_timing = False
        self.planos_verticales = []
        self.normal_suelo = None

    def _obtener_helper(self):
        if self._helper is None:
            helper_path = Path(__file__).resolve().parent / "implementations" / "muro.py"
            spec = importlib.util.spec_from_file_location(
                "src.application.implementations_wall_plane_detection",
                helper_path,
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"No se pudo cargar helper de muro: {helper_path}")
            helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helper)
            self._helper = helper
            self._sincronizar_estado()
        return self._helper

    def _sincronizar_estado(self) -> None:
        if self._helper is None:
            return
        self.debug_timing = self._helper.DEBUG_TIMING

    def detectar(self, *args: Any, **kwargs: Any) -> Any:
        wall_helpers = self._obtener_helper()
        if len(args) >= 5 or (len(args) >= 4 and not isinstance(args[3], dict)):
            call_args = args
        elif len(args) >= 3:
            _imagen_rgb, mapa_profundidad, rayos = args[:3]
            parametros = args[3] if len(args) > 3 else kwargs.pop("parametros", None)
            if parametros is not None:
                self.actualizar_parametros(parametros)
            h, w = mapa_profundidad.shape[:2]
            call_args = (mapa_profundidad, rayos, h, w)
            kwargs.setdefault("wallParams", self.parametros)
        else:
            call_args = args
        resultado = wall_helpers.get_wall_planes(*call_args, **kwargs)
        if isinstance(resultado, dict):
            self.mascara = resultado.get("wall_mask")
            self.planos_verticales = resultado.get("planes") or []
        else:
            self.mascara = resultado
        self._sincronizar_estado()
        return resultado

    def estimar_planos_verticales(self, *args: Any, **kwargs: Any) -> Any:
        return self.detectar(*args, **kwargs)

    def filtrar_con_suelo(self, resultado: Any) -> Any:
        return resultado

    def algoritmo_segmentacion_muro(self, rgb: Any, profundidad: Any, parametros: Dict[str, Any]) -> Any:
        rayos = parametros.get("rayos") if isinstance(parametros, dict) else None
        return self.detectar(rgb, profundidad, rayos, parametros)

    def obtener_estado_global(self) -> Dict[str, Any]:
        self._sincronizar_estado()
        return {
            "debug_timing": self.debug_timing,
            "planos_verticales": self.planos_verticales,
            "normal_suelo": self.normal_suelo,
        }
