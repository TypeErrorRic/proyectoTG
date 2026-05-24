"""
Class facade for wall-plane detection.
"""
import importlib.util
from pathlib import Path
from typing import Any, Dict


class Muro:
    """Encapsula deteccion de muros y su estado."""

    def __init__(self) -> None:
        self._helper = None
        self.debug_timing = False

    def _obtener_helper(self):
        if self._helper is None:
            helper_path = Path(__file__).resolve().parent / "helpers" / "WallPlaneDetection.py"
            spec = importlib.util.spec_from_file_location(
                "src.utilities.helpers_wall_plane_detection",
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
        resultado = wall_helpers.get_wall_planes(*args, **kwargs)
        self._sincronizar_estado()
        return resultado

    def obtener_estado_global(self) -> Dict[str, Any]:
        self._sincronizar_estado()
        return {"debug_timing": self.debug_timing}
