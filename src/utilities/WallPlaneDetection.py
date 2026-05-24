"""
Class facade for wall-plane detection.
"""
import importlib.util
from pathlib import Path
from typing import Any, Dict


_HELPER_PATH = Path(__file__).resolve().parent / "helpers" / "WallPlaneDetection.py"
_HELPER_SPEC = importlib.util.spec_from_file_location(
    "src.utilities.helpers_wall_plane_detection",
    _HELPER_PATH,
)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:
    raise ImportError(f"No se pudo cargar helper de muro: {_HELPER_PATH}")
wall_helpers = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(wall_helpers)


class Muro:
    """Encapsula deteccion de muros y su estado."""

    def __init__(self) -> None:
        self._sincronizar_estado()

    def _sincronizar_estado(self) -> None:
        self.debug_timing = wall_helpers.DEBUG_TIMING

    def detectar(self, *args: Any, **kwargs: Any) -> Any:
        resultado = wall_helpers.get_wall_planes(*args, **kwargs)
        self._sincronizar_estado()
        return resultado

    def obtener_estado_global(self) -> Dict[str, Any]:
        self._sincronizar_estado()
        return {"debug_timing": self.debug_timing}
