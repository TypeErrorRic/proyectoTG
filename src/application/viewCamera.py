"""
Class facades for RealSense camera utilities.
"""
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional


class ViewCameraHelperMixin:
    """Carga diferida del helper funcional de camara."""

    _helper = None

    @classmethod
    def helper(cls):
        if cls._helper is None:
            helper_path = Path(__file__).resolve().parent / "helpers" / "viewCamera.py"
            spec = importlib.util.spec_from_file_location(
                "src.application.helpers_view_camera",
                helper_path,
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"No se pudo cargar helper de camara: {helper_path}")
            helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helper)
            cls._helper = helper
        return cls._helper


class DepthToColorAlignerGPU(ViewCameraHelperMixin):
    """Fachada para el alineador GPU depth->color."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._impl = self.helper().DepthToColorAlignerGPU(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._impl, name)

    def align(self, depth_m):
        return self._impl.align(depth_m)


class Camara(ViewCameraHelperMixin):
    """Fachada POO para utilidades de RealSense."""

    def __init__(
        self,
        resolucion_color=(640, 480),
        resolucion_profundidad=(640, 480),
        fps: int = 30,
    ) -> None:
        self.resolucion_color = resolucion_color
        self.resolucion_profundidad = resolucion_profundidad
        self.fps = fps

    @property
    def rs(self):
        return self.helper().rs

    def get_depth_scale(self, *args: Any, **kwargs: Any) -> Any:
        return self.helper().get_depth_scale(*args, **kwargs)

    def extract_rgb(self, *args: Any, **kwargs: Any) -> Any:
        return self.helper().extract_rgb(*args, **kwargs)

    def extract_depth_raw(self, *args: Any, **kwargs: Any) -> Any:
        return self.helper().extract_depth_raw(*args, **kwargs)

    def extract_depth_meters(self, *args: Any, **kwargs: Any) -> Any:
        return self.helper().extract_depth_meters(*args, **kwargs)

    def compute_rays_from_intrinsics(self, *args: Any, **kwargs: Any) -> Any:
        return self.helper().compute_rays_from_intrinsics(*args, **kwargs)

    def compute_normalized_rays(self, *args: Any, **kwargs: Any) -> Any:
        return self.helper().compute_normalized_rays(*args, **kwargs)

    def make_depth_to_color_aligner(self, *args: Any, **kwargs: Any) -> Any:
        return self.helper().make_depth_to_color_aligner(*args, **kwargs)

    def precompute_rays_for_stream(self, *args: Any, **kwargs: Any) -> Any:
        return self.helper().precompute_rays_for_stream(*args, **kwargs)

    def init_camera(self, *args: Any, **kwargs: Any) -> Any:
        return self.helper().init_camera(*args, **kwargs)

    def inicializar_camara(self, *args: Any, **kwargs: Any) -> Any:
        return self.init_camera(*args, **kwargs)

    def capturar_rgb(self, frames: Any, *args: Any, **kwargs: Any) -> Any:
        return self.extract_rgb(frames, *args, **kwargs)

    def capturar_profundidad(self, frames: Any, *args: Any, **kwargs: Any) -> Any:
        return self.extract_depth_meters(frames, *args, **kwargs)

    def calcular_rayos(self, *args: Any, **kwargs: Any) -> Any:
        return self.compute_normalized_rays(*args, **kwargs)

    def alinear_profundidad_color(self, *args: Any, **kwargs: Any) -> Any:
        return self.make_depth_to_color_aligner(*args, **kwargs)

    def capturar_imagenes(self, parametros_camara: Optional[Dict[str, Any]] = None) -> Any:
        params = parametros_camara or {}
        pipeline, _profile = self.init_camera(
            color_width=params.get("color_width", self.resolucion_color[0]),
            color_height=params.get("color_height", self.resolucion_color[1]),
            depth_width=params.get("depth_width", self.resolucion_profundidad[0]),
            depth_height=params.get("depth_height", self.resolucion_profundidad[1]),
            fps=params.get("fps", self.fps),
        )
        frames = pipeline.wait_for_frames()
        align_depth = self.make_depth_to_color_aligner(pipeline)
        rgb = self.extract_rgb(frames)
        profundidad = align_depth(frames) if align_depth is not None else self.extract_depth_meters(frames)
        rayos = self.compute_normalized_rays(rgb.shape[0], rgb.shape[1]) if rgb is not None else None
        return rgb, profundidad, rayos


camara = Camara()
