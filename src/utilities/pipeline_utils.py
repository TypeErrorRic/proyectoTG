"""
Class facades for pipeline utilities.
"""
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional


class PipelineHelperMixin:
    """Carga diferida del helper funcional de utilidades."""

    _helper = None

    @classmethod
    def helper(cls):
        if cls._helper is None:
            helper_path = Path(__file__).resolve().parent / "helpers" / "pipeline_utils.py"
            spec = importlib.util.spec_from_file_location(
                "src.utilities.helpers_pipeline_utils",
                helper_path,
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"No se pudo cargar helper de pipeline: {helper_path}")
            helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helper)
            cls._helper = helper
        return cls._helper


class Geometria(PipelineHelperMixin):
    """Operaciones geometricas y de nube de puntos."""

    def points_from_rays_and_depth(self, *args: Any, **kwargs: Any) -> Any:
        return self.helper().points_from_rays_and_depth(*args, **kwargs)

    def render_pointcloud_numpy(self, *args: Any, **kwargs: Any) -> Any:
        return self.helper().render_pointcloud_numpy(*args, **kwargs)


class Mascaras(PipelineHelperMixin):
    """Operaciones de visibilidad, refinamiento y overlay de mascaras."""

    def set_mask_visibility(self, *args: Any, **kwargs: Any) -> Any:
        return self.helper().set_mask_visibility(*args, **kwargs)

    def toggle_mask_visibility(self, *args: Any, **kwargs: Any) -> Any:
        return self.helper().toggle_mask_visibility(*args, **kwargs)

    def mejorar_mascara_pared(self, *args: Any, **kwargs: Any) -> Any:
        return self.helper().mejorar_mascara_pared(*args, **kwargs)

    def mejorar_mascara_suelo(self, *args: Any, **kwargs: Any) -> Any:
        return self.helper().mejorar_mascara_suelo(*args, **kwargs)

    def apply_mask_to_rgb(self, *args: Any, **kwargs: Any) -> Any:
        return self.helper().apply_mask_to_rgb(*args, **kwargs)


class DatasetFrames(PipelineHelperMixin):
    """Carga de frames RGB/depth del dataset."""

    def load_dataset_frame(self, *args: Any, **kwargs: Any) -> Any:
        return self.helper().load_dataset_frame(*args, **kwargs)


class ConfiguracionDataset(PipelineHelperMixin):
    """Gestion de configuraciones por imagen del dataset."""

    def list_dataset_image_filenames(self, *args: Any, **kwargs: Any) -> Any:
        return self.helper().list_dataset_image_filenames(*args, **kwargs)

    def resolve_dataset_filename_by_index(self, *args: Any, **kwargs: Any) -> Any:
        return self.helper().resolve_dataset_filename_by_index(*args, **kwargs)

    def resolve_image_config_path(self, *args: Any, **kwargs: Any) -> Any:
        return self.helper().resolve_image_config_path(*args, **kwargs)

    def ensure_dataset_image_config_files(self, *args: Any, **kwargs: Any) -> Any:
        return self.helper().ensure_dataset_image_config_files(*args, **kwargs)

    def load_default_segment_params(self, *args: Any, **kwargs: Any) -> Any:
        return self.helper().load_default_segment_params(*args, **kwargs)

    def load_dataset_image_params_by_index(self, *args: Any, **kwargs: Any) -> Any:
        return self.helper().load_dataset_image_params_by_index(*args, **kwargs)


geometria = Geometria()
mascaras = Mascaras()
dataset_frames = DatasetFrames()
configuracion_dataset = ConfiguracionDataset()
