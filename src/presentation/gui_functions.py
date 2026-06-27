"""
Class facade for GUI helper behavior.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple

import tkinter as tk
from PIL import Image


class FuncionesGUI:
    """Agrupa estado y metodos auxiliares usados por GUI.py."""

    def __init__(self) -> None:
        self._helper = None
        self.default_config_fallback: Dict[str, str] = {
            "subsample_stride": "1",
            "dist_thresh": "0.03",
            "max_iters": "400",
            "min_inliers": "400",
            "max_angle_deg": "60.0",
            "max_up_dot": "0.35",
            "score_subset": "4096",
            "early_stop_ratio": "0.92",
            "batch_size": "128",
            "low_height_pct": "25.0",
            "roi_bottom_fraction": "0.34",
            "roi_expand_step": "0.2",
            "max_agg_points": "150000",
            "refine_full_res": "1",
            "refine_max_points": "200000",
            "refine_dist_mult": "1.6",
            "ground_mask_refine": "0",
            "wall_subsample_stride": "2",
            "wall_dist_thresh": "0.03",
            "wall_max_iters": "300",
            "wall_min_inliers": "400",
            "wall_max_angle_deg": "20.0",
            "wall_score_subset": "4096",
            "wall_early_stop_ratio": "0.90",
            "wall_batch_size": "1024",
            "wall_refine_dist_mult": "1.6",
            "wall_mask_refine": "0",
            "ground_perp_deg": "20.0",
            "wall_ortho_deg": "20.0",
            "wall_parallel_deg": "10.0",
            "wall_parallel_distance_m": "0.60",
            "door_hue_tol": "18",
            "door_hsv_enabled": "1",
            "door_min_s": "30",
            "door_min_v": "20",
            "door_glare_s_max": "35",
            "door_glare_v_min": "210",
            "door_glare_v_clip": "200",
            "door_ground_parallel_deg": "15.0",
            "door_plane_inlier_ratio": "0.40",
        }
        self.segmentacion = None
        self.mascaras = None

    def _obtener_helper(self):
        if self._helper is None:
            try:
                from src.presentation.helpers import GUIFunctions as gui_helpers
            except ModuleNotFoundError:
                from presentation.helpers import GUIFunctions as gui_helpers  # type: ignore

            self._helper = gui_helpers
            self.segmentacion = getattr(gui_helpers, "segmentacion", None)
            self.mascaras = getattr(gui_helpers, "helpers_mod", None)
        return self._helper

    def ensure_upload_dir(self, upload_dir: str) -> None:
        return self._obtener_helper().ensure_upload_dir(upload_dir)

    def cargar_parametros_defecto(self) -> Dict[str, str]:
        return dict(self.default_config_fallback)

    def load_upload_images(
        self,
        upload_dir: str,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp"),
    ) -> List[str]:
        return self._obtener_helper().load_upload_images(upload_dir, extensions)

    def cargar_imagenes(self, ruta: str) -> List[str]:
        return self.load_upload_images(ruta)

    def visualize_capture(self, app: Any) -> None:
        return self._obtener_helper().visualize_capture(app)

    def make_icon(self, kind: str) -> Image.Image:
        return self._obtener_helper().make_icon(kind)

    def load_sidebar_icons(self, base_path: str, assets: Dict[str, str]) -> Dict[str, Image.Image]:
        return self._obtener_helper().load_sidebar_icons(base_path, assets)

    def cargar_iconos(self, ruta: str, assets: Dict[str, str]) -> Dict[str, Image.Image]:
        return self.load_sidebar_icons(ruta, assets)

    def init_config_defaults(
        self,
        runtime_params_loader: Callable[[], Dict[str, Any]],
        fallback: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        return self._obtener_helper().init_config_defaults(
            runtime_params_loader,
            fallback or self.default_config_fallback,
        )

    def param_summary_fields(self) -> List[Tuple[str, str]]:
        return self._obtener_helper().param_summary_fields()

    def validate_numeric_entry(self, proposed: str) -> bool:
        return self._obtener_helper().validate_numeric_entry(proposed)

    def config_field_descriptions(self, descriptions_path: Optional[str] = None) -> Dict[str, str]:
        return self._obtener_helper().config_field_descriptions(descriptions_path)

    def parse_config_params(self, values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self._obtener_helper().parse_config_params(values)

    def validar_parametros(self, valores: Dict[str, Any]) -> bool:
        return self.parse_config_params(valores) is not None

    def convertir_parametros(self, valores: Dict[str, Any]) -> Dict[str, Any]:
        return self.parse_config_params(valores) or {}

    def capture_panel_screenshot(
        self,
        panel: Optional[tk.Widget],
        upload_dir: str,
    ) -> Optional[Dict[str, Any]]:
        return self._obtener_helper().capture_panel_screenshot(panel, upload_dir)

    def capturar_panel(self, panel: Optional[tk.Widget], upload_dir: str) -> Optional[Dict[str, Any]]:
        return self.capture_panel_screenshot(panel, upload_dir)

    def toggle_indicator_label(
        self,
        label: Optional[tk.Label],
        on_color: str = "#ffffff",
        off_color: str = "#000000",
    ) -> None:
        return self._obtener_helper().toggle_indicator_label(label, on_color, off_color)

    def set_indicator_label_state(
        self,
        label: Optional[tk.Label],
        enabled: Optional[bool],
        on_color: str = "#ffffff",
        off_color: str = "#000000",
    ) -> None:
        return self._obtener_helper().set_indicator_label_state(
            label,
            enabled,
            on_color,
            off_color,
        )

    def toggle_mask_flag(self, name: str) -> Optional[bool]:
        return self._obtener_helper().toggle_mask_flag(name)

    def alternar_mascara(self, nombre: str) -> Optional[bool]:
        return self.toggle_mask_flag(nombre)

    def actualizar_indicador(
        self,
        label: Optional[tk.Label],
        activo: Optional[bool],
    ) -> None:
        return self.set_indicator_label_state(label, activo)

    def capturar_parametros_metodos(self) -> Dict[str, Any]:
        estado = self.obtener_estado_global()
        if self.segmentacion is not None:
            try:
                estado["parametros_segmentacion"] = self.segmentacion.obtener_parametros()
            except Exception:
                estado["parametros_segmentacion"] = {}
        return estado

    def on_indicator_floor(self, app: Any, label: Optional[tk.Label] = None) -> None:
        return self._obtener_helper().on_indicator_floor(app, label)

    def on_indicator_wall(self, app: Any, label: Optional[tk.Label] = None) -> None:
        return self._obtener_helper().on_indicator_wall(app, label)

    def on_indicator_door(self, app: Any, label: Optional[tk.Label] = None) -> None:
        return self._obtener_helper().on_indicator_door(app, label)

    def obtener_estado_global(self) -> Dict[str, Any]:
        self._obtener_helper()
        return {
            "default_config_fallback": dict(self.default_config_fallback),
            "segmentacion": self.segmentacion,
            "mascaras": self.mascaras,
        }
