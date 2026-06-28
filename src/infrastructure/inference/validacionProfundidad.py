"""
Validacion geometrica de mascaras usando profundidad.
"""
from typing import Any, Dict, Optional

import numpy as np


class PuertaDeep:
    """Encapsula el filtrado 3D de puertas basado en profundidad y mascaras."""

    def __init__(self) -> None:
        self._helper = None
        self.debug_door_deep = False

    def _obtener_helper(self):
        if self._helper is None:
            from src.infrastructure.inference.helpers import validacionProfundidad as deep_helpers

            self._helper = deep_helpers
            self.debug_door_deep = deep_helpers.DEBUG_DOOR_DEEP
        return self._helper

    def convertir_a_numpy(self, arr: Any):
        deep_helpers = self._obtener_helper()
        return deep_helpers._to_numpy(arr)

    def mascaras_roi_puntos(
        self,
        door_mask_raw: np.ndarray,
        hsv_mask: np.ndarray,
        depth_m: np.ndarray,
        rays: Any,
        imagen_rgb: Optional[np.ndarray] = None,
        stride: int = 4,
        ground_normal: Any = None,
        ground_parallel_deg: float = 15.0,
        merge_gap_px: int = 20,
        density_voxel: float = 0.05,
        seed_radius_ratio: float = 0.12,
        min_plane_points: int = 50,
        max_density_points: int = 20000,
        plane_inlier_dist: float = 0.02,
        plane_inlier_ratio: float = 0.30,
        trim_keep_ratio: float = 0.70,
        trim_iters: int = 2,
        edge_sigma: float = 0.33,
        edge_dilate: int = 1,
        max_iters: int = 64,
        min_island_pixels: int = 300,
        use_realsense: bool = True,
    ) -> Dict[str, Any]:
        deep_helpers = self._obtener_helper()
        return deep_helpers.door_roi_pointclouds(
            door_mask_raw,
            hsv_mask,
            depth_m,
            rays,
            imagen_rgb=imagen_rgb,
            stride=stride,
            ground_normal=ground_normal,
            ground_parallel_deg=ground_parallel_deg,
            merge_gap_px=merge_gap_px,
            density_voxel=density_voxel,
            seed_radius_ratio=seed_radius_ratio,
            min_plane_points=min_plane_points,
            max_density_points=max_density_points,
            plane_inlier_dist=plane_inlier_dist,
            plane_inlier_ratio=plane_inlier_ratio,
            trim_keep_ratio=trim_keep_ratio,
            trim_iters=trim_iters,
            edge_sigma=edge_sigma,
            edge_dilate=edge_dilate,
            max_iters=max_iters,
            min_island_pixels=min_island_pixels,
            use_realsense=use_realsense,
        )

    def puntos_desde_mascaras(
        self,
        door_mask_raw: np.ndarray,
        hsv_mask: np.ndarray,
        depth_m: np.ndarray,
        rays: Any,
        imagen_rgb: Optional[np.ndarray] = None,
        stride: int = 4,
        ground_normal: Any = None,
        ground_parallel_deg: float = 15.0,
        merge_gap_px: int = 20,
        density_voxel: float = 0.05,
        seed_radius_ratio: float = 0.12,
        min_plane_points: int = 50,
        max_density_points: int = 20000,
        plane_inlier_dist: float = 0.02,
        plane_inlier_ratio: float = 0.30,
        trim_keep_ratio: float = 0.70,
        trim_iters: int = 2,
        edge_sigma: float = 0.33,
        edge_dilate: int = 1,
        max_iters: int = 64,
        min_island_pixels: int = 300,
        use_realsense: bool = True,
    ) -> Dict[str, Any]:
        deep_helpers = self._obtener_helper()
        return deep_helpers.door_points_from_masks(
            door_mask_raw,
            hsv_mask,
            depth_m,
            rays,
            imagen_rgb=imagen_rgb,
            stride=stride,
            ground_normal=ground_normal,
            ground_parallel_deg=ground_parallel_deg,
            merge_gap_px=merge_gap_px,
            density_voxel=density_voxel,
            seed_radius_ratio=seed_radius_ratio,
            min_plane_points=min_plane_points,
            max_density_points=max_density_points,
            plane_inlier_dist=plane_inlier_dist,
            plane_inlier_ratio=plane_inlier_ratio,
            trim_keep_ratio=trim_keep_ratio,
            trim_iters=trim_iters,
            edge_sigma=edge_sigma,
            edge_dilate=edge_dilate,
            max_iters=max_iters,
            min_island_pixels=min_island_pixels,
            use_realsense=use_realsense,
        )

    def obtener_estado_global(self) -> Dict[str, bool]:
        return {"debug_door_deep": self.debug_door_deep}


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
        from application.gestorFotogramas import geometria

        return geometria.points_from_rays_and_depth(rayos, mapa_profundidad)
