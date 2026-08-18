"""
Class facade for the segmentation pipeline.
"""
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional


class Segmentacion:
    """Fachada POO para la implementacion de segmentacion."""

    def __init__(self) -> None:
        self._helper = None
        self._impl = None

    def _obtener_impl(self):
        if self._impl is not None:
            return self._impl
        helper_path = Path(__file__).resolve().parent / "implementations" / "segmentacion.py"
        spec = importlib.util.spec_from_file_location(
            "src.application.implementations_segment",
            helper_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"No se pudo cargar helper de segmentacion: {helper_path}")
        helper_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper_module)
        self._helper = helper_module
        self._impl = helper_module.segmentacion
        return self._impl

    def __getattr__(self, name: str) -> Any:
        return getattr(self._obtener_impl(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_helper", "_impl"}:
            object.__setattr__(self, name, value)
            return
        setattr(self._obtener_impl(), name, value)

    def obtener_estado_global(self) -> Dict[str, Any]:
        return self._obtener_impl().obtener_estado_global()

    def algoritmos_segmentacion(self, *args: Any, **kwargs: Any) -> Any:
        return self._obtener_impl().algoritmos_segmentacion(*args, **kwargs)

    def ejecutar_algoritmo_segmentacion(self, *args: Any, **kwargs: Any) -> Any:
        return self.algoritmos_segmentacion(*args, **kwargs)

    def segmentar(self, frame_started_at: Optional[float] = None) -> Any:
        return self._obtener_impl().segmentar(frame_started_at=frame_started_at)

    def segmentacion(self, rgb: Any, profundidad: Any, rayos: Any, parametros: Dict[str, Any]) -> Any:
        impl = self._obtener_impl()
        impl["imagenRGB"] = rgb
        impl["mapaProfundidad"] = profundidad
        impl["rays_cp"] = rayos
        if hasattr(rgb, "shape"):
            impl["H"], impl["W"] = rgb.shape[:2]
        if parametros:
            impl.actualizar_parametros(parametros)
        return impl.segmentar()

    def preprocesamiento(self, rgb: Any, profundidad: Any) -> Any:
        return rgb, profundidad

    def preprocesar(
        self,
        pipeline=None,
        mode: str = "camera",
        dataset_index: Optional[int] = None,
    ) -> bool:
        return self._obtener_impl().preprocesar(
            pipeline=pipeline,
            mode=mode,
            dataset_index=dataset_index,
        )

    def configurar_tarea(self, funcion, *args: Any, **kwargs: Any) -> None:
        self._obtener_impl().configurar_tarea(funcion, *args, **kwargs)

    def obtener_resultado(
        self,
        bloqueante: bool = False,
        timeout: Optional[float] = None,
    ) -> Any:
        return self._obtener_impl().obtener_resultado(bloqueante=bloqueante, timeout=timeout)

    def iniciar_hilo_secundario(self, daemon: bool = True) -> None:
        self._obtener_impl().iniciar_hilo_secundario(daemon=daemon)

    def detener_hilo_secundario(self, timeout: Optional[float] = 2.0) -> None:
        self._obtener_impl().detener_hilo_secundario(timeout=timeout)

    def inicializar(
        self,
        color_width: int = 640,
        color_height: int = 480,
        depth_width: int = 640,
        depth_height: int = 480,
        fps: int = 30,
        stride: int = 2,
        mode: str = "camera",
    ) -> None:
        self._obtener_impl().inicializar(
            color_width=color_width,
            color_height=color_height,
            depth_width=depth_width,
            depth_height=depth_height,
            fps=fps,
            stride=stride,
            mode=mode,
        )

    def actualizar_parametros(self, nuevos_params: Dict[str, Any]) -> Dict[str, Any]:
        return self._obtener_impl().actualizar_parametros(nuevos_params)

    def obtener_parametros(self, copy: bool = True) -> Dict[str, Any]:
        return self._obtener_impl().obtener_parametros(copy=copy)

    def obtener_metricas(self, copy: bool = True) -> Dict[str, Any]:
        return self._obtener_impl().obtener_metricas(copy=copy)

    def obtener_mascaras(self, copy: bool = True) -> Dict[str, Any]:
        return self._obtener_impl().obtener_mascaras(copy=copy)

    def liberar_recursos(self) -> None:
        self._obtener_impl().liberar_recursos()

    def esta_cargando_modelo_puerta(self) -> bool:
        return self._obtener_impl().esta_cargando_modelo_puerta()

    def modelo_puerta_cargando(self) -> bool:
        return self.esta_cargando_modelo_puerta()

    def construir_parametros_muro(
        self,
        ground_params: Dict[str, Any],
        wall_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self._obtener_impl().construir_parametros_muro(ground_params, wall_cfg)


segmentacion = Segmentacion()
