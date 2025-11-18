"""
Modulo de hilos para Jetson Nano.

Define un hilo secundario que ejecuta periodicamente una funcion de usuario
(la "tarea") en segundo plano y comparte los resultados con el hilo principal
mediante una cola.
"""

#Librerías Personales:
import src.utilities.viewCamera as viewCamera
from src.utilities.ransacCellingGround import get_ground
from src.utilities.helpers import ColocarMascara

#Librerias de Funcionamiento
import cupy as cp
import cv2
import threading
import time
import queue
from typing import Optional, Callable, Any

# =======================
# Estado y parámetros runtime (para inicialización de cámara y rayos)
# =======================

# Diccionario de estado para evitar variables globales sueltas
_runtime = {
    'initialized': False,
    'pipeline': None,
    'rays_cp': None,
    'H': None,
    'W': None,
    'align_depth_fn': None,
    'params': None,
    'result_dict': {},
    'fps_t0': None,
    'subsample_stride': None,
}

# Parámetros por defecto para mantener FPS aceptable
SUBSAMPLE_STRIDE = 4        # muestreo 1/s^2 para RANSAC
DIST_THRESH_RUN = 0.03      # tolerancia más estricta

# Evento para detener el hilo de forma limpia
_detener_evento = threading.Event()
_hilo_trabajador: Optional[threading.Thread] = None

# Cola de resultados compartida con el hilo principal (buffer de 1 elemento).
# Esto hace que el hilo secundario se bloquee cuando ya hay un resultado
# pendiente, y solo vuelva a ejecutar la tarea cuando el hilo principal
# haya consumido ese resultado.
_resultados: "queue.Queue[Any]" = queue.Queue(maxsize=1)

# Tipo de la funcion que se ejecutara en el hilo secundario.
# Debe devolver el "resultado" que se quiere compartir con el hilo principal.
TareaFuncion = Callable[..., Any]

# Referencia a la funcion que se ejecutara en el hilo
_tarea_funcion: Optional[TareaFuncion] = None
# Argumentos con los que se llamara a la funcion en el hilo
_tarea_args: tuple[Any, ...] = ()
_tarea_kwargs: dict[str, Any] = {}


def configurar_tarea(funcion: TareaFuncion, *args: Any, **kwargs: Any) -> None:
    global _tarea_funcion, _tarea_args, _tarea_kwargs
    _tarea_funcion = funcion
    _tarea_args = args
    _tarea_kwargs = kwargs


def _bucle_hilo() -> None:
    while not _detener_evento.is_set():
        if _tarea_funcion is not None:
            try:
                resultado = _tarea_funcion(*_tarea_args, **_tarea_kwargs)
                # Solo el hilo secundario inserta resultados en la cola.
                # Si ya hay un resultado pendiente, se bloquea hasta que
                # el hilo principal lo consuma.
                if resultado is not None:
                    while not _detener_evento.is_set():
                        try:
                            _resultados.put(resultado, timeout=0.1)
                            break
                        except queue.Full:
                            # Espera a que el hilo principal consuma el dato
                            continue
            except Exception as exc:
                # Puedes cambiar este print por logging si lo prefieres
                print(f"[thread] Error en tarea de fondo: {exc}")
        else:
            # Si aún no se ha configurado una tarea, evita un bucle ocupado.
            time.sleep(0.01)

def obtener_resultado(bloqueante: bool = False,
                      timeout: Optional[float] = None) -> Any:
    """
    Devuelve un resultado producido por la tarea.

    - Si `bloqueante=False` (por defecto), devuelve inmediatamente:
      * un resultado, o
      * None si no hay nada disponible.
    - Si `bloqueante=True`, espera hasta que haya un resultado o
      hasta que expire `timeout` (si se indica). Si se agota el tiempo,
      devuelve None.
    """
    try:
        if bloqueante:
            return _resultados.get(block=True, timeout=timeout)
        return _resultados.get(block=False)
    except queue.Empty:
        return None


def iniciar_hilo_secundario(daemon: bool = True) -> None:
    """
    Crea e inicia el hilo secundario (si no esta ya iniciado).
    """
    global _hilo_trabajador
    if _hilo_trabajador is not None and _hilo_trabajador.is_alive():
        return

    _detener_evento.clear()
    _hilo_trabajador = threading.Thread(
        target=_bucle_hilo,
        name="hilo_tarea_secundaria",
        daemon=daemon,
    )
    _hilo_trabajador.start()


def detener_hilo_secundario(timeout: Optional[float] = 2.0) -> None:
    """
    Solicita la parada del hilo y espera a que termine.
    """
    global _hilo_trabajador
    if _hilo_trabajador is None:
        return

    _detener_evento.set()
    if _hilo_trabajador.is_alive():
        _hilo_trabajador.join(timeout)
    _hilo_trabajador = None


def _lazy_init(color_width=640, color_height=480, depth_width=640,
               depth_height=480, fps=30, stride=2,) -> None:
    """Inicializa cámara y rayos en la primera llamada."""
    if _runtime['initialized']:
        return
    print("Inicializando cámara RealSense…")
    pipeline, params = viewCamera.init_camera(
        color_width,
        color_height,
        depth_width,
        depth_height,
        fps,
        stride,        # submuestreo para nube
        yaw=-45.0,
        pitch=25.0,
        roll=0.0,
        fov=60.0,
        point_size=1
    )
    rays_np, H, W, align_depth_fn = viewCamera.precompute_rays_for_stream(pipeline, viewCamera.rs.stream.color)
    _runtime['pipeline'] = pipeline
    _runtime['rays_cp'] = cp.asarray(rays_np)
    _runtime['H'] = H
    _runtime['W'] = W
    _runtime['align_depth_fn'] = align_depth_fn
    _runtime['params'] = params
    _runtime['result_dict'] = {'dist_thresh': DIST_THRESH_RUN}
    _runtime['fps_t0'] = time.time()
    _runtime.setdefault('algoritmo', 1)
    _runtime.setdefault('mascara', None)
    # Usar el stride de params para el submuestreo de RANSAC si viene configurado
    try:
        _runtime['subsample_stride'] = int(params.get('stride', SUBSAMPLE_STRIDE))
    except Exception:
        _runtime['subsample_stride'] = SUBSAMPLE_STRIDE
    _runtime['initialized'] = True


def preprocesar(pipeline=None) -> tuple[Any, Any, Any, Any]:
    """
    Extrae y prepara los datos necesarios para RANSAC:
    - Imagen RGB
    - Mapa de profundidad alineado
    - Mapa de profundidad en cupy
    - Rayos precalculados
    """
    frames = pipeline.wait_for_frames()
    H = _runtime['H']
    W = _runtime['W']
    align_depth_fn = _runtime['align_depth_fn']
    rays_cp = _runtime['rays_cp']

    # Extraer RGB y Depth nativos
    imagenRGB = viewCamera.extract_rgb(frames)
    mapaProfundidad = align_depth_fn(frames) if align_depth_fn is not None else viewCamera.extract_depth_meters(frames)
    if imagenRGB is None or mapaProfundidad is None:
        return None, None, None, None, None

    # Asegurar shape del depth al tamaño de COLOR
    if mapaProfundidad.shape[0] != H or mapaProfundidad.shape[1] != W:
        mapaProfundidad = cv2.resize(mapaProfundidad, (W, H), interpolation=cv2.INTER_NEAREST)

    return imagenRGB, mapaProfundidad, rays_cp, H, W

def AlgoritmosSegmentacion(color_width=640, color_height=480, depth_width=640,
               depth_height=480, fps=30, stride=2,) -> Any:
    """
    Lee el resultado del hilo secundario, hace el preprocesamiento
    y programa el siguiente algoritmo de segmentacion (1 suelo,
    2 pared, 3 puerta) para el hilo secundario.
    """

    _lazy_init(color_width, color_height, depth_width, depth_height, fps, stride,)

    # Intentar obtener un resultado reciente del hilo secundario
    resultado = obtener_resultado()
    if resultado is not None:
        # Actualizar la mǭscara con el resultado mǭs reciente
        _runtime['mascara'] = resultado
    else:
        # Preprocesamiento comun a los tres casos: debe devolver
        # los parametros de entrada para el hilo secundario.
        imagenRGB, mapaProfundidad, rays_cp, H, W = preprocesar(_runtime['pipeline'])

        if imagenRGB is None:
            # No hay datos nuevos; devolver la ǭltima mǭscara conocida (si existe)
            if _runtime['mascara'] is not None:
                return ColocarMascara(_runtime['mascara'])
            return None

        algoritmo = _runtime.get('algoritmo', 1)
        if algoritmo == 1:
            # Configurar tarea get_ground con los argumentos correctos
            configurar_tarea(get_ground, imagenRGB, mapaProfundidad, rays_cp, H, W)
            iniciar_hilo_secundario()

        # Mientras el hilo calcula, usar al menos la imagen RGB actual
        if _runtime['mascara'] is None:
            _runtime['mascara'] = imagenRGB

    return ColocarMascara(_runtime['mascara'])
