# Diagrama de clases actualizado para el flujo de segmentaciÃ³n

Este diagrama conserva la estructura de relaciones del modelo de clases original e integra las funciones principales descritas en el pseudocÃ³digo del flujo principal y del procedimiento de segmentaciÃ³n.

ConvenciÃ³n de visibilidad: `+` pÃºblico, `-` privado y `#` protegido.

```mermaid
classDiagram
    direction LR

    class SegmentacionApp {
        -tuple dimensiones
        -tuple posicion
        -bool redimensionamiento
        -List componentes
        -Dict configuracion_interfaz
        +inicializar() None
        -construir_paneles() None
        +mostrar_modo(modo: str) None
        -establecer_modo(modo: str) None
        -aplicar_configuracion() None
        +seleccionar_imagen_dataset(indice: int) None
        +iniciar_procesamiento() None
        +detener_procesamiento() None
        -ejecutar_en_segundo_plano() None
        -actualizar_imagen() None
        +capturar_pantalla() None
        +cerrar_aplicacion() None
        +flujo_principal_segmentacion() Any
    }

    class FuncionesGUI {
        -Dict parametros_configuracion
        +cargar_parametros_defecto() Dict
        +validar_parametros(valores: Dict) bool
        +convertir_parametros(valores: Dict) Dict
        -cargar_imagenes(ruta: str) List
        -cargar_iconos(ruta: str) Dict
        -capturar_panel(panel: Any) Dict
        +alternar_mascara(nombre: str) bool
        +actualizar_indicador(nombre: str, activo: bool) None
        +capturar_parametros_metodos() Dict
    }

    class Camara {
        -tuple resolucion_color
        -tuple resolucion_profundidad
        -int fps
        +inicializar_camara() Any
        -capturar_rgb() Any
        -capturar_profundidad() Any
        -calcular_rayos() Any
        -alinear_profundidad_color() Any
        +capturar_imagenes(parametros_camara: Dict) tuple
    }

    class DatasetFrames {
        -str ruta_imagenes
        -str ruta_profundidad
        -int indice_actual
        +cargar_frame(indice: int) tuple
        +listar_imagenes() List
        +obtener_nombre_frame(indice: int) str
    }

    class Segmentacion {
        -str modo
        -Any imagen_rgb
        -Any mapa_profundidad
        -Any rayos
        -Dict mascaras_resultado
        -Dict metricas_resultado
        -bool inicializado
        +inicializar(modo: str) None
        -preprocesar(modo: str, indice_dataset: int) bool
        +segmentar() Any
        -ejecutar_algoritmo_segmentacion(modo: str, indice_dataset: int) Any
        +actualizar_parametros(parametros: Dict) Dict
        +obtener_parametros() Dict
        +obtener_metricas() Dict
        +obtener_mascaras() Dict
        +liberar_recursos() None
        +modelo_puerta_cargando() bool
        +segmentacion(rgb: Any, profundidad: Any, rayos: Any, parametros: Dict) Any
        +preprocesamiento(rgb: Any, profundidad: Any) tuple
    }

    class Mascaras {
        -Any mascara_suelo
        -Any mascara_muro
        -Any mascara_puerta
        -Dict visibilidad
        -refinar_mascara_suelo(mascara: Any) Any
        -refinar_mascara_muro(mascara: Any) Any
        -combinar_mascaras() Dict
        -aplicar_sobre_rgb(imagen_rgb: Any) Any
        +alternar_visibilidad(nombre: str) bool
        +fusionar_mascaras(rgb: Any, suelo: Any, muro: Any, puerta: Any) Any
    }

    class DetectorClase {
        <<Abstract>>
        -Dict parametros
        -Any mascara
        -Dict metricas
        +detectar(imagen_rgb: Any, mapa_profundidad: Any, rayos: Any) Any
        +actualizar_parametros(parametros: Dict) None
        +obtener_mascara() Any
        +obtener_metricas() Dict
    }

    class CaminoTransitable {
        -Any plano_suelo
        -float tiempo_ransac_ms
        +detectar(imagen_rgb: Any, mapa_profundidad: Any, rayos: Any) Any
        -estimar_plano_suelo() Any
        -ajustar_plano_ransac() Any
        -refinar_plano() Any
        +algoritmo_segmentacion_suelo(rgb: Any, profundidad: Any, parametros: Dict) Any
    }

    class Muro {
        -List planos_verticales
        -Any normal_suelo
        +detectar(imagen_rgb: Any, mapa_profundidad: Any, rayos: Any) Any
        -estimar_planos_verticales() List
        -filtrar_con_suelo() Any
        +algoritmo_segmentacion_muro(rgb: Any, profundidad: Any, parametros: Dict) Any
    }

    class Puerta {
        -Any modelo
        -str ruta_modelo
        -tuple tamano_entrada
        -bool cargando_modelo
        +detectar(imagen_rgb: Any, mapa_profundidad: Any, rayos: Any) Any
        -preprocesar_imagen(imagen_rgb: Any) Any
        -postprocesar_salida(salida: Any) Any
        +modelo_cargando() bool
        +algoritmo_segmentacion_puerta(rgb: Any, profundidad: Any, parametros: Dict) Any
    }

    class ValidacionProfundidad {
        -float distancia_plano
        -float porcentaje_inliers
        +validar_por_profundidad(mascara: Any, mapa_profundidad: Any, rayos: Any) Any
        -generar_nube_puntos(mapa_profundidad: Any, rayos: Any) Any
    }

    class InferenciaTensorRT {
        -Any motor
        -Any contexto
        -List buffers_entrada
        -List buffers_salida
        +cargar_modelo(ruta_modelo: str) None
        +inferir(entrada: Any) Any
        +liberar_recursos() None
    }

    class RefinamientoHSV {
        -int tolerancia_tono
        -int saturacion_minima
        -int valor_minimo
        +refinar_por_color(imagen_rgb: Any, mascara: Any) Any
        -rellenar_huecos(mascara: Any) Any
    }

    SegmentacionApp "1" *-- "1" FuncionesGUI : contiene
    SegmentacionApp "1" *-- "1" Segmentacion : controla
    SegmentacionApp "1" *-- "1" Camara : modo camara
    SegmentacionApp "1" *-- "1" DatasetFrames : modo prueba

    Segmentacion "1" *-- "1" CaminoTransitable : detector suelo
    Segmentacion "1" *-- "1" Muro : detector muro
    Segmentacion "1" *-- "1" Puerta : detector puerta
    Segmentacion "1" *-- "1" Mascaras : resultados

    DetectorClase <|-- CaminoTransitable
    DetectorClase <|-- Muro
    DetectorClase <|-- Puerta

    Puerta "1" *-- "1" InferenciaTensorRT : inferencia
    Puerta "1" *-- "1" RefinamientoHSV : color
    Puerta "1" *-- "1" ValidacionProfundidad : geometria 3D
```

## Correspondencia con el pseudocÃ³digo

| FunciÃ³n del pseudocÃ³digo | Clase del diagrama | FunciÃ³n integrada | Visibilidad |
| --- | --- | --- | --- |
| `Capturar_Parametros_Metodos()` | `FuncionesGUI` | `capturar_parametros_metodos()` | PÃºblica |
| `Capturar_Imagenes(parametros_camara)` | `Camara` | `capturar_imagenes(parametros_camara)` | PÃºblica |
| `Preprocesamiento(RGB, Profundidad)` | `Segmentacion` | `preprocesamiento(rgb, profundidad)` | PÃºblica |
| `Segmentacion(...)` | `Segmentacion` | `segmentacion(rgb, profundidad, rayos, parametros)` | PÃºblica |
| `AlgoritmoSegmentacionSuelo(...)` | `CaminoTransitable` | `algoritmo_segmentacion_suelo(...)` | PÃºblica |
| `AlgoritmoSegmentacionMuro(...)` | `Muro` | `algoritmo_segmentacion_muro(...)` | PÃºblica |
| `AlgoritmoSegmentacionPuerta(...)` | `Puerta` | `algoritmo_segmentacion_puerta(...)` | PÃºblica |
| `FusionarMascaras(...)` | `Mascaras` | `fusionar_mascaras(...)` | PÃºblica |

Las funciones auxiliares que no aparecen como punto de entrada del pseudocÃ³digo ni como contrato entre clases se modelan como privadas. Por ejemplo, `capturar_rgb`, `capturar_profundidad`, `calcular_rayos`, `alinear_profundidad_color`, `estimar_plano_suelo`, `estimar_planos_verticales`, `preprocesar_imagen`, `postprocesar_salida`, `generar_nube_puntos`, `combinar_mascaras` y `aplicar_sobre_rgb` quedan encapsuladas dentro de sus respectivas clases.

## Flujo resumido

1. `SegmentacionApp` controla la interfaz, obtiene parÃ¡metros desde `FuncionesGUI` y activa el procesamiento en segundo plano.
2. En modo cÃ¡mara, `Camara` captura RGB, profundidad y rayos; en modo prueba, `DatasetFrames` carga los frames del dataset.
3. `Segmentacion` ejecuta el preprocesamiento y coordina los detectores de suelo, muro y puerta.
4. `CaminoTransitable`, `Muro` y `Puerta` implementan los algoritmos grandes de segmentaciÃ³n de cada clase.
5. `Puerta` usa `InferenciaTensorRT`, `RefinamientoHSV` y `ValidacionProfundidad` como mÃ³dulos auxiliares.
6. `Mascaras` combina los resultados y genera la imagen final con mÃ¡scaras sobre RGB.
