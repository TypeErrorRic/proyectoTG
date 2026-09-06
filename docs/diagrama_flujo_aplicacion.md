# Diagrama de flujo del funcionamiento general de la aplicación

El siguiente diagrama representa el flujo operativo de la aplicación de segmentación semántica RGB-D. Distingue el procesamiento en línea mediante una cámara Intel RealSense del procesamiento fuera de línea mediante el conjunto de datos de prueba.

```mermaid
flowchart TD
    A([Inicio]) --> B[Inicializar aplicación e interfaz gráfica]
    B --> C[Cargar parámetros de segmentación almacenados]
    C --> D{¿Configuración válida?}

    D -- No --> E[Notificar parámetros inválidos]
    E --> F[Modificar parámetros de segmentación]
    F --> G[Validar y aplicar configuración]
    G --> D

    D -- Sí --> H[Presentar panel de ejecución]
    H --> I{¿Qué fuente de datos se utilizará?}

    I -- Dataset de pruebas --> J[Seleccionar muestra RGB-D e índice]
    J --> K[Cargar imagen RGB, mapa de profundidad y parámetros asociados]
    K --> M

    I -- Cámara RGB-D --> L[Inicializar Intel RealSense y flujo de captura]
    L --> L1{¿Dispositivo y flujo disponibles?}
    L1 -- No --> L2[Notificar fallo de adquisición]
    L2 --> I
    L1 -- Sí --> L3[Capturar y alinear fotogramas RGB y profundidad]
    L3 --> M

    M{¿Datos RGB-D válidos?}
    M -- No --> N[Descartar fotograma y registrar incidencia]
    N --> Q
    M -- Sí --> O[Preprocesar datos RGB-D]
    O --> P[Ejecutar segmentación de camino transitable, muros y puertas]
    P --> P1[Aplicar refinamiento y validación geométrica por profundidad]
    P1 --> P2[Componer máscaras y superponer resultados sobre la imagen RGB]
    P2 --> P3[Actualizar métricas de rendimiento y visualización]
    P3 --> Q{¿Continuar procesamiento?}

    Q -- Sí, dataset --> J
    Q -- Sí, cámara --> L3
    Q -- Cambiar modo --> I
    Q -- No --> R[Detener hilo de procesamiento]
    R --> S[Liberar cámara, modelos y recursos de ejecución]
    S --> T([Fin])

    classDef terminal fill:#E8F1FB,stroke:#1F4E79,stroke-width:2px,color:#111;
    classDef process fill:#F7F9FC,stroke:#44546A,stroke-width:1.5px,color:#111;
    classDef decision fill:#FFF2CC,stroke:#BF9000,stroke-width:1.5px,color:#111;
    classDef error fill:#FCE4D6,stroke:#C00000,stroke-width:1.5px,color:#111;

    class A,T terminal;
    class B,C,F,G,H,J,K,L,L3,O,P,P1,P2,P3,R,S process;
    class D,I,L1,M,Q decision;
    class E,L2,N error;
```

## Convenciones

- Los nodos ovalados representan el inicio y el fin del proceso.
- Los rectángulos representan actividades o procesos del sistema.
- Los rombos representan decisiones con salidas explícitamente etiquetadas.
- El color rojo identifica rutas de excepción o datos no válidos.

**Título sugerido para el documento:** *Diagrama de flujo del procesamiento de segmentación semántica RGB-D en modos en línea y fuera de línea*.

**Nota metodológica:** la configuración de parámetros se presenta como una etapa validable e iterativa. La selección de la fuente se formula como una pregunta con dos entradas técnicamente diferenciadas: cámara RGB-D y conjunto de datos de prueba.
