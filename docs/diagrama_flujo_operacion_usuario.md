# Flujo general de operación de la aplicación

Este diagrama presenta, desde la perspectiva del usuario, la secuencia recomendada para configurar y ejecutar correctamente la aplicación.

```mermaid
flowchart TD
    A([Inicio]) --> B[Ejecutar la aplicación]
    B --> C[Esperar la inicialización de la interfaz gráfica]
    C --> D[Ingresar al módulo de configuración]
    D --> E[Revisar o modificar los parámetros de segmentación]
    E --> F[Aplicar la configuración]
    F --> G{¿Los parámetros son válidos?}

    G -- No --> H[Corregir los valores indicados]
    H --> E
    G -- Sí --> I[Ingresar al módulo de ejecución]

    I --> J{¿Qué modo de ejecución se utilizará?}

    J -- Dataset de pruebas --> K[Seleccionar una muestra del dataset]
    K --> L[Ejecutar el procesamiento de la muestra]

    J -- Cámara RGB-D --> M[Verificar la conexión de la cámara Intel RealSense]
    M --> N[Presionar «Iniciar transmisión»]
    N --> O{¿La cámara inicia correctamente?}
    O -- No --> P[Revisar conexión y disponibilidad del dispositivo]
    P --> M
    O -- Sí --> Q[Procesar el flujo RGB-D en tiempo real]

    L --> R[Visualizar los resultados de segmentación]
    Q --> R
    R --> S{¿Realizar otra acción?}

    S -- Probar otra muestra --> K
    S -- Cambiar modo --> J
    S -- Ajustar parámetros --> D
    S -- Guardar evidencia --> T[Capturar y almacenar el resultado]
    T --> R
    S -- Finalizar --> U[Detener la transmisión o el procesamiento]
    U --> V[Cerrar la aplicación]
    V --> W([Fin])

    classDef terminal fill:#D9EAF7,stroke:#1F4E79,stroke-width:2px,color:#111;
    classDef action fill:#F7F9FC,stroke:#44546A,stroke-width:1.5px,color:#111;
    classDef decision fill:#FFF2CC,stroke:#BF9000,stroke-width:1.5px,color:#111;
    classDef warning fill:#FCE4D6,stroke:#C00000,stroke-width:1.5px,color:#111;

    class A,W terminal;
    class B,C,D,E,F,I,K,L,M,N,Q,R,T,U,V action;
    class G,J,O,S decision;
    class H,P warning;
```

**Título sugerido para el documento:** *Diagrama de flujo general para la operación de la aplicación de segmentación RGB-D*.

El diagrama representa las acciones visibles para el usuario. Los procesos internos de adquisición, preprocesamiento y segmentación pueden documentarse por separado en un diagrama técnico del sistema.
