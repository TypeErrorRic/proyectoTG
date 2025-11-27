## Aceleración GPU opcional (CuPy)

Este proyecto usa CuPy para acelerar cómputo intensivo en GPU cuando está disponible, manteniendo compatibilidad en CPU (NumPy) sin cambios:

- `utilities/ransacCellingGround.py` detecta CuPy automáticamente y acelera el RANSAC de planos en GPU; si CuPy no está instalado, usa NumPy.
- `utilities/tx_fifo.py` minimiza copias y evita conversiones costosas; la parte pesada (detección de plano) ya se ejecuta con CuPy a través del módulo anterior.

Notas:

- Para CuPy, instala la variante adecuada a tu versión de CUDA: consulta https://docs.cupy.dev/ para el comando `pip` correcto.
- No es obligatorio CuPy; el código funciona íntegramente en CPU.

## Actividades y estado

- [x] ~~1. Realizar una revisión bibliográfica de antecedentes para definir los requerimientos funcionales y no funcionales.~~
  - [x] ~~1.1 Elaboración de un análisis comparativo de antecedentes técnicos para la identificación de requerimientos funcionales relevantes al sistema propuesto.~~
  - [x] ~~1.2 Realización de un análisis de las limitaciones comunes en los antecedentes técnicos consultados.~~
  - [x] ~~1.3 Documentación de los requerimientos funcionales y no funcionales del sistema, con base en las funcionalidades y limitaciones identificadas en los antecedentes.~~
- [ ] 2. Construir un dataset de prueba con anotaciones para reconocimiento de muros, puertas y caminos transitables en entornos interiores estructurados, a partir de nubes de puntos capturadas o de bases de datos existentes.
  - [x] ~~2.1 Recolección de datos mediante cámaras RGB-D propias o a partir de bases de datos existentes en entornos interiores estructurados.~~
  - [x] ~~2.2 Procesamiento de las nubes de puntos para la eliminación de ruido y optimización de la estructura espacial.~~
  - [x] ~~2.3 Anotación semántica de muros, puertas, caminos transitables y otros objetos en los datos recopilados.~~
  - [ ] 2.4 Validación del dataset en un formato estructurado para el entrenamiento, con documentación de los resultados obtenidos.
- [ ] 3. Diseñar un modelo de reconocimiento de muros, puertas, caminos transitables y otros objetos usando el dataset de prueba.
  - [x] ~~3.1 Elaboración del diagrama conceptual del modelo de reconocimiento, según la metodología SCRUM.~~
  - [x] ~~3.2 Definición de las historias de usuario en una herramienta de planificación de proyectos, según la metodología SCRUM.~~
  - [x] ~~3.3 Priorización del backlog con los elementos técnicos y operativos necesarios para el entrenamiento del modelo, conforme a la metodología SCRUM.~~
  - [ ] 3.4 Diseño del modelo de reconocimiento según la metodología SCRUM.
  - [ ] 3.5 Elaboración del informe técnico del modelo de reconocimiento junto con el diseño de los esbozos de la interfaz gráfica de usuario (GUI).
- [ ] 4. Implementar en un sistema embebido el modelo de reconocimiento de muros, puertas, caminos transitables y otros objetos.
  - [x] ~~4.1 Implementación de la configuración del entorno del sistema embebido para garantizar la compatibilidad y el despliegue del modelo entrenado.~~
  - [x] ~~4.2 Implementación del módulo de percepción mediante la integración del modelo de reconocimiento en el sistema embebido.~~
  - [ ] 4.3 Implementación del módulo de procesamiento para aplicar el modelo integrado sobre los datos de entrada, generando salidas clasificadas en condiciones operativas controladas.
  - [ ] 4.4 Implementación del módulo de realimentación con registro de métricas de desempeño.
  - [ ] 4.5 Documentación de la integración del modelo de reconocimiento en el sistema embebido.
- [ ] 5. Validar la funcionalidad del aplicativo, así como su exactitud, en el reconocimiento de las clases seleccionadas del sistema en entornos reales mediante un protocolo de pruebas.
  - [ ] 5.1 Definición del protocolo de validación de la exactitud del reconocimiento y la funcionalidad, con métricas, escenarios y criterios de aceptación.
  - [ ] 5.2 Ejecución del protocolo de pruebas en entornos reales para validar la exactitud en el reconocimiento y la funcionalidad del sistema.
  - [ ] 5.3 Realización del registro de los resultados de desempeño, errores y observaciones técnicas.

**Resumen:** Pendientes 13 de 25 actividades (12 completadas).

Progreso (completadas): 48%  
`[########################--------------------------]` 12/25  
![Progreso 48%](https://img.shields.io/badge/Progreso-48%25-00b86b?labelColor=111&color=00b86b)
