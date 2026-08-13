# Organización de `tests`

Los archivos se agrupan por responsabilidad:

```text
tests/
├── data/                 # Dataset PNG para evaluación
├── evaluation/           # Cálculo de métricas y evaluación NYU Depth V2
├── tools/                # Utilidades generales de mantenimiento
└── video/
    ├── config/           # Configuración de captura
    ├── data/             # Resultados de segmentación (ignorado por Git)
    ├── results/          # Métricas generadas por el procesamiento
    ├── scripts/          # Captura, conversión y procesamiento de video
    └── videos/           # Capturas, video y anotaciones CVAT
        └── assets/       # ZIP versionado; máscaras extraídas ignoradas
```

## Evaluación

- `evaluation/evaluate_nyu_v2.py`: evalúa el modelo con NYU Depth V2.
- `evaluation/metrics.py`: calcula métricas sobre las máscaras de `tests/data`.
- `evaluation/metrics.json`: último resultado guardado.

Los accesos habituales siguen disponibles mediante `run20.sh`:

```bash
./run20.sh eval-nyu
./run20.sh metrics
```

## Flujo de video

Ejecuta los scripts desde la raíz del repositorio:

```bash
python3 tests/video/scripts/captureVideo.py
python3 tests/video/scripts/createHDF5.py
python3 tests/video/scripts/extractVideoFrames.py
python3 tests/video/scripts/createVideo.py --input /ruta/absoluta/a/imagenes
python3 tests/video/scripts/extractAnnotationMasks.py
```

- `captureVideo.py`: captura pares RGB y profundidad RealSense.
- `createHDF5.py`: empaqueta la captura sincronizada en HDF5.
- `extractVideoFrames.py`: reconstruye y segmenta los fotogramas.
- `createVideo.py`: crea un MP4 a partir de imágenes.
- `extractAnnotationMasks.py`: separa el ZIP de CVAT en puerta, suelo y muro.
- `cleanData.py`: elimina capturas RGB-D locales.

La configuración de captura está en `video/config/capture_config.json`.

## Limpieza

Para previsualizar o borrar los PNG de `tests/data`:

```bash
python3 tests/tools/cleanEvaluationData.py --dry-run
python3 tests/tools/cleanEvaluationData.py
```

Para borrar capturas locales de video:

```bash
python3 tests/video/scripts/cleanData.py
```
