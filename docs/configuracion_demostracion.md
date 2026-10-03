# Configuración de demostración: alta sensibilidad

Este perfil permite demostrar el efecto del módulo de configuración sin modificar
`config/segmentar_defaults.json`. Está pensado para escenas interiores con ruido de
profundidad, superficies parcialmente visibles o una cámara ligeramente inclinada.

El archivo completo está en
`config/segmentar_demo_alta_sensibilidad.json`. Para probarlo desde el panel de
configuración, introduzca los siguientes valores visibles y pulse **Aplicar**.

## Camino transitable

| Campo | Valor predeterminado | Valor de demostración |
|---|---:|---:|
| Submuestreo | 2 | 1 |
| Umbral de distancia | 0.03 m | 0.045 m |
| Iteraciones máximas | 300 | 600 |
| Mínimo de puntos compatibles | 400 | 300 |
| Ángulo máximo | 60° | 70° |
| Subconjunto para puntuar | 2048 | 4096 |
| Corte temprano | 0.90 | 0.95 |
| Tamaño de lote | 512 | 256 |
| Percentil bajo de altura | 25 | 30 |
| Fracción inferior ROI | 0.34 | 0.45 |
| Refinar a resolución completa | Activado | Activado |
| Mejorar máscara de suelo | Activado | Activado |
| Multiplicador de refino | 1.6 | 1.8 |

## Muros

| Campo | Valor predeterminado | Valor de demostración |
|---|---:|---:|
| Submuestreo | 2 | 1 |
| Umbral de distancia | 0.03 m | 0.045 m |
| Iteraciones máximas | 300 | 600 |
| Mínimo de puntos compatibles | 400 | 300 |
| Ángulo máximo | 20° | 25° |
| Subconjunto para puntuar | 2048 | 4096 |
| Corte temprano | 0.90 | 0.95 |
| Tamaño de lote | 512 | 512 |
| Multiplicador de refino | 1.6 | 1.8 |
| Máximo producto vertical | 0.35 | 0.42 |
| Perpendicular al suelo | 20° | 25° |
| Paredes ortogonales | 20° | 25° |
| Paredes paralelas | 10° | 15° |
| Distancia entre paredes | 0.60 m | 0.80 m |
| Mejorar máscara de pared | Activado | Activado |

## Puerta

| Campo | Valor predeterminado | Valor de demostración |
|---|---:|---:|
| Filtro HSV | Activado | Activado |
| Rango de color | 18 | 25 |
| Color mínimo | 50 | 25 |
| Luz mínima | 20 | 15 |
| Reflejo: color máximo | 35 | 45 |
| Reflejo: luz mínima | 210 | 195 |
| Reducción de reflejo | 200 | 180 |
| Inclinación máxima | 15° | 20° |
| Proporción mínima en el plano | 0.40 | 0.30 |

## Resultado esperado

- El suelo debe cubrir una región más continua, incluyendo zonas con mediciones de
  profundidad irregulares y una porción mayor de la imagen.
- Las paredes estrechas, parcialmente ocultas o algo inclinadas deben detectarse con
  mayor frecuencia y mostrar menos huecos en la máscara.
- El filtro de puertas debe conservar más variaciones de color, sombras y reflejos,
  y aceptar puertas parcialmente visibles.
- El cambio debe verse inmediatamente después de pulsar **Aplicar**; si el proceso
  está activo, el trabajador se reinicia con los nuevos parámetros.
- Como efecto secundario intencional, puede aumentar el tiempo de cálculo y pueden
  incluirse algunos píxeles de objetos cercanos en las máscaras. Esto hace visible
  el compromiso entre sensibilidad, precisión y rendimiento.

El botón **Cancelar** recupera la última configuración aplicada; no restaura
necesariamente los valores originales una vez que este perfil ya fue aplicado. Para
volver al perfil base, use nuevamente los valores de
`config/segmentar_defaults.json` o reinicie la aplicación.
