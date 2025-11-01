# Desarrollo de Trabajo de Grado

Lista de Requirimientos.

## Aceleración GPU opcional (CuPy)

Este proyecto usa CuPy para acelerar cómputo intensivo en GPU cuando está disponible, manteniendo compatibilidad en CPU (NumPy) sin cambios:

- `utilities/ransacCellingGround.py` detecta CuPy automáticamente y acelera el RANSAC de planos en GPU; si CuPy no está instalado, usa NumPy.
- `utilities/tx_fifo.py` minimiza copias y evita conversiones costosas; la parte pesada (detección de plano) ya se ejecuta con CuPy a través del módulo anterior.

Notas:

- Para CuPy, instala la variante adecuada a tu versión de CUDA: consulta https://docs.cupy.dev/ para el comando `pip` correcto.
- No es obligatorio CuPy; el código funciona íntegramente en CPU.

