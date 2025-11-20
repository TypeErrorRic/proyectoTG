"""
Punto de entrada de la aplicacion de segmentacion.
Lanza la interfaz gráfica definida en src/GUI.py.
"""

import os
import sys

# Asegurar que se pueda importar el paquete src cuando se ejecuta como script
if __package__ is None or __package__ == "":
    sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
    )

from src.GUI import run_app


def main() -> None:
    run_app(mode="camera")


if __name__ == "__main__":
    main()
