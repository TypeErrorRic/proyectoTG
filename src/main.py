import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
    )

from src.GUI import run_app
from src.api.dbConection import test_query


def main() -> None:
    try:
        result = test_query()
        print(f"[DB] Conexion exitosa: {result}")
    except Exception as exc:
        print(f"[DB] No se pudo conectar/consultar: {exc}")
    run_app(mode="camera")


if __name__ == "__main__":
    main()
