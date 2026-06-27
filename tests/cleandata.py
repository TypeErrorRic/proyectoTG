"""
Elimina todos los archivos .png dentro de tests/data (recursivo).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Limpia PNGs en tests/data")
    default_dir = Path(__file__).resolve().parent / "data"
    parser.add_argument(
        "--dir",
        default=str(default_dir),
        help="Directorio base a limpiar (por defecto: tests/data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo listar los archivos a borrar",
    )
    args = parser.parse_args()

    base_dir = Path(args.dir).resolve()
    if not base_dir.exists():
        print(f"No existe el directorio: {base_dir}")
        return 1

    removed = 0
    for path in base_dir.rglob("*.png"):
        if not path.is_file():
            continue
        if args.dry_run:
            print(f"[DRY-RUN] {path}")
            continue
        try:
            path.unlink()
            removed += 1
        except Exception as exc:
            print(f"Error borrando {path}: {exc}")

    print(f"PNG eliminados: {removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
