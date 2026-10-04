"""Extract one synchronized RGB/depth pair from a capture HDF5 file.

The output is ready for the application's test dataset: matching PNG names in
``src/infrastructure/datasets/images`` and ``depths``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_H5 = Path(__file__).resolve().parent.parent / "videos" / "capture.h5"
DEFAULT_OUTPUT = PROJECT_ROOT / "src" / "infrastructure" / "datasets"


def extract_pair(
    h5_path: Path,
    position: int,
    output_dir: Path = DEFAULT_OUTPUT,
    name: str | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Extract a one-based position as matching RGB and uint16 depth PNGs."""
    if position < 1:
        raise ValueError("La posición debe ser mayor o igual a 1.")
    if not h5_path.is_file():
        raise FileNotFoundError(f"No existe el archivo HDF5: {h5_path}")

    filename = name or f"test_{position:05d}.png"
    if Path(filename).name != filename or Path(filename).suffix.lower() != ".png":
        raise ValueError("El nombre de salida debe ser un archivo .png sin directorios.")
    rgb_path = output_dir / "images" / filename
    depth_path = output_dir / "depths" / filename
    if not overwrite and (rgb_path.exists() or depth_path.exists()):
        raise FileExistsError(
            f"Ya existe {rgb_path if rgb_path.exists() else depth_path}; "
            "usa --overwrite para reemplazar el par."
        )

    with h5py.File(h5_path, "r") as source:
        if "rgb" not in source or "depth" not in source:
            raise ValueError("El HDF5 debe contener los datasets 'rgb' y 'depth'.")
        rgb_dataset, depth_dataset = source["rgb"], source["depth"]
        if len(rgb_dataset) != len(depth_dataset):
            raise ValueError("Los datasets RGB y depth tienen tamaños diferentes.")
        if position > len(rgb_dataset):
            raise ValueError(
                f"Posición {position} fuera de rango; hay {len(rgb_dataset)} pares."
            )

        index = position - 1
        if int(source.attrs.get("format_version", 1)) >= 2:
            rgb_bytes = np.asarray(rgb_dataset[index], dtype=np.uint8)
            depth_bytes = np.asarray(depth_dataset[index], dtype=np.uint8)
            rgb = cv2.imdecode(rgb_bytes, cv2.IMREAD_COLOR)
            depth = cv2.imdecode(depth_bytes, cv2.IMREAD_UNCHANGED)
        else:
            rgb_array = np.asarray(rgb_dataset[index])
            depth = np.asarray(depth_dataset[index])
            if rgb_array.ndim != 3 or rgb_array.shape[2] != 3 or rgb_array.dtype != np.uint8:
                raise ValueError(f"RGB inválido en la posición {position}.")
            rgb = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"No se pudo decodificar el RGB de la posición {position}.")
    if depth is None or depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError(f"Depth inválido en la posición {position}; se requiere PNG uint16.")
    if rgb.shape[:2] != depth.shape:
        raise ValueError(f"RGB y depth tienen dimensiones diferentes en la posición {position}.")

    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(rgb_path), rgb):
        raise OSError(f"No se pudo guardar RGB: {rgb_path}")
    if not cv2.imwrite(str(depth_path), depth):
        rgb_path.unlink(missing_ok=True)
        raise OSError(f"No se pudo guardar depth: {depth_path}")
    return rgb_path, depth_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("position", nargs="?", type=int, help="Índice del par en el HDF5 (empieza en 1).")
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5, help="Archivo HDF5 de entrada.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Carpeta que contiene images/ y depths/.")
    parser.add_argument("--name", help="Nombre compartido de salida, por ejemplo test_01348.png.")
    parser.add_argument("--overwrite", action="store_true", help="Reemplazar archivos existentes.")
    args = parser.parse_args()
    position = args.position
    if position is None:
        try:
            while True:
                value = input("Índice del par (número entero desde 1): ").strip()
                try:
                    position = int(value)
                except ValueError:
                    print("El índice debe ser un número entero.")
                    continue
                if position < 1:
                    print("El índice debe ser mayor o igual a 1.")
                    continue
                break
        except (EOFError, KeyboardInterrupt):
            print("\nEntrada cancelada.", file=sys.stderr)
            return 1
    try:
        rgb_path, depth_path = extract_pair(
            args.h5, position, args.output_dir, args.name, args.overwrite
        )
    except (OSError, ValueError, KeyError, cv2.error) as exc:
        print(f"Error al extraer el par: {exc}", file=sys.stderr)
        return 1
    print(f"RGB:   {rgb_path}\nDepth: {depth_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
