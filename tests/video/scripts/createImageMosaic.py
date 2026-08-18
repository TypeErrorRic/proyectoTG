"""Crea un mosaico usando una seleccion de imagenes guardada en JSON."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps


VIDEO_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = VIDEO_DIR / "data" / "overlay"
DEFAULT_SELECTION = VIDEO_DIR / "config" / "outdoor_mosaic_selection.json"
DEFAULT_OUTPUT = VIDEO_DIR / "results" / "outdoor_mosaic_4x6.jpg"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_selection(path: Path) -> tuple[list[int], int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo de seleccion: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON invalido en {path}, linea {exc.lineno}: {exc.msg}"
        ) from exc
    if not isinstance(config, dict):
        raise ValueError("La configuracion del mosaico debe ser un objeto JSON.")

    images = config.get("images")
    columns = config.get("columns")
    rows = config.get("rows")
    if not isinstance(images, list) or not images:
        raise ValueError("El campo 'images' debe ser una lista no vacia.")
    if not all(isinstance(index, int) and index >= 0 for index in images):
        raise ValueError("Todos los valores de 'images' deben ser enteros positivos.")
    if len(images) != len(set(images)):
        raise ValueError("El campo 'images' contiene indices repetidos.")
    if not isinstance(columns, int) or columns <= 0:
        raise ValueError("El campo 'columns' debe ser un entero positivo.")
    if not isinstance(rows, int) or rows <= 0:
        raise ValueError("El campo 'rows' debe ser un entero positivo.")
    if columns * rows != len(images):
        raise ValueError(
            f"La cuadricula {columns}x{rows} requiere {columns * rows} imagenes, "
            f"pero el JSON contiene {len(images)}."
        )
    declared_count = config.get("image_count")
    if declared_count is not None and declared_count != len(images):
        raise ValueError(
            f"image_count indica {declared_count}, pero hay {len(images)} indices."
        )
    return images, columns, rows


def image_index(path: Path) -> Optional[int]:
    """Obtiene el ultimo numero del nombre (p. ej. overlay_0018 -> 18)."""
    numbers = re.findall(r"\d+", path.stem)
    return int(numbers[-1]) if numbers else None


def index_images(folder: Path) -> dict[int, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de imagenes: {folder}")
    indexed: dict[int, Path] = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        index = image_index(path)
        if index is not None:
            indexed.setdefault(index, path)
    return indexed


def select_images(available: dict[int, Path], indices: list[int]) -> list[tuple[int, Path]]:
    missing = [index for index in indices if index not in available]
    if missing:
        raise FileNotFoundError(
            "No se encontraron en la carpeta de entrada los indices: "
            + ", ".join(map(str, missing))
        )
    return [(index, available[index]) for index in indices]


def create_mosaic(
    selected: list[tuple[int, Path]], output: Path, columns: int, rows: int,
    cell_width: int, cell_height: int, quality: int,
) -> None:
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "black")
    for position, (_, path) in enumerate(selected):
        with Image.open(path) as source:
            tile = ImageOps.fit(
                source.convert("RGB"), (cell_width, cell_height),
                method=Image.Resampling.LANCZOS,
            )
        x = (position % columns) * cell_width
        y = (position // columns) * cell_height
        canvas.paste(tile, (x, y))

    output.parent.mkdir(parents=True, exist_ok=True)
    save_options = {"quality": quality, "optimize": True} if output.suffix.lower() in {
        ".jpg", ".jpeg"
    } else {}
    canvas.save(output, **save_options)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crea un mosaico usando los indices y la distribucion definidos "
            "en un archivo JSON."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION,
                        help="JSON que contiene images, columns y rows.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cell-width", type=int, default=320)
    parser.add_argument("--cell-height", type=int, default=240)
    parser.add_argument("--quality", type=int, default=92)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cell_width <= 0 or args.cell_height <= 0:
        raise ValueError("Las dimensiones de cada celda deben ser positivas.")
    if not 1 <= args.quality <= 100:
        raise ValueError("--quality debe estar entre 1 y 100.")

    indices, columns, rows = load_selection(args.selection.resolve())
    available = index_images(args.input.resolve())
    selected = select_images(available, indices)
    create_mosaic(
        selected, args.output.resolve(), columns, rows,
        args.cell_width, args.cell_height, args.quality,
    )
    print(f"Mosaico creado: {args.output.resolve()}")
    print(f"Distribucion: {columns}x{rows} | Imagenes: {len(selected)}")
    print(f"Seleccion leida desde: {args.selection.resolve()}")
    print("Seleccion: " + ", ".join(map(str, indices)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

