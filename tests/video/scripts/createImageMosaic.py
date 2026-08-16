"""Crea un mosaico 6x4 usando los indices de ImagenesArreglar.txt."""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps


PROJECT_DIR = Path(__file__).resolve().parents[3]
VIDEO_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = VIDEO_DIR / "data" / "overlay"
DEFAULT_CANDIDATES = PROJECT_DIR / "ImagenesArreglar.txt"
DEFAULT_OUTPUT = VIDEO_DIR / "results" / "mosaico_6x4.jpg"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_candidates(path: Path) -> tuple[list[int], list[int]]:
    """Devuelve todos los indices y los que estan marcados como venus."""
    candidates: list[int] = []
    venus: list[int] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^(\d+)", line)
        if not match:
            raise ValueError(
                f"Linea {line_number} invalida en {path}: {raw_line!r}"
            )
        index = int(match.group(1))
        candidates.append(index)
        if "venus" in line.casefold():
            venus.append(index)
    if not candidates:
        raise ValueError(f"No hay candidatos en {path}")
    if not venus:
        raise ValueError(f"No hay ningun candidato marcado como venus en {path}")
    return candidates, venus


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


def choose_images(
    available: dict[int, Path], candidates: list[int], venus: list[int],
    amount: int, seed: Optional[int],
) -> list[tuple[int, Path]]:
    valid = [index for index in candidates if index in available]
    valid_venus = [index for index in venus if index in available]
    if not valid_venus:
        raise FileNotFoundError(
            "Ninguna imagen marcada como venus existe en la carpeta de entrada. "
            f"Indices venus esperados: {venus}"
        )
    if len(valid) < amount:
        missing = sorted(set(candidates) - set(available))
        raise ValueError(
            f"Solo hay {len(valid)} candidatos disponibles y se necesitan {amount}. "
            f"Indices ausentes: {missing}"
        )

    rng = random.Random(seed)
    selected_venus = rng.choice(valid_venus)
    remaining = [index for index in valid if index != selected_venus]
    chosen = [selected_venus, *rng.sample(remaining, amount - 1)]
    rng.shuffle(chosen)
    return [(index, available[index]) for index in chosen]


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
            "Crea un mosaico de 6 columnas por 4 filas con candidatos de "
            "ImagenesArreglar.txt e incluye al menos una imagen venus."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=None,
                        help="Semilla para repetir exactamente la seleccion.")
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

    columns, rows = 6, 4
    candidates, venus = parse_candidates(args.candidates.resolve())
    available = index_images(args.input.resolve())
    selected = choose_images(
        available, candidates, venus, columns * rows, args.seed
    )
    create_mosaic(
        selected, args.output.resolve(), columns, rows,
        args.cell_width, args.cell_height, args.quality,
    )
    selected_indices = [index for index, _ in selected]
    included_venus = sorted(set(selected_indices) & set(venus))
    print(f"Mosaico creado: {args.output.resolve()}")
    print(f"Distribucion: {columns}x{rows} | Imagenes: {len(selected)}")
    print(f"Indices venus incluidos: {included_venus}")
    print("Seleccion: " + ", ".join(map(str, selected_indices)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

