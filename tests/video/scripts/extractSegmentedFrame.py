"""Extrae las tres mascaras de un fotograma a las etiquetas del dataset.

Lee las imagenes de camino transitable, puerta y muro generadas por
``extractVideoFrames.py``. Los tres archivos se guardan como PNG con un nombre
comun (por defecto, ``test_XXXXX.png``) dentro de sus respectivas subcarpetas
en ``src/infrastructure/datasets/labels``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PROJECT_ROOT / "tests" / "video" / "data"
DEFAULT_DATASET = PROJECT_ROOT / "src" / "infrastructure" / "datasets"
SOURCE_DIRECTORIES = {
    "camino_transitable": "camino_transitable",
    "puerta": "puerta",
    "muro": "muro",
}
DESTINATION_DIRECTORIES = {
    "camino_transitable": Path("labels") / "floorGroundTruth",
    "puerta": Path("labels") / "doorGrounTruth",
    "muro": Path("labels") / "wallGroundTruth",
}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def find_frame(directory: Path, frame_number: int) -> Path:
    """Encuentra un unico ``frame_NNNNNN`` sin asumir su extension."""
    stem = f"frame_{frame_number:06d}"
    matches = [
        directory / f"{stem}{extension}"
        for extension in IMAGE_EXTENSIONS
        if (directory / f"{stem}{extension}").is_file()
    ]
    if not matches:
        raise FileNotFoundError(f"No se encontro {stem} en {directory}")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(f"Hay varios archivos para {stem} en {directory}: {names}")
    return matches[0]


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"No se pudo leer la imagen: {path}")
    return image


def extract_segmented_frame(
    frame_number: int,
    input_dir: Path = DEFAULT_INPUT,
    dataset_dir: Path = DEFAULT_DATASET,
    name: str | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Copia las mascaras de camino, puerta y muro a ``labels``."""
    if frame_number < 1:
        raise ValueError("El numero de fotograma debe ser mayor o igual a 1.")

    filename = name or f"test_{frame_number:05d}.png"
    if Path(filename).name != filename or Path(filename).suffix.lower() != ".png":
        raise ValueError("El nombre de salida debe ser un archivo .png sin directorios.")

    sources = {
        kind: find_frame(input_dir / directory, frame_number)
        for kind, directory in SOURCE_DIRECTORIES.items()
    }
    images = {kind: read_image(path) for kind, path in sources.items()}

    reference_shape = images["camino_transitable"].shape[:2]
    for kind in ("puerta", "muro"):
        if images[kind].shape[:2] != reference_shape:
            raise ValueError(
                f"Las dimensiones de {sources[kind]} ({images[kind].shape}) "
                f"no coinciden con camino_transitable ({reference_shape})."
            )

    destinations = {
        kind: dataset_dir / directory / filename
        for kind, directory in DESTINATION_DIRECTORIES.items()
    }
    existing = [path for path in destinations.values() if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Ya existen archivos de destino: {names}. Usa --overwrite para reemplazarlos."
        )

    for path in destinations.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    try:
        for kind, destination in destinations.items():
            if not cv2.imwrite(str(destination), images[kind]):
                raise OSError(f"No se pudo guardar: {destination}")
            written.append(destination)
    except Exception:
        # Solo revierte archivos nuevos; nunca elimina destinos preexistentes.
        if not overwrite:
            for path in written:
                path.unlink(missing_ok=True)
        raise

    return destinations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame", type=int, help="Numero de frame (empieza en 1).")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--name",
        help="Nombre PNG comun de salida; por defecto test_XXXXX.png.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        outputs = extract_segmented_frame(
            frame_number=args.frame,
            input_dir=args.input_dir.resolve(),
            dataset_dir=args.dataset_dir.resolve(),
            name=args.name,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, cv2.error) as exc:
        print(f"Error al extraer el fotograma: {exc}", file=sys.stderr)
        return 1

    print("Fotograma extraido correctamente:")
    for kind, path in outputs.items():
        print(f"  {kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
