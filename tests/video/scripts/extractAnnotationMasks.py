"""Extract binary class masks from a CVAT Segmentation Mask ZIP archive.

The output contains one directory per requested class. Every generated PNG is a
single-channel image where 255 (white) belongs to the class and 0 is background.
"""

from __future__ import annotations

import argparse
import io
import re
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image


DEFAULT_CLASSES = {
    "Door": "puerta",
    "Ground": "suelo",
    "Wall": "muro",
}


def read_labelmap(archive: zipfile.ZipFile) -> dict[str, tuple[int, int, int]]:
    """Return ``label -> RGB color`` entries from CVAT's labelmap.txt."""
    try:
        content = archive.read("labelmap.txt").decode("utf-8-sig")
    except KeyError as exc:
        raise ValueError("El ZIP no contiene labelmap.txt") from exc

    colors: dict[str, tuple[int, int, int]] = {}
    pattern = re.compile(r"^\s*([^#:]+?)\s*:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*:")
    for line in content.splitlines():
        match = pattern.match(line)
        if match:
            colors[match.group(1).strip().casefold()] = tuple(
                int(value) for value in match.groups()[1:]
            )
    return colors


def extract_masks(
    archive_path: Path,
    output_dir: Path,
    classes: dict[str, str] = DEFAULT_CLASSES,
) -> dict[str, int]:
    """Extract one binary PNG per class and source image."""
    if not archive_path.is_file():
        raise FileNotFoundError(f"No existe el archivo: {archive_path}")

    counts = {folder: 0 for folder in classes.values()}
    with zipfile.ZipFile(archive_path) as archive:
        label_colors = read_labelmap(archive)
        missing = [name for name in classes if name.casefold() not in label_colors]
        if missing:
            raise ValueError(
                "Clases ausentes en labelmap.txt: " + ", ".join(missing)
            )

        members = sorted(
            name
            for name in archive.namelist()
            if name.startswith("SegmentationClass/") and name.lower().endswith(".png")
        )
        if not members:
            raise ValueError("El ZIP no contiene PNG en SegmentationClass/")

        for folder in classes.values():
            (output_dir / folder).mkdir(parents=True, exist_ok=True)

        for member in members:
            filename = PurePosixPath(member).name
            with Image.open(io.BytesIO(archive.read(member))) as image:
                rgb = np.asarray(image.convert("RGB"))

            for label, folder in classes.items():
                color = np.asarray(label_colors[label.casefold()], dtype=np.uint8)
                binary = np.all(rgb == color, axis=2).astype(np.uint8) * 255
                Image.fromarray(binary, mode="L").save(output_dir / folder / filename)
                counts[folder] += 1

    return counts


def find_single_zip(assets_dir: Path) -> Path:
    archives = sorted(assets_dir.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"No se encontró ningún ZIP en {assets_dir}")
    if len(archives) > 1:
        names = ", ".join(path.name for path in archives)
        raise ValueError(f"Hay varios ZIP ({names}); indica uno con --archive")
    return archives[0]


def parse_args() -> argparse.Namespace:
    video_test_dir = Path(__file__).resolve().parent.parent
    default_assets = video_test_dir / "videos" / "assets"
    parser = argparse.ArgumentParser(
        description="Separa las máscaras de CVAT en puerta, suelo y muro."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="ZIP exportado por CVAT (por defecto, el único ZIP de assets).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_assets,
        help="Carpeta de salida.",
    )
    parser.set_defaults(assets_dir=default_assets)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = args.archive or find_single_zip(args.assets_dir)
    counts = extract_masks(archive.resolve(), args.output.resolve())
    print(f"Archivo procesado: {archive.resolve()}")
    print(f"Salida: {args.output.resolve()}")
    for class_name, count in counts.items():
        print(f"  {class_name}: {count} máscaras")


if __name__ == "__main__":
    main()
