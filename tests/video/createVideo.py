"""Crea un video MP4 a partir de las imagenes de la carpeta RGB."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import cv2

THIS_DIR = Path(__file__).resolve().parent
# Modifica esta ruta absoluta si la carpeta RGB cambia de ubicacion.
DEFAULT_INPUT = Path(
    "/home/jetson/Desktop/proyectoTG/tests/video/data/overlay"
)
DEFAULT_OUTPUT = THIS_DIR / "videos" / "rgb_video.mp4"
DEFAULT_METADATA = THIS_DIR / "videos" / "capture_metadata.json"
FPS_MULTIPLIER = 0.5
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def find_images(folder: Path) -> list[Path]:
    images = [path for path in folder.iterdir()
              if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(images, key=natural_key)


def metadata_fps(path: Path, fallback: float = 10.0) -> float:
    if not path.is_file():
        return fallback
    try:
        value = float(json.loads(path.read_text(encoding="utf-8"))["fps"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return fallback
    return value if value > 0 else fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forma un video MP4 a partir de una secuencia de imagenes RGB."
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="Ruta absoluta de la carpeta RGB que se procesara.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=None,
                        help="FPS; si se omite se leen del metadato.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Numero maximo de imagenes; 0 significa todas.")
    return parser.parse_args()


def create_video(input_dir: Path, output_path: Path, fps: float,
                 max_frames: int = 0) -> int:
    if not input_dir.is_absolute():
        raise ValueError(f"La ruta de entrada debe ser absoluta: {input_dir}")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de imagenes: {input_dir}")
    if fps <= 0:
        raise ValueError("Los FPS deben ser mayores que cero.")
    if max_frames < 0:
        raise ValueError("--max-frames no puede ser negativo.")
    images = find_images(input_dir)
    if max_frames:
        images = images[:max_frames]
    if not images:
        raise FileNotFoundError(
            f"No se encontraron imagenes PNG, JPG o JPEG en: {input_dir}"
        )
    first_frame = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"No se pudo leer la primera imagen: {images[0]}")
    height, width = first_frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"No se pudo crear el video: {output_path}")
    written = 0
    try:
        for index, image_path in enumerate(images, start=1):
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"No se pudo leer la imagen: {image_path}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            written += 1
            print(f"\rFotogramas: {index}/{len(images)}", end="", flush=True)
    finally:
        writer.release()
    print(f"\nVideo creado: {output_path.resolve()}")
    print(f"Resolucion: {width}x{height} | FPS: {fps:g} | Fotogramas: {written}")
    return written


def main() -> int:
    args = parse_args()
    original_fps = (
        args.fps if args.fps is not None else metadata_fps(args.metadata)
    )
    fps = original_fps * FPS_MULTIPLIER
    print(
        f"FPS originales: {original_fps:g} | "
        f"Multiplicador: {FPS_MULTIPLIER:g} | FPS de salida: {fps:g}"
    )
    create_video(args.input, args.output.resolve(), fps, args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
