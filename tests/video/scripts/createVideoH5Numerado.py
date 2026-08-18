"""Crea un video MP4 directamente desde los fotogramas RGB de un HDF5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


VIDEO_TEST_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = VIDEO_TEST_DIR / "videos" / "capture.h5"
DEFAULT_OUTPUT = VIDEO_TEST_DIR / "videos" / "capture_numerado.mp4"
DEFAULT_FPS = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crea un MP4 desde el dataset RGB de un archivo HDF5 y muestra "
            "el numero de cada fotograma dentro del video."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help=f"Archivo HDF5 de entrada (predeterminado: {DEFAULT_INPUT}).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Video MP4 de salida (predeterminado: {DEFAULT_OUTPUT}).")
    parser.add_argument("--fps", type=float, default=None,
                        help="FPS de salida; por defecto se leen del HDF5.")
    parser.add_argument("--dataset", default="rgb",
                        help="Dataset de fotogramas (predeterminado: rgb).")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Numero maximo de fotogramas; 0 procesa todos.")
    return parser.parse_args()


def read_metadata(h5_file: h5py.File) -> dict:
    raw = h5_file.attrs.get("metadata_json", "{}")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        metadata = json.loads(str(raw))
    except json.JSONDecodeError:
        metadata = {}
    return metadata if isinstance(metadata, dict) else {}


def decode_frame(value: object) -> np.ndarray:
    """Decodifica JPEG/PNG almacenado o acepta un arreglo RGB HxWx3."""
    array = np.asarray(value)
    if array.ndim == 1:
        frame = cv2.imdecode(array.astype(np.uint8, copy=False), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("No se pudo decodificar el fotograma almacenado.")
        return frame
    if array.ndim == 3 and array.shape[2] in (3, 4):
        frame = array.astype(np.uint8, copy=False)
        conversion = cv2.COLOR_RGBA2BGR if frame.shape[2] == 4 else cv2.COLOR_RGB2BGR
        return cv2.cvtColor(frame, conversion)
    raise ValueError(
        f"Formato no compatible: shape={array.shape}, dtype={array.dtype}."
    )


def add_frame_number(frame: np.ndarray, number: int, total: int) -> None:
    text = f"Fotograma {number}/{total}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.65, min(frame.shape[:2]) / 600.0)
    thickness = max(2, round(scale * 2))
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )
    x, y, padding = 16, 16 + text_height, 8
    cv2.rectangle(
        frame,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + baseline + padding),
        (0, 0, 0), cv2.FILLED,
    )
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255),
                thickness, cv2.LINE_AA)


def create_video(input_path: Path, output_path: Path, dataset_name: str = "rgb",
                 fps_override: float | None = None, max_frames: int = 0) -> int:
    if not input_path.is_file():
        raise FileNotFoundError(f"No existe el archivo HDF5: {input_path}")
    if max_frames < 0:
        raise ValueError("--max-frames no puede ser negativo.")

    with h5py.File(input_path, "r") as h5_file:
        if dataset_name not in h5_file:
            available = ", ".join(h5_file.keys()) or "ninguno"
            raise KeyError(f"No existe '{dataset_name}'. Disponibles: {available}.")
        dataset = h5_file[dataset_name]
        if not isinstance(dataset, h5py.Dataset) or len(dataset) == 0:
            raise ValueError(f"El dataset '{dataset_name}' no contiene fotogramas.")

        source_total = len(dataset)
        total = min(source_total, max_frames) if max_frames else source_total
        metadata = read_metadata(h5_file)
        fps = fps_override if fps_override is not None else float(
            metadata.get("fps", DEFAULT_FPS)
        )
        if fps <= 0:
            raise ValueError("Los FPS deben ser mayores que cero.")

        first_frame = decode_frame(dataset[0])
        height, width = first_frame.shape[:2]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"No se pudo crear el video: {output_path}")

        written = 0
        try:
            for index in range(total):
                frame = first_frame.copy() if index == 0 else decode_frame(dataset[index])
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                add_frame_number(frame, index + 1, total)
                writer.write(frame)
                written += 1
                print(f"\rFotogramas escritos: {written}/{total}", end="", flush=True)
        finally:
            writer.release()

    print(f"\nVideo creado: {output_path.resolve()}")
    print(f"Resolucion: {width}x{height} | FPS: {fps:g} | Fotogramas: {written}")
    return written


def main() -> int:
    args = parse_args()
    create_video(args.input.resolve(), args.output.resolve(), args.dataset,
                 args.fps, args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
