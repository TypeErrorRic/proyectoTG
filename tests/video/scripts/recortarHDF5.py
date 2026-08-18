"""Elimina intervalos de fotogramas y crea un nuevo HDF5 recortado.

El archivo original nunca se modifica. Los numeros de fotograma mostrados al
usuario empiezan en 1 y ambos extremos de cada intervalo se eliminan.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


VIDEO_TEST_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = VIDEO_TEST_DIR / "videos" / "capture.h5"
DEFAULT_OUTPUT = VIDEO_TEST_DIR / "videos" / "recortado.h5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Elimina intervalos de fotogramas y crea recortado.h5."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help=f"HDF5 original (predeterminado: {DEFAULT_INPUT}).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"HDF5 recortado (predeterminado: {DEFAULT_OUTPUT}).")
    return parser.parse_args()


def frame_count(h5_file: h5py.File) -> int:
    if "rgb" not in h5_file or not isinstance(h5_file["rgb"], h5py.Dataset):
        raise ValueError("El HDF5 no contiene el dataset de fotogramas 'rgb'.")
    count = len(h5_file["rgb"])
    if count == 0:
        raise ValueError("El HDF5 no contiene fotogramas.")
    return count


def ask_frame_number(label: str, total: int) -> int:
    while True:
        value = input(f"{label} [1-{total}]: ").strip()
        try:
            number = int(value)
        except ValueError:
            print("Introduce un numero entero valido.")
            continue
        if 1 <= number <= total:
            return number
        print(f"El numero debe estar entre 1 y {total}.")


def ask_ranges(total: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    print(f"El archivo contiene {total} fotogramas.")
    print("Los fotogramas inicial y final tambien seran eliminados.")

    while True:
        start = ask_frame_number("Fotograma inicial que deseas borrar", total)
        end = ask_frame_number("Fotograma final que deseas borrar", total)
        if start > end:
            print("El inicio no puede ser mayor que el final. Intenta de nuevo.")
            continue
        ranges.append((start, end))
        answer = input("¿Deseas agregar otro intervalo? [s/N]: ").strip().lower()
        if answer not in {"s", "si", "sí", "y", "yes"}:
            break
    return ranges


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def indices_to_keep(total: int, ranges: list[tuple[int, int]]) -> np.ndarray:
    keep = np.ones(total, dtype=bool)
    for start, end in ranges:
        keep[start - 1:end] = False
    return np.flatnonzero(keep)


def copy_attributes(source: h5py.AttributeManager,
                    target: h5py.AttributeManager) -> None:
    for key, value in source.items():
        target[key] = value


def copy_group(source: h5py.Group, target: h5py.Group,
               keep_indices: np.ndarray, total: int) -> None:
    copy_attributes(source.attrs, target.attrs)
    for name, item in source.items():
        if isinstance(item, h5py.Group):
            child = target.create_group(name)
            copy_group(item, child, keep_indices, total)
            continue

        # Se consideran ligados a los fotogramas todos los datasets cuya
        # primera dimension coincide con el numero de imagenes RGB.
        is_frame_dataset = item.ndim >= 1 and item.shape[0] == total
        if is_frame_dataset:
            output_shape = (len(keep_indices), *item.shape[1:])
            output = target.create_dataset(name, shape=output_shape, dtype=item.dtype)
            for output_index, source_index in enumerate(keep_indices):
                output[output_index] = item[int(source_index)]
                if name == "rgb":
                    print(
                        f"\rCopiando fotogramas: {output_index + 1}/{len(keep_indices)}",
                        end="", flush=True,
                    )
        else:
            output = target.create_dataset(name, data=item[...], dtype=item.dtype)
        copy_attributes(item.attrs, output.attrs)


def update_metadata(output: h5py.File, remaining: int,
                    removed_ranges: list[tuple[int, int]]) -> None:
    raw = output.attrs.get("metadata_json")
    if raw is not None:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            metadata = json.loads(str(raw))
        except json.JSONDecodeError:
            metadata = {}
        if isinstance(metadata, dict):
            metadata["frame_count"] = remaining
            metadata["removed_frame_ranges"] = [list(item) for item in removed_ranges]
            output.attrs["metadata_json"] = json.dumps(metadata)
    output.attrs["frame_count"] = remaining
    output.attrs["removed_frame_ranges"] = json.dumps(removed_ranges)


def create_trimmed_hdf5(input_path: Path, output_path: Path,
                        ranges: list[tuple[int, int]]) -> tuple[int, int]:
    if not input_path.is_file():
        raise FileNotFoundError(f"No existe el HDF5 original: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("La salida debe ser diferente del archivo original.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)

    try:
        with h5py.File(input_path, "r") as source:
            total = frame_count(source)
            merged = merge_ranges(ranges)
            keep = indices_to_keep(total, merged)
            if len(keep) == 0:
                raise ValueError("Los intervalos borrarian todos los fotogramas.")
            with h5py.File(temporary, "w") as output:
                copy_group(source, output, keep, total)
                update_metadata(output, len(keep), merged)
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return total, len(keep)


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    with h5py.File(input_path, "r") as h5_file:
        total = frame_count(h5_file)

    ranges = merge_ranges(ask_ranges(total))
    removed = sum(end - start + 1 for start, end in ranges)
    print("Intervalos que se borraran: " +
          ", ".join(f"{start}-{end}" for start, end in ranges))
    print(f"Se borraran {removed} fotogramas y quedaran {total - removed}.")
    confirmation = input("¿Crear el archivo recortado? [s/N]: ").strip().lower()
    if confirmation not in {"s", "si", "sí", "y", "yes"}:
        print("Operacion cancelada; no se creo ningun archivo.")
        return 0

    original, remaining = create_trimmed_hdf5(input_path, output_path, ranges)
    print(f"\nHDF5 creado: {output_path}")
    print(f"Fotogramas originales: {original}")
    print(f"Fotogramas eliminados: {original - remaining}")
    print(f"Fotogramas restantes: {remaining}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
