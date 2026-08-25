"""Extract only RGB images from a capture HDF5 file.

By default this reads
``C:\\Users\\equin\\Desktop\\proyectosPersonales\\proyectoTG\\tests\\video\\videos\\capture.h5``
and reconstructs the ``RGB`` folder beside it. Depth images and segmentation
are not processed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np


VIDEO_DIR = Path(
    r"C:\Users\equin\Desktop\proyectosPersonales\proyectoTG\tests\video\videos\backup\indoorLowLight\scena1"
)
DEFAULT_HDF5_PATH = VIDEO_DIR / "capture.h5"
DEFAULT_RGB_OUTPUT_DIR = VIDEO_DIR / "RGB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract only the RGB folder from an HDF5 capture."
    )
    parser.add_argument(
        "--h5",
        type=Path,
        default=DEFAULT_HDF5_PATH,
        help=f"Input HDF5 file (default: {DEFAULT_HDF5_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RGB_OUTPUT_DIR,
        help=f"RGB output folder (default: {DEFAULT_RGB_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite RGB files that already exist.",
    )
    return parser.parse_args()


def decode_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def load_metadata(source: h5py.File) -> dict:
    metadata_json = source.attrs.get("metadata_json")
    if metadata_json is None:
        return {}
    try:
        return json.loads(decode_text(metadata_json))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def safe_filename(name: str, index: int, suffix: str) -> str:
    filename = Path(name).name
    if filename in ("", ".", ".."):
        filename = f"frame_{index + 1:06d}{suffix}"
    if not Path(filename).suffix:
        filename += suffix
    return filename


def extract_rgb(hdf5_path: Path, output_dir: Path | None, overwrite: bool) -> int:
    hdf5_path = hdf5_path.resolve()
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as source:
        if "rgb" not in source:
            raise ValueError(f"The HDF5 file has no 'rgb' dataset: {hdf5_path}")

        metadata = load_metadata(source)
        rgb_directory = str(metadata.get("rgb_directory", "RGB"))
        destination = (
            output_dir.resolve()
            if output_dir is not None
            else hdf5_path.parent / rgb_directory
        )
        destination.mkdir(parents=True, exist_ok=True)

        rgb_dataset = source["rgb"]
        filenames = source.get("filenames")
        encoded_images = int(source.attrs.get("format_version", 1)) >= 2
        rgb_pattern = str(metadata.get("rgb_pattern", "frame_%06d.jpg"))
        default_suffix = Path(rgb_pattern.replace("%06d", "000001")).suffix or ".jpg"

        print(f"Reading RGB dataset from: {hdf5_path}")
        print(f"Saving RGB images to:     {destination}")

        extracted = 0
        skipped = 0
        for index in range(len(rgb_dataset)):
            stored_name = (
                decode_text(filenames[index])
                if filenames is not None
                else f"frame_{index + 1:06d}{default_suffix}"
            )
            filename = safe_filename(stored_name, index, default_suffix)
            output_path = destination / filename

            if output_path.exists() and not overwrite:
                skipped += 1
                continue

            if encoded_images:
                image_bytes = bytes(rgb_dataset[index])
                if not image_bytes:
                    raise ValueError(f"RGB frame {index + 1} is empty.")
                output_path.write_bytes(image_bytes)
            else:
                rgb = np.asarray(rgb_dataset[index])
                if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
                    raise ValueError(
                        f"Invalid RGB frame {index + 1}: "
                        f"shape={rgb.shape}, dtype={rgb.dtype}"
                    )
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(output_path), bgr):
                    raise OSError(f"Could not save RGB image: {output_path}")

            extracted += 1
            print(
                f"\rExtracted RGB images: {extracted} | skipped: {skipped}",
                end="",
                flush=True,
            )

    print(f"\nDone. Extracted: {extracted}; skipped: {skipped}.")
    return extracted


def main() -> int:
    args = parse_args()
    try:
        extract_rgb(args.h5, args.output, args.overwrite)
    except (OSError, ValueError, KeyError) as exc:
        print(f"RGB extraction failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
