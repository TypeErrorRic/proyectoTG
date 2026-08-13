"""Convert a RealSense PNG capture into one HDF5 dataset.

The script reads ``videos/RGB``, ``videos/depth``, and
``videos/capture_metadata.json`` and creates ``videos/capture.h5``.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import cv2


VIDEO_DIR = Path(__file__).resolve().parent.parent / "videos"
METADATA_PATH = VIDEO_DIR / "capture_metadata.json"
OUTPUT_PATH = VIDEO_DIR / "capture.h5"


def load_metadata(path: Path = METADATA_PATH) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("capture_metadata.json must contain a JSON object.")
    return metadata


def find_frame_pairs(
    metadata: dict,
    video_dir: Path = VIDEO_DIR,
) -> list[tuple[Path, Path]]:
    rgb_dir = video_dir / str(metadata["rgb_directory"])
    depth_dir = video_dir / str(metadata["depth_directory"])
    rgb_pattern = str(metadata["rgb_pattern"])
    rgb_suffix = Path(rgb_pattern.replace("%06d", "000001")).suffix
    rgb_frames = sorted(rgb_dir.glob(f"frame_*{rgb_suffix}"))

    if not rgb_frames:
        raise FileNotFoundError(f"No RGB frames found in: {rgb_dir}")

    pairs = []
    for rgb_path in rgb_frames:
        depth_path = depth_dir / f"{rgb_path.stem}.png"
        if not depth_path.is_file():
            raise FileNotFoundError(
                f"Depth frame missing for {rgb_path.name}: {depth_path}"
            )
        pairs.append((rgb_path, depth_path))

    depth_names = {path.stem for path in depth_dir.glob("frame_*.png")}
    rgb_names = {path.stem for path in rgb_frames}
    unmatched_depth = sorted(depth_names - rgb_names)
    if unmatched_depth:
        raise ValueError(
            f"Found {len(unmatched_depth)} depth frame(s) without an RGB pair."
        )
    return pairs


def read_pair(rgb_path: Path, depth_path: Path) -> tuple[np.ndarray, np.ndarray]:
    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb_bgr is None:
        raise ValueError(f"Could not read RGB image: {rgb_path}")
    if depth is None:
        raise ValueError(f"Could not read depth image: {depth_path}")
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError(
            f"Expected a uint16 depth image, got {depth.dtype} {depth.shape}: "
            f"{depth_path}"
        )
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != depth.shape:
        raise ValueError(
            f"RGB and depth dimensions differ for {rgb_path.name}: "
            f"{rgb.shape[:2]} versus {depth.shape}"
        )
    return rgb, depth


def write_hdf5(
    metadata: dict,
    pairs: list[tuple[Path, Path]],
    output_path: Path = OUTPUT_PATH,
) -> None:
    frame_count = len(pairs)
    temporary_path = output_path.with_suffix(".h5.tmp")
    temporary_path.unlink(missing_ok=True)

    try:
        with h5py.File(temporary_path, "w") as output:
            encoded_dtype = h5py.vlen_dtype(np.dtype("uint8"))
            rgb_dataset = output.create_dataset(
                "rgb",
                shape=(frame_count,),
                dtype=encoded_dtype,
            )
            depth_dataset = output.create_dataset(
                "depth",
                shape=(frame_count,),
                dtype=encoded_dtype,
            )
            filenames = output.create_dataset(
                "filenames",
                shape=(frame_count,),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )

            for index, (rgb_path, depth_path) in enumerate(pairs):
                read_pair(rgb_path, depth_path)
                rgb_dataset[index] = np.frombuffer(rgb_path.read_bytes(), dtype=np.uint8)
                depth_dataset[index] = np.frombuffer(
                    depth_path.read_bytes(), dtype=np.uint8
                )
                filenames[index] = rgb_path.name
                print(
                    f"\rWriting synchronized pairs: {index + 1}/{frame_count}",
                    end="",
                    flush=True,
                )

            stored_metadata = dict(metadata)
            stored_metadata["frame_count"] = frame_count
            output.attrs["format_version"] = 2
            output.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            output.attrs["rgb_color_order"] = "RGB"
            output.attrs["rgb_encoding"] = "jpeg"
            output.attrs["depth_encoding"] = "png"
            output.attrs["depth_dtype"] = "uint16"
            output.attrs["metadata_json"] = json.dumps(stored_metadata)

        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def create_hdf5(video_dir: Path = VIDEO_DIR) -> Path:
    video_dir = video_dir.resolve()
    output_path = video_dir / "capture.h5"
    metadata = load_metadata(video_dir / "capture_metadata.json")
    pairs = find_frame_pairs(metadata, video_dir)
    write_hdf5(metadata, pairs, output_path)
    print(f"\nCreated {output_path} with {len(pairs)} RGB-D pairs.")
    return output_path


def main() -> int:
    try:
        create_hdf5()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"HDF5 creation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
