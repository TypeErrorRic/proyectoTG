"""Capture synchronized RGB and raw depth images from an Intel RealSense.

New recordings are stored as lossless, one-to-one frame pairs:

    tests/video/videos/RGB/frame_000001.png
    tests/video/videos/depth/frame_000001.png
    tests/video/videos/capture_metadata.json

RGB is stored as an 8-bit PNG. Depth is stored exactly as delivered by the
aligned RealSense Z16 stream (uint16). The JET colour map is used only for the
live preview and is never stored as depth data.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

try:
    import cv2
except ImportError as exc:
    raise SystemExit(
        "opencv-python is required. Install OpenCV before running this script."
    ) from exc

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as exc:
    raise SystemExit(
        "pyrealsense2 is required. Install Intel RealSense SDK/librealsense "
        "and the Python bindings before running this script."
    ) from exc


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "videos"
DEFAULT_RECORDING_FPS = 30
FRAME_PATTERN = "frame_%06d.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture synchronized RGB and raw Z16 depth image pairs."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output folder for the synchronized RGB-D data.",
    )
    parser.add_argument("--width", type=int, default=640, help="Frame width.")
    parser.add_argument("--height", type=int, default=480, help="Frame height.")
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_RECORDING_FPS,
        help="Camera capture FPS.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Number of pairs to capture. Use 0 to capture until Ctrl+C.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Maximum capture duration in seconds. Use 0 for no time limit.",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Initial camera frames to discard for exposure warmup.",
    )
    parser.add_argument(
        "--rgb-dir",
        default="RGB",
        help="Folder name for RGB PNG images.",
    )
    parser.add_argument(
        "--depth-dir",
        default="depth",
        help="Folder name for raw uint16 depth PNG images.",
    )
    parser.add_argument(
        "--depth-alpha",
        type=float,
        default=0.03,
        help="Scale used only for the coloured depth preview.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_false",
        dest="preview",
        help="Capture without showing the live preview window.",
    )
    parser.set_defaults(preview=True)
    return parser.parse_args()


def assert_realsense_device_available() -> None:
    context = rs.context()
    devices = context.query_devices()
    if len(devices) == 0:
        raise RuntimeError(
            "No RealSense camera was detected. Check the USB connection and "
            "make sure the camera is visible inside the runtime environment."
        )

    for index, device in enumerate(devices):
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        print(f"Detected RealSense camera {index + 1}: {name} [{serial}]")


def configure_pipeline(
    width: int,
    height: int,
    fps: int,
) -> tuple[rs.pipeline, rs.align]:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    try:
        pipeline.start(config)
    except RuntimeError as exc:
        raise RuntimeError(
            "Could not start the RealSense stream with "
            f"{width}x{height} at {fps} FPS. Try --width 640 --height 480 "
            "--fps 30 or inspect supported profiles in RealSense Viewer."
        ) from exc
    return pipeline, rs.align(rs.stream.color)


def stream_intrinsics_to_dict(
    pipeline: rs.pipeline,
    stream: rs.stream,
) -> dict:
    profile = (
        pipeline.get_active_profile()
        .get_stream(stream)
        .as_video_stream_profile()
    )
    intrinsics = profile.get_intrinsics()
    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "ppx": intrinsics.ppx,
        "ppy": intrinsics.ppy,
        "model": str(intrinsics.model),
        "coeffs": list(intrinsics.coeffs),
    }


def get_depth_scale(pipeline: rs.pipeline) -> float:
    return float(
        pipeline.get_active_profile()
        .get_device()
        .first_depth_sensor()
        .get_depth_scale()
    )


def write_capture_metadata(
    path: Path,
    pipeline: rs.pipeline,
    args: argparse.Namespace,
) -> None:
    metadata = {
        "fps": args.fps,
        "frame_count": 0,
        "rgb_storage": "png_sequence",
        "rgb_directory": args.rgb_dir,
        "rgb_pattern": FRAME_PATTERN,
        "depth_storage": "uint16_png_sequence",
        "depth_directory": args.depth_dir,
        "depth_pattern": FRAME_PATTERN,
        "depth_scale": get_depth_scale(pipeline),
        "depth_aligned_to": "color",
        "color_intrinsics": stream_intrinsics_to_dict(
            pipeline,
            rs.stream.color,
        ),
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def update_frame_count(metadata_path: Path, frame_count: int) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["frame_count"] = int(frame_count)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def prepare_capture_dirs(
    output_dir: Path,
    rgb_dir_name: str,
    depth_dir_name: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = output_dir / rgb_dir_name
    depth_dir = output_dir / depth_dir_name

    for legacy_name in ("rgb.avi", "depth_map.avi"):
        legacy_path = output_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    for directory in (rgb_dir, depth_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)

    return rgb_dir, depth_dir


def extract_frame_pair(
    rgb_frame: rs.video_frame,
    depth_frame: rs.depth_frame,
) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asanyarray(rgb_frame.get_data())
    depth_raw = np.asanyarray(depth_frame.get_data())
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if depth_raw.dtype != np.uint16:
        raise RuntimeError(
            f"Expected Z16 depth data, received dtype={depth_raw.dtype}."
        )
    return rgb_bgr, depth_raw


def save_frame_pair(
    rgb_dir: Path,
    depth_dir: Path,
    frame_number: int,
    rgb_bgr: np.ndarray,
    depth_raw: np.ndarray,
) -> None:
    filename = FRAME_PATTERN % frame_number
    rgb_path = rgb_dir / filename
    depth_path = depth_dir / filename

    if not cv2.imwrite(str(rgb_path), rgb_bgr):
        raise RuntimeError(f"Could not save RGB frame: {rgb_path}")
    if not cv2.imwrite(str(depth_path), depth_raw):
        rgb_path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not save raw depth frame: {depth_path}")


def make_depth_preview(
    depth_raw: np.ndarray,
    depth_alpha: float,
) -> np.ndarray:
    depth_8bit = cv2.convertScaleAbs(depth_raw, alpha=depth_alpha)
    return cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)


def capture_frames(args: argparse.Namespace) -> int:
    output_dir = args.output.resolve()
    rgb_dir, depth_dir = prepare_capture_dirs(
        output_dir,
        args.rgb_dir,
        args.depth_dir,
    )
    metadata_path = output_dir / "capture_metadata.json"

    assert_realsense_device_available()
    pipeline, align = configure_pipeline(args.width, args.height, args.fps)
    write_capture_metadata(metadata_path, pipeline, args)

    captured = 0
    observed = 0
    started_at = time.monotonic()

    print(f"Saving RGB frames to:       {rgb_dir}")
    print(f"Saving raw depth frames to: {depth_dir}")
    print(f"Capturing at {args.fps} FPS.")
    if args.preview:
        print("JET colours are preview-only. Press Q or Esc to stop.")
    else:
        print("Press Ctrl+C to stop.")

    try:
        while True:
            if args.duration > 0 and time.monotonic() - started_at >= args.duration:
                break
            if args.frames > 0 and captured >= args.frames:
                break

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            rgb_frame = aligned_frames.get_color_frame()
            if not depth_frame or not rgb_frame:
                continue

            observed += 1
            if observed <= args.skip:
                continue

            rgb_bgr, depth_raw = extract_frame_pair(rgb_frame, depth_frame)
            frame_number = captured + 1
            save_frame_pair(
                rgb_dir,
                depth_dir,
                frame_number,
                rgb_bgr,
                depth_raw,
            )
            captured = frame_number
            print(f"\rCaptured synchronized pairs: {captured}", end="", flush=True)

            if args.preview:
                depth_preview = make_depth_preview(depth_raw, args.depth_alpha)
                preview = np.hstack((rgb_bgr, depth_preview))
                cv2.imshow(
                    "RealSense capture: RGB | Depth preview (not stored)",
                    preview,
                )
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("\nCapture stopped from preview window.")
                    break
    except KeyboardInterrupt:
        print("\nCapture stopped by user.")
    finally:
        if args.preview:
            cv2.destroyAllWindows()
        pipeline.stop()
        update_frame_count(metadata_path, captured)

    print(f"\nDone. Captured {captured} synchronized RGB-D pairs.")
    return captured


def main() -> int:
    args = parse_args()
    if args.frames < 0:
        print("--frames must be 0 or greater.", file=sys.stderr)
        return 2
    if args.duration < 0:
        print("--duration must be 0 or greater.", file=sys.stderr)
        return 2
    if args.skip < 0:
        print("--skip must be 0 or greater.", file=sys.stderr)
        return 2
    if args.fps <= 0:
        print("--fps must be greater than 0.", file=sys.stderr)
        return 2
    if args.depth_alpha <= 0:
        print("--depth-alpha must be greater than 0.", file=sys.stderr)
        return 2

    capture_frames(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
