"""Capture synchronized RGB and raw depth images from an Intel RealSense.

New recordings are stored as synchronized, one-to-one frame pairs:

    tests/video/videos/RGB/frame_000001.jpg
    tests/video/videos/depth/frame_000001.png
    tests/video/videos/capture_metadata.json

RGB is stored as a configurable-quality JPEG. Depth is stored losslessly exactly as
delivered by the aligned RealSense Z16 stream (uint16 PNG). The JET colour map
is used only for the live preview and is never stored as depth data.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

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


VIDEO_TEST_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = VIDEO_TEST_DIR / "videos"
DEFAULT_CONFIG_PATH = VIDEO_TEST_DIR / "config" / "capture_config.json"
DEFAULT_RECORDING_FPS = 30
DEFAULT_SAVED_FPS = 10.0


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> SimpleNamespace:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Capture configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {path} at line {exc.lineno}, column {exc.colno}."
        ) from exc

    if not isinstance(config, dict):
        raise ValueError(f"Capture configuration must be a JSON object: {path}")

    defaults = {
        "output": str(DEFAULT_OUTPUT_DIR),
        "width": 640,
        "height": 480,
        "fps": DEFAULT_RECORDING_FPS,
        "target_fps": DEFAULT_SAVED_FPS,
        "frames": 0,
        "duration": 0.0,
        "skip": 0,
        "rgb_dir": "RGB",
        "depth_dir": "depth",
        "rgb_pattern": "frame_%06d.jpg",
        "depth_pattern": "frame_%06d.png",
        "depth_alpha": 0.03,
        "rgb_jpeg_quality": 90,
        "depth_png_compression": 3,
        "preview": True,
    }
    unknown = sorted(set(config) - set(defaults))
    if unknown:
        raise ValueError(f"Unknown capture setting(s): {', '.join(unknown)}")

    settings = {**defaults, **config}
    output = Path(settings["output"])
    if not output.is_absolute():
        output = (path.parent / output).resolve()
    settings["output"] = output
    return SimpleNamespace(**settings)


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
    args: SimpleNamespace,
    saved_fps: float,
    sample_every: int,
) -> None:
    metadata = {
        "fps": saved_fps,
        "camera_fps": args.fps,
        "sample_every": sample_every,
        "frame_count": 0,
        "rgb_storage": "jpeg_sequence",
        "rgb_directory": args.rgb_dir,
        "rgb_pattern": args.rgb_pattern,
        "rgb_jpeg_quality": args.rgb_jpeg_quality,
        "depth_storage": "uint16_png_sequence",
        "depth_directory": args.depth_dir,
        "depth_pattern": args.depth_pattern,
        "depth_png_compression": args.depth_png_compression,
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
        directory.mkdir(parents=True, exist_ok=True)
        for frame_path in directory.glob("frame_*.*"):
            try:
                frame_path.unlink()
            except PermissionError as exc:
                raise PermissionError(
                    f"Could not remove the previous frame {frame_path}. "
                    "Close any program using that file and check its Windows "
                    "permissions."
                ) from exc

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
    rgb_pattern: str,
    depth_pattern: str,
    rgb_jpeg_quality: int,
    depth_png_compression: int,
) -> None:
    rgb_path = rgb_dir / (rgb_pattern % frame_number)
    depth_path = depth_dir / (depth_pattern % frame_number)

    if not cv2.imwrite(
        str(rgb_path),
        rgb_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, rgb_jpeg_quality],
    ):
        raise RuntimeError(f"Could not save RGB frame: {rgb_path}")
    if not cv2.imwrite(
        str(depth_path),
        depth_raw,
        [cv2.IMWRITE_PNG_COMPRESSION, depth_png_compression],
    ):
        rgb_path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not save raw depth frame: {depth_path}")


def make_depth_preview(
    depth_raw: np.ndarray,
    depth_alpha: float,
) -> np.ndarray:
    depth_8bit = cv2.convertScaleAbs(depth_raw, alpha=depth_alpha)
    return cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)


def compute_sample_every(camera_fps: float, target_fps: float) -> int:
    if camera_fps <= 0 or target_fps <= 0:
        raise ValueError("Camera FPS and target FPS must be greater than 0.")
    return max(1, int(round(camera_fps / target_fps)))


def capture_frames(args: SimpleNamespace) -> int:
    output_dir = args.output.resolve()
    rgb_dir, depth_dir = prepare_capture_dirs(
        output_dir,
        args.rgb_dir,
        args.depth_dir,
    )
    metadata_path = output_dir / "capture_metadata.json"

    assert_realsense_device_available()
    pipeline, align = configure_pipeline(args.width, args.height, args.fps)
    sample_every = compute_sample_every(args.fps, args.target_fps)
    saved_fps = args.fps / float(sample_every)
    write_capture_metadata(
        metadata_path,
        pipeline,
        args,
        saved_fps,
        sample_every,
    )

    captured = 0
    observed = 0
    eligible = 0
    started_at = time.monotonic()

    print(f"Saving RGB frames to:       {rgb_dir}")
    print(f"Saving raw depth frames to: {depth_dir}")
    print(f"Camera stream FPS:           {args.fps}")
    print(f"Requested saved FPS:         {args.target_fps:.2f}")
    print(f"Actual saved FPS:            {saved_fps:.2f}")
    print(f"Saving every:                {sample_every} camera frame(s)")
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
            eligible += 1
            if (eligible - 1) % sample_every != 0:
                continue

            rgb_bgr, depth_raw = extract_frame_pair(rgb_frame, depth_frame)
            frame_number = captured + 1
            save_frame_pair(
                rgb_dir,
                depth_dir,
                frame_number,
                rgb_bgr,
                depth_raw,
                args.rgb_pattern,
                args.depth_pattern,
                args.rgb_jpeg_quality,
                args.depth_png_compression,
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
    try:
        args = load_config()
    except (OSError, TypeError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    for name in (
        "width",
        "height",
        "fps",
        "frames",
        "skip",
        "rgb_jpeg_quality",
        "depth_png_compression",
    ):
        if not isinstance(getattr(args, name), int):
            print(f"{name} must be an integer.", file=sys.stderr)
            return 2
    for name in ("target_fps", "duration", "depth_alpha"):
        if not isinstance(getattr(args, name), (int, float)):
            print(f"{name} must be a number.", file=sys.stderr)
            return 2
    for name in ("preview",):
        if not isinstance(getattr(args, name), bool):
            print(f"{name} must be true or false.", file=sys.stderr)
            return 2
    if args.frames < 0:
        print("frames must be 0 or greater.", file=sys.stderr)
        return 2
    if args.duration < 0:
        print("duration must be 0 or greater.", file=sys.stderr)
        return 2
    if args.skip < 0:
        print("skip must be 0 or greater.", file=sys.stderr)
        return 2
    if args.fps <= 0:
        print("fps must be greater than 0.", file=sys.stderr)
        return 2
    if args.target_fps <= 0:
        print("target_fps must be greater than 0.", file=sys.stderr)
        return 2
    if args.depth_alpha <= 0:
        print("depth_alpha must be greater than 0.", file=sys.stderr)
        return 2
    if not 0 <= args.rgb_jpeg_quality <= 100:
        print("rgb_jpeg_quality must be between 0 and 100.", file=sys.stderr)
        return 2
    if not 0 <= args.depth_png_compression <= 9:
        print("depth_png_compression must be between 0 and 9.", file=sys.stderr)
        return 2
    try:
        rgb_example = args.rgb_pattern % 1
        depth_example = args.depth_pattern % 1
    except (TypeError, ValueError) as exc:
        print(f"Invalid frame pattern: {exc}", file=sys.stderr)
        return 2
    if Path(rgb_example).suffix.lower() not in (".jpg", ".jpeg"):
        print("rgb_pattern must produce a .jpg or .jpeg filename.", file=sys.stderr)
        return 2
    if Path(depth_example).suffix.lower() != ".png":
        print("depth_pattern must produce a .png filename.", file=sys.stderr)
        return 2

    captured = capture_frames(args)
    if captured == 0:
        print("No frames were captured; HDF5 creation was skipped.")
        return 0

    print("Creating compressed HDF5 capture...")
    try:
        from createHDF5 import create_hdf5

        hdf5_path = create_hdf5(args.output.resolve())
    except (ImportError, OSError, TypeError, ValueError) as exc:
        print(f"HDF5 creation failed: {exc}", file=sys.stderr)
        return 1
    print(f"HDF5 capture ready: {hdf5_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
