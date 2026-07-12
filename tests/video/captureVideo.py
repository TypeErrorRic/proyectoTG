"""Record synchronized RGB-D videos from an Intel RealSense camera.

The script saves synchronized videos in tests/videos by default:

    rgb.avi
    depth_map.avi

The depth video is a visual depth map generated from the raw z16 stream.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import cv2
except ImportError as exc:
    raise SystemExit(
        "opencv-python is required to write video files. Install OpenCV before "
        "running this script."
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record synchronized RGB and depth-map videos from a RealSense camera."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output folder for the RGB and depth-map videos.",
    )
    parser.add_argument("--width", type=int, default=640, help="Frame width.")
    parser.add_argument("--height", type=int, default=480, help="Frame height.")
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_RECORDING_FPS,
        help="Camera and video FPS.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Number of frames to record. Use 0 to record until Ctrl+C.",
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
        help="Frames to discard after starting the camera, useful for exposure warmup.",
    )
    parser.add_argument(
        "--rgb-name",
        default="rgb.avi",
        help="Filename for the RGB video.",
    )
    parser.add_argument(
        "--depth-name",
        default="depth_map.avi",
        help="Filename for the visual depth-map video.",
    )
    parser.add_argument(
        "--depth-alpha",
        type=float,
        default=0.03,
        help="Scale used to convert raw depth to an 8-bit visual map.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        default=True,
        help="Show a live RGB/depth preview window while recording.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_false",
        dest="preview",
        help="Record without showing the live preview window.",
    )
    return parser.parse_args()


def create_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def assert_realsense_device_available() -> None:
    context = rs.context()
    devices = context.query_devices()
    if len(devices) == 0:
        raise RuntimeError(
            "No RealSense camera was detected. Check the USB connection and, "
            "if you are using Docker, make sure the camera is visible inside "
            "the container."
        )

    for index, device in enumerate(devices):
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        print(f"Detected RealSense camera {index + 1}: {name} [{serial}]")


def configure_pipeline(width: int, height: int, fps: int) -> tuple[rs.pipeline, rs.align]:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    try:
        pipeline.start(config)
    except RuntimeError as exc:
        raise RuntimeError(
            "Could not start the RealSense stream with "
            f"{width}x{height} at {fps} FPS. Try a supported profile such as "
            "--width 640 --height 480 --fps 30, or check the camera profiles "
            "with the Intel RealSense Viewer."
        ) from exc
    return pipeline, rs.align(rs.stream.color)


def stream_intrinsics_to_dict(pipeline: rs.pipeline, stream: rs.stream) -> dict:
    profile = pipeline.get_active_profile().get_stream(stream).as_video_stream_profile()
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


def write_capture_metadata(path: Path, pipeline: rs.pipeline, fps: int, depth_alpha: float) -> None:
    metadata = {
        "fps": fps,
        "depth_video": "visual_colormap",
        "depth_alpha": depth_alpha,
        "color_intrinsics": stream_intrinsics_to_dict(pipeline, rs.stream.color),
        "depth_intrinsics": stream_intrinsics_to_dict(pipeline, rs.stream.depth),
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def create_video_writer(path: Path, fps: int, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def frames_to_video_images(
    rgb_frame: rs.video_frame,
    depth_frame: rs.depth_frame,
    depth_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    rgb_image = np.asanyarray(rgb_frame.get_data())
    depth_raw = np.asanyarray(depth_frame.get_data())

    rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    depth_8bit = cv2.convertScaleAbs(depth_raw, alpha=depth_alpha)
    depth_map = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
    return rgb_bgr, depth_map


def record_videos(args: argparse.Namespace) -> int:
    output_dir = create_output_dir(args.output)
    rgb_path = output_dir / args.rgb_name
    depth_path = output_dir / args.depth_name
    assert_realsense_device_available()
    pipeline, align = configure_pipeline(args.width, args.height, args.fps)
    write_capture_metadata(output_dir / "capture_metadata.json", pipeline, args.fps, args.depth_alpha)
    rgb_writer = create_video_writer(rgb_path, args.fps, args.width, args.height)
    depth_writer = create_video_writer(depth_path, args.fps, args.width, args.height)
    recorded_frames = 0
    observed_frames = 0
    start_time = time.monotonic()

    print(f"Recording RGB video to:       {rgb_path}")
    print(f"Recording depth-map video to: {depth_path}")
    print(f"Recording at {args.fps} photograms per second.")
    if args.preview:
        print("Preview enabled. Press q in the preview window or Ctrl+C to stop.")
    else:
        print("Press Ctrl+C to stop recording.")

    try:
        while True:
            if args.duration > 0 and time.monotonic() - start_time >= args.duration:
                break
            if args.frames > 0 and recorded_frames >= args.frames:
                break

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            rgb_frame = aligned_frames.get_color_frame()

            if not depth_frame or not rgb_frame:
                continue

            observed_frames += 1
            if observed_frames <= args.skip:
                continue

            rgb_bgr, depth_map = frames_to_video_images(
                rgb_frame=rgb_frame,
                depth_frame=depth_frame,
                depth_alpha=args.depth_alpha,
            )
            rgb_writer.write(rgb_bgr)
            depth_writer.write(depth_map)

            recorded_frames += 1
            print(f"\rRecorded photograms: {recorded_frames}", end="", flush=True)

            if args.preview:
                preview = np.hstack((rgb_bgr, depth_map))
                cv2.imshow("RealSense recording: RGB | Depth map", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nRecording stopped from preview window.")
                    break
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    finally:
        rgb_writer.release()
        depth_writer.release()
        if args.preview:
            cv2.destroyAllWindows()
        pipeline.stop()

    print(f"\nDone. Recorded {recorded_frames} synchronized RGB-D photograms.")
    return recorded_frames


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

    record_videos(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
