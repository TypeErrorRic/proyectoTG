"""Process recorded RGB/depth videos and save RGB + segmentation overlays.

This script follows the offline processing pattern used by tests/evaluate_nyu_v2.py:
it injects an in-memory dataset loader, calls segmentacion in "prueba" mode, and
saves only the RGB frame and the segmentation overlay.

Output structure by default:

    tests/video/data/RGB/frame_000001.png
    tests/video/data/overlay/frame_000001.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import cupy as cp
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from application.gestorFotogramas import dataset_frames  # noqa: E402
from application.segmentacion import segmentacion  # noqa: E402


DEFAULT_VIDEO_DIR = THIS_DIR / "videos"
DEFAULT_OUTPUT_DIR = THIS_DIR / "data"
DEFAULT_TARGET_FPS = 10.0


class VideoFrameCache:
    """Small cache that exposes video frames through the dataset loader API."""

    def __init__(self) -> None:
        self.index: Optional[int] = None
        self.rgb_bgr: Optional[np.ndarray] = None
        self.depth: Optional[np.ndarray] = None

    def set_frame(self, index: int, rgb_bgr: np.ndarray, depth: np.ndarray) -> None:
        self.index = int(index)
        self.rgb_bgr = rgb_bgr
        self.depth = depth

    def load(self, index=None):
        if self.rgb_bgr is None or self.depth is None:
            return None, None
        return self.rgb_bgr, self.depth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process recorded RGB/depth videos and save RGB + overlay frames."
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=DEFAULT_VIDEO_DIR,
        help="Folder containing the recorded videos.",
    )
    parser.add_argument("--rgb-video", default="rgb.avi", help="RGB video filename.")
    parser.add_argument(
        "--depth-video",
        default="depth_map.avi",
        help="Depth-map video filename.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output folder where RGB/ and overlay/ will be created.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to capture_metadata.json. Defaults to video-dir/capture_metadata.json.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="First number used in output filenames.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to process. Use 0 to process until the videos end.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=DEFAULT_TARGET_FPS,
        help="Temporal subsampling target FPS for segmentation output.",
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=2,
        help="Retries when segmentation does not return an overlay.",
    )
    return parser.parse_args()


def open_video(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return capture


def compute_sample_every(source_fps: float, target_fps: float) -> int:
    if target_fps <= 0:
        raise ValueError("--target-fps must be greater than 0.")
    if source_fps <= 0:
        return 1
    return max(1, int(round(source_fps / target_fps)))


def create_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    rgb_dir = output_dir / "RGB"
    overlay_dir = output_dir / "overlay"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    return rgb_dir, overlay_dir


def depth_video_frame_to_depth(depth_frame: np.ndarray, rgb_shape: tuple[int, int]) -> np.ndarray:
    """Convert the visual depth-map video frame to a normalized depth matrix."""
    if depth_frame.ndim == 3:
        depth = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
    else:
        depth = depth_frame

    if depth.shape[:2] != rgb_shape:
        depth = cv2.resize(
            depth,
            (rgb_shape[1], rgb_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    return depth.astype(np.float32)


def load_color_intrinsics(metadata_path: Path) -> dict:
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Camera metadata not found: {metadata_path}. "
            "Record again with captureVideo.py so capture_metadata.json is created."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    intrinsics = metadata.get("color_intrinsics")
    if not isinstance(intrinsics, dict):
        raise ValueError(f"Missing color_intrinsics in {metadata_path}")
    for key in ("width", "height", "fx", "fy", "ppx", "ppy"):
        if key not in intrinsics:
            raise ValueError(f"Missing color_intrinsics.{key} in {metadata_path}")
    return intrinsics


def compute_rays_from_intrinsics(intrinsics: dict, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    ppx = float(intrinsics["ppx"])
    ppy = float(intrinsics["ppy"])

    intr_w = int(intrinsics["width"])
    intr_h = int(intrinsics["height"])
    scale_x = w / float(intr_w)
    scale_y = h / float(intr_h)

    fx *= scale_x
    fy *= scale_y
    ppx *= scale_x
    ppy *= scale_y

    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    x = (uu - ppx) / fx
    y = (vv - ppy) / fy
    ones = np.ones_like(x, dtype=np.float32)
    return np.stack([x, y, ones], axis=-1).astype(np.float32)


def apply_camera_intrinsics_to_runtime(intrinsics: dict, rgb_shape: tuple[int, int]) -> None:
    impl = segmentacion._obtener_impl()
    h, w = rgb_shape
    rays_np = compute_rays_from_intrinsics(intrinsics, (h, w))
    impl["H"] = h
    impl["W"] = w
    impl["rays_cp"] = cp.asarray(rays_np)
    impl["mode"] = "camera"


def write_frame(path: Path, frame: np.ndarray, label: str) -> None:
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Could not save {label}: {path}")


def process_current_frame(
    cache: VideoFrameCache,
    frame_index: int,
    retry: int,
    intrinsics: dict,
) -> Optional[np.ndarray]:
    dataset_frames.load_dataset_frame = cache.load
    ok = segmentacion.preprocesar(mode="prueba", dataset_index=frame_index)
    if not ok or cache.rgb_bgr is None:
        return None
    apply_camera_intrinsics_to_runtime(intrinsics, cache.rgb_bgr.shape[:2])

    overlay = None
    attempts = max(0, int(retry)) + 1
    for _ in range(attempts):
        try:
            overlay = segmentacion.segmentar()
        except Exception:
            overlay = None
        if overlay is not None:
            break

    return overlay


def process_videos(args: argparse.Namespace) -> int:
    if args.start_index < 0:
        raise ValueError("--start-index must be 0 or greater.")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be 0 or greater.")
    if args.target_fps <= 0:
        raise ValueError("--target-fps must be greater than 0.")

    rgb_path = args.video_dir / args.rgb_video
    depth_path = args.video_dir / args.depth_video
    metadata_path = args.metadata if args.metadata is not None else args.video_dir / "capture_metadata.json"
    intrinsics = load_color_intrinsics(metadata_path)
    rgb_dir, overlay_dir = create_output_dirs(args.output)

    rgb_capture = open_video(rgb_path)
    depth_capture = open_video(depth_path)
    source_fps = rgb_capture.get(cv2.CAP_PROP_FPS)
    sample_every = compute_sample_every(source_fps, args.target_fps)
    cache = VideoFrameCache()
    processed = 0
    read_frames = 0

    print(f"Reading RGB video:        {rgb_path}")
    print(f"Reading depth-map video:  {depth_path}")
    print(f"Using camera metadata:    {metadata_path}")
    print(f"Source FPS:               {source_fps:.2f}" if source_fps > 0 else "Source FPS:               unknown")
    print(f"Target processing FPS:    {args.target_fps:.2f}")
    print(f"Processing every:         {sample_every} frame(s)")
    print(f"Saving RGB frames to:     {rgb_dir}")
    print(f"Saving overlays to:       {overlay_dir}")

    segmentacion.detener_hilo_secundario()
    segmentacion.inicializar(mode="prueba")

    try:
        while True:
            if args.max_frames > 0 and processed >= args.max_frames:
                break

            ok_rgb, rgb_frame = rgb_capture.read()
            ok_depth, depth_frame = depth_capture.read()
            if not ok_rgb or not ok_depth:
                break
            read_frames += 1

            if (read_frames - 1) % sample_every != 0:
                continue

            frame_number = args.start_index + processed
            filename = f"frame_{frame_number:06d}.png"
            depth = depth_video_frame_to_depth(depth_frame, rgb_frame.shape[:2])
            cache.set_frame(frame_number, rgb_frame, depth)

            overlay = process_current_frame(cache, frame_number, args.retry, intrinsics)
            if overlay is None:
                overlay = rgb_frame

            write_frame(rgb_dir / filename, rgb_frame, "RGB frame")
            write_frame(overlay_dir / filename, overlay, "overlay frame")

            processed += 1
            print(f"\rProcessed frames: {processed}", end="", flush=True)
    finally:
        rgb_capture.release()
        depth_capture.release()
        segmentacion.detener_hilo_secundario()

    print(f"\nDone. Processed {processed} synchronized frame pairs.")
    return processed


def main() -> int:
    args = parse_args()
    process_videos(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
