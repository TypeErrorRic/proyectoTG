"""Delete the recorded RGB video, depth video, and capture metadata."""

from __future__ import annotations

import argparse
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = THIS_DIR / "videos"
RECORDED_FILES = (
    "rgb.avi",
    "depth_map.avi",
    "capture_metadata.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete recorded video data from tests/video/videos."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Delete without asking for confirmation.",
    )
    return parser.parse_args()


def clean_recorded_data(videos_dir: Path) -> int:
    deleted = 0

    for filename in RECORDED_FILES:
        path = videos_dir / filename
        if path.is_file() or path.is_symlink():
            path.unlink()
            print(f"Deleted: {path}")
            deleted += 1
        else:
            print(f"Not found: {path}")

    return deleted


def main() -> int:
    args = parse_args()
    videos_dir = VIDEOS_DIR.resolve()

    if not args.yes:
        filenames = ", ".join(RECORDED_FILES)
        answer = input(
            f"Delete {filenames} from '{videos_dir}'? [y/N]: "
        )
        if answer.strip().lower() not in {"y", "yes"}:
            print("Cleanup cancelled.")
            return 0

    deleted = clean_recorded_data(videos_dir)
    print(f"Cleanup complete. Deleted {deleted} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
