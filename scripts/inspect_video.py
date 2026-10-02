from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show video metadata for a thesis traffic recording."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video file.",
    )
    args = parser.parse_args()

    video_path = Path(args.video)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    duration_seconds = frame_count / fps if fps > 0 else 0.0
    duration_minutes = duration_seconds / 60.0

    capture.release()

    print(f"Video file : {video_path.name}")
    print(f"Resolution : {width} x {height}")
    print(f"FPS        : {fps:.2f}")
    print(f"Frames     : {frame_count:,}")
    print(f"Duration   : {duration_seconds:.2f} seconds ({duration_minutes:.2f} minutes)")


if __name__ == "__main__":
    main()