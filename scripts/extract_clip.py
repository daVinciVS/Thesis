from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a frame-accurate video clip using OpenCV."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input MP4 video path.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output MP4 clip path.",
    )

    parser.add_argument(
        "--start-frame",
        required=True,
        type=int,
        help="Zero-based first frame to include.",
    )

    parser.add_argument(
        "--frames",
        required=True,
        type=int,
        help="Number of frames to extract.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if args.start_frame < 0:
        raise ValueError("--start-frame must be zero or greater.")

    if args.frames <= 0:
        raise ValueError("--frames must be greater than zero.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))

    if fps <= 0:
        raise RuntimeError("Could not determine FPS.")

    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    codec = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        codec,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    written = 0

    try:
        while written < args.frames:
            success, frame = capture.read()

            if not success:
                break

            writer.write(frame)
            written += 1

    finally:
        capture.release()
        writer.release()

    print(f"Input: {input_path.resolve()}")
    print(f"Output: {output_path.resolve()}")
    print(f"Start frame: {args.start_frame}")
    print(f"Requested frames: {args.frames}")
    print(f"Written frames: {written}")
    print(f"FPS: {fps:.4f}")
    print(f"Resolution: {width}x{height}")


if __name__ == "__main__":
    main()