from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


def polygon_from_points(points: list[list[int]]) -> np.ndarray:
    return np.array(points, dtype=np.int32).reshape((-1, 1, 2))


def draw_polygon(
    frame: np.ndarray,
    polygon: np.ndarray,
    label: str,
    color: tuple[int, int, int],
) -> None:
    overlay = frame.copy()

    cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)

    cv2.polylines(
        frame,
        [polygon],
        isClosed=True,
        color=color,
        thickness=3,
    )

    x, y = polygon[0][0]

    cv2.putText(
        frame,
        label,
        (int(x), max(30, int(y) - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_arrow(
    frame: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    label: str,
    color: tuple[int, int, int],
) -> None:
    cv2.arrowedLine(
        frame,
        start,
        end,
        color,
        thickness=4,
        tipLength=0.15,
    )

    cv2.putText(
        frame,
        label,
        (start[0] + 10, start[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview calibrated stop-line ROIs on a video frame."
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to video.",
    )

    parser.add_argument(
        "--config",
        default="configs/pipeline.yaml",
        help="Path to pipeline YAML configuration.",
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=35000,
        help="Frame index to preview.",
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    config_path = Path(args.config)

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    success, frame = capture.read()
    capture.release()

    if not success:
        raise RuntimeError(f"Could not read frame {args.frame}.")

    stop_lines = config["roi"]["stop_lines"]

    near_polygon = polygon_from_points(stop_lines["near_side"]["polygon"])
    far_polygon = polygon_from_points(stop_lines["far_side"]["polygon"])

    draw_polygon(
        frame,
        near_polygon,
        "Near-side R_stop (movement: UP)",
        (0, 0, 255),
    )

    draw_polygon(
        frame,
        far_polygon,
        "Far-side R_stop (movement: DOWN)",
        (255, 0, 255),
    )

    frame_height, frame_width = frame.shape[:2]

    draw_arrow(
        frame,
        (frame_width - 170, frame_height - 120),
        (frame_width - 170, frame_height - 280),
        "Near-side cars",
        (0, 0, 255),
    )

    draw_arrow(
        frame,
        (frame_width - 330, 180),
        (frame_width - 330, 340),
        "Far-side cars",
        (255, 0, 255),
    )

    cv2.putText(
        frame,
        f"Frame: {args.frame}",
        (20, frame_height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    window_name = "Stop-line ROI Preview"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1500, 850)
    cv2.imshow(window_name, frame)

    print("Press any key in the preview window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()