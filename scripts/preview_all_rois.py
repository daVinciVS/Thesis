from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


WINDOW_NAME = "All ROI Preview"


def polygon_from_points(points: list[list[int]]) -> np.ndarray:
    return np.array(points, dtype=np.int32).reshape((-1, 1, 2))


def draw_transparent_polygon(
    frame: np.ndarray,
    polygon: np.ndarray,
    label: str,
    color: tuple[int, int, int],
    alpha: float = 0.20,
) -> None:
    overlay = frame.copy()

    cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    cv2.polylines(
        frame,
        [polygon],
        isClosed=True,
        color=color,
        thickness=3,
        lineType=cv2.LINE_AA,
    )

    x, y = polygon[0][0]

    cv2.putText(
        frame,
        label,
        (int(x), max(30, int(y) - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_traffic_light_rois(
    frame: np.ndarray,
    rectangles: list[list[int]],
) -> None:
    color = (0, 255, 255)

    for index, rect in enumerate(rectangles, start=1):
        x1, y1, x2, y2 = [int(value) for value in rect]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            thickness=3,
            lineType=cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"ROI_tl_near_{index}",
            (x1, max(30, y1 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_direction_arrow(
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
        line_type=cv2.LINE_AA,
        tipLength=0.18,
    )

    cv2.putText(
        frame,
        label,
        (start[0] + 10, start[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_legend(frame: np.ndarray) -> None:
    entries = [
        ("Near vehicle ROI", (255, 180, 0)),
        ("Far vehicle ROI", (0, 255, 0)),
        ("Near stop band", (0, 0, 255)),
        ("Far stop band", (255, 0, 255)),
        ("Central violation region", (0, 165, 255)),
        ("Near-side traffic-light ROIs", (0, 255, 255)),
    ]

    x1, y1 = 15, 15
    width = 370
    height = 42 + len(entries) * 32

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x1, y1),
        (x1 + width, y1 + height),
        (0, 0, 0),
        thickness=-1,
    )

    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(
        frame,
        "ROI Legend",
        (x1 + 12, y1 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for index, (label, color) in enumerate(entries):
        y = y1 + 60 + index * 32

        cv2.rectangle(
            frame,
            (x1 + 12, y - 15),
            (x1 + 34, y + 5),
            color,
            thickness=-1,
        )

        cv2.putText(
            frame,
            label,
            (x1 + 46, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.63,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview all calibrated vehicle, stop-line, and violation ROIs."
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video.",
    )

    parser.add_argument(
        "--config",
        default="configs/pipeline.yaml",
        help="Path to pipeline configuration YAML.",
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=35000,
        help="Video frame index to display.",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the annotated preview image to outputs/reports.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_path = Path(args.video)
    config_path = Path(args.config)

    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    success, frame = capture.read()
    capture.release()

    if not success:
        raise RuntimeError(
            f"Could not read frame {args.frame} from {video_path.name}."
        )

    roi = config["roi"]

    near_vehicle_polygon = polygon_from_points(
        roi["vehicle_regions"]["near_side"]["polygon"]
    )
    far_vehicle_polygon = polygon_from_points(
        roi["vehicle_regions"]["far_side"]["polygon"]
    )
    near_stop_polygon = polygon_from_points(
        roi["stop_lines"]["near_side"]["polygon"]
    )
    far_stop_polygon = polygon_from_points(
        roi["stop_lines"]["far_side"]["polygon"]
    )
    violation_polygon = polygon_from_points(
        roi["violation_region"]["polygon"]
    )

    draw_transparent_polygon(
        frame,
        near_vehicle_polygon,
        "ROI_vehicle_near",
        (255, 180, 0),
        alpha=0.12,
    )

    draw_transparent_polygon(
        frame,
        far_vehicle_polygon,
        "ROI_vehicle_far",
        (0, 255, 0),
        alpha=0.18,
    )

    draw_transparent_polygon(
        frame,
        near_stop_polygon,
        "R_stop_near",
        (0, 0, 255),
        alpha=0.42,
    )

    draw_transparent_polygon(
        frame,
        far_stop_polygon,
        "R_stop_far",
        (255, 0, 255),
        alpha=0.42,
    )

    draw_transparent_polygon(
        frame,
        violation_polygon,
        "R_violation",
        (0, 165, 255),
        alpha=0.15,
    )

    draw_traffic_light_rois(
        frame,
        roi["traffic_light_rois"]["common"],
    )

    height, width = frame.shape[:2]

    draw_legend(frame)

    cv2.putText(
        frame,
        f"Frame {args.frame} | {video_path.name}",
        (20, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if args.save:
        output_dir = Path("outputs/reports")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{video_path.stem}_frame_{args.frame}_all_rois.jpg"

        saved = cv2.imwrite(str(output_path), frame)

        if not saved:
            raise RuntimeError(f"Could not save preview to {output_path}")

        print(f"Saved preview: {output_path.resolve()}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1500, 850)
    cv2.imshow(WINDOW_NAME, frame)

    print("Press any key in the preview window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()