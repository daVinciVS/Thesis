from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME = "ROI Calibration"


class ROICalibrator:
    def __init__(self, frame: np.ndarray) -> None:
        self.original_frame = frame
        self.points: list[tuple[int, int]] = []

    def mouse_callback(
        self,
        event: int,
        x: int,
        y: int,
        flags: int,
        param: object,
    ) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            print(f"Added point: [{x}, {y}]")

        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            removed = self.points.pop()
            print(f"Removed point: [{removed[0]}, {removed[1]}]")

    def draw(self) -> np.ndarray:
        canvas = self.original_frame.copy()

        for index, point in enumerate(self.points):
            cv2.circle(canvas, point, 7, (0, 255, 255), -1)

            cv2.putText(
                canvas,
                str(index + 1),
                (point[0] + 10, point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if len(self.points) >= 2:
            points_array = np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))

            cv2.polylines(
                canvas,
                [points_array],
                isClosed=False,
                color=(0, 255, 255),
                thickness=2,
            )

        help_lines = [
            "Left click: add point",
            "Right click: remove latest point",
            "P: print and save polygon",
            "R: print and save rectangle from first 2 points",
            "C: clear all points",
            "Q / ESC: quit",
        ]

        overlay = canvas.copy()
        panel_height = 35 + len(help_lines) * 27

        cv2.rectangle(
            overlay,
            (10, 10),
            (500, panel_height),
            (0, 0, 0),
            thickness=-1,
        )

        cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

        for index, line in enumerate(help_lines):
            cv2.putText(
                canvas,
                line,
                (20, 38 + index * 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.63,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return canvas

    def save_polygon(self, output_path: Path) -> None:
        if len(self.points) < 3:
            print("Polygon requires at least 3 points.")
            return

        payload = {
            "polygon": [[x, y] for x, y in self.points]
        }

        output_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

        print("\nSaved polygon:")
        print(json.dumps(payload, indent=2))
        print(f"Output: {output_path.resolve()}\n")

    def save_rectangle(self, output_path: Path) -> None:
        if len(self.points) < 2:
            print("Rectangle requires at least 2 points.")
            return

        x1, y1 = self.points[0]
        x2, y2 = self.points[1]

        rectangle = [
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2),
        ]

        payload = {
            "rectangle": rectangle
        }

        output_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

        print("\nSaved rectangle:")
        print(json.dumps(payload, indent=2))
        print(f"Output: {output_path.resolve()}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive tool for calibrating ROI polygons or rectangles."
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to an MP4 video.",
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=100,
        help="Zero-based frame index to display. Default: 100.",
    )

    parser.add_argument(
        "--output",
        default="roi_coordinates.json",
        help="Output JSON file path. Default: roi_coordinates.json",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_path = Path(args.video)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

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

    calibrator = ROICalibrator(frame)
    output_path = Path(args.output)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1400, 800)

    cv2.setMouseCallback(WINDOW_NAME, calibrator.mouse_callback)

    print("\nROI calibration started.")
    print("Use the OpenCV window to click coordinate points.\n")

    while True:
        canvas = calibrator.draw()
        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(20) & 0xFF

        if key == ord("p"):
            calibrator.save_polygon(output_path)

        elif key == ord("r"):
            calibrator.save_rectangle(output_path)

        elif key == ord("c"):
            calibrator.points.clear()
            print("Points cleared.")

        elif key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()