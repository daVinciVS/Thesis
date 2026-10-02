from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


WINDOW_NAME = "YOLOv8 Car Detection Test"


def polygon_from_points(points: list[list[int]]) -> np.ndarray:
    return np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))


def point_inside_polygon(
    point: tuple[float, float],
    polygon: np.ndarray,
) -> bool:
    x, y = point

    return cv2.pointPolygonTest(
        polygon,
        (float(x), float(y)),
        False,
    ) >= 0


def bbox_bottom_center(
    bbox_xyxy: tuple[float, float, float, float],
) -> tuple[float, float]:
    x1, _, x2, y2 = bbox_xyxy
    return ((x1 + x2) / 2.0, y2)


def draw_transparent_polygon(
    frame: np.ndarray,
    polygon: np.ndarray,
    label: str,
    color: tuple[int, int, int],
    alpha: float = 0.15,
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
        (int(x), max(30, int(y) - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_detection(
    frame: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    confidence: float,
    region_name: str,
) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox_xyxy]

    color_map = {
        "near_side": (255, 180, 0),
        "far_side": (0, 255, 0),
        "outside": (120, 120, 120),
    }

    color = color_map[region_name]

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    label = f"car | {confidence:.2f} | {region_name}"

    cv2.putText(
        frame,
        label,
        (x1, max(25, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_status(
    frame: np.ndarray,
    frame_index: int,
    fps: float,
    all_car_count: int,
    near_car_count: int,
    far_car_count: int,
) -> None:
    lines = [
        f"Frame: {frame_index}",
        f"Time: {frame_index / fps:.2f} seconds",
        f"YOLO car detections: {all_car_count}",
        f"Near ROI cars: {near_car_count}",
        f"Far ROI cars: {far_car_count}",
        "Controls: SPACE pause/resume | Q or ESC quit",
    ]

    overlay = frame.copy()

    panel_x1, panel_y1 = 15, 15
    panel_x2, panel_y2 = 590, 15 + 35 + len(lines) * 31

    cv2.rectangle(
        overlay,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (0, 0, 0),
        thickness=-1,
    )

    cv2.addWeighted(overlay, 0.66, frame, 0.34, 0, frame)

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (panel_x1 + 15, panel_y1 + 33 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test YOLOv8 car detection with calibrated vehicle ROIs."
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video.",
    )

    parser.add_argument(
        "--config",
        default="configs/pipeline.yaml",
        help="Path to pipeline configuration YAML.",
    )

    parser.add_argument(
        "--start-frame",
        type=int,
        default=35000,
        help="First frame to process.",
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=120,
        help="Number of frames to process.",
    )

    parser.add_argument(
        "--display-every",
        type=int,
        default=1,
        help="Display one frame every N processed frames.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_path = Path(args.video)
    config_path = Path(args.config)

    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Config does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    model_config = config["model"]
    roi_config = config["roi"]["vehicle_regions"]

    near_polygon = polygon_from_points(
        roi_config["near_side"]["polygon"]
    )

    far_polygon = polygon_from_points(
        roi_config["far_side"]["polygon"]
    )

    weights_path = PROJECT_ROOT / model_config["weights"]

    print(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1500, 850)

    processed = 0
    paused = False
    latest_frame = None

    print("YOLO detection test started.")
    print("SPACE = pause/resume | Q or ESC = quit")

    while processed < args.frames:
        if not paused:
            success, frame = capture.read()

            if not success:
                print("Reached the end of the video.")
                break

            current_frame_index = args.start_frame + processed

            result = model.predict(
                source=frame,
                classes=[int(model_config["car_class_id"])],
                conf=float(model_config["confidence"]),
                iou=float(model_config["iou"]),
                imgsz=int(model_config["imgsz"]),
                device=model_config["device"],
                verbose=False,
            )[0]

            annotated = frame.copy()

            draw_transparent_polygon(
                annotated,
                near_polygon,
                "ROI_vehicle_near",
                (255, 180, 0),
            )

            draw_transparent_polygon(
                annotated,
                far_polygon,
                "ROI_vehicle_far",
                (0, 255, 0),
            )

            all_car_count = 0
            near_car_count = 0
            far_car_count = 0

            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()

                for bbox_array, confidence in zip(boxes, confidences):
                    bbox_xyxy = tuple(
                        float(value)
                        for value in bbox_array.tolist()
                    )

                    all_car_count += 1

                    bottom_center = bbox_bottom_center(bbox_xyxy)

                    if point_inside_polygon(bottom_center, near_polygon):
                        region_name = "near_side"
                        near_car_count += 1

                    elif point_inside_polygon(bottom_center, far_polygon):
                        region_name = "far_side"
                        far_car_count += 1

                    else:
                        region_name = "outside"

                    draw_detection(
                        annotated,
                        bbox_xyxy,
                        float(confidence),
                        region_name,
                    )

            draw_status(
                annotated,
                current_frame_index,
                fps,
                all_car_count,
                near_car_count,
                far_car_count,
            )

            latest_frame = annotated
            processed += 1

        if latest_frame is not None and (
            processed % args.display_every == 0 or paused
        ):
            cv2.imshow(WINDOW_NAME, latest_frame)

        key = cv2.waitKey(1 if not paused else 30) & 0xFF

        if key == ord(" "):
            paused = not paused

        elif key == ord("q") or key == 27:
            break

    capture.release()
    cv2.destroyAllWindows()

    print(f"Frames processed: {processed}")


if __name__ == "__main__":
    main()