from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rlvd.traffic_light import TrafficLightEstimator


WINDOW_NAME = "Traffic-Light HSV Test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test HSV traffic-light state estimation on a video segment."
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
        default=300,
        help="Number of frames to test.",
    )

    parser.add_argument(
        "--display-every",
        type=int,
        default=1,
        help="Display one frame every N processed frames.",
    )

    return parser.parse_args()


def draw_traffic_light_rois(
    frame: cv2.typing.MatLike,
    rectangles: list[list[int]],
    estimate_state: str,
) -> None:
    state_colors = {
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
        "green": (0, 255, 0),
        "unknown": (150, 150, 150),
    }

    color = state_colors.get(estimate_state, (150, 150, 150))

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
            f"TL ROI {index}",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_status(
    frame: cv2.typing.MatLike,
    frame_index: int,
    fps: float,
    raw_state: str,
    smoothed_state: str,
    red_pixels: int,
    yellow_pixels: int,
    green_pixels: int,
) -> None:
    state_colors = {
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
        "green": (0, 255, 0),
        "unknown": (150, 150, 150),
    }

    lines = [
        f"Frame: {frame_index}",
        f"Time: {frame_index / fps:.2f} seconds",
        f"Raw state: {raw_state.upper()}",
        f"Smoothed state: {smoothed_state.upper()}",
        f"HSV pixels | R: {red_pixels}  Y: {yellow_pixels}  G: {green_pixels}",
        "Controls: SPACE pause/resume | Q or ESC quit",
    ]

    overlay = frame.copy()
    panel_x1, panel_y1 = 15, 15
    panel_x2, panel_y2 = 680, 15 + 38 + len(lines) * 32

    cv2.rectangle(
        overlay,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (0, 0, 0),
        thickness=-1,
    )

    cv2.addWeighted(overlay, 0.66, frame, 0.34, 0, frame)

    for index, line in enumerate(lines):
        color = (
            state_colors.get(smoothed_state, (255, 255, 255))
            if index == 3
            else (255, 255, 255)
        )

        cv2.putText(
            frame,
            line,
            (panel_x1 + 15, panel_y1 + 35 + index * 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            color,
            2,
            cv2.LINE_AA,
        )


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

    traffic_light_config = config["traffic_light"]
    roi_rectangles = config["roi"]["traffic_light_rois"]["common"]

    estimator = TrafficLightEstimator(
        hsv_thresholds=traffic_light_config["hsv_thresholds"],
        smoothing_window=traffic_light_config["smoothing_window"],
        morphology_kernel_size=traffic_light_config["morphology_kernel_size"],
        min_active_pixels=traffic_light_config["min_active_pixels"],
    )

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

    print("Traffic-light test started.")
    print("SPACE = pause/resume | Q or ESC = quit")

    while processed < args.frames:
        if not paused:
            success, frame = capture.read()

            if not success:
                print("Reached the end of the video.")
                break

            current_frame_index = args.start_frame + processed

            estimate = estimator.update(
                frame_bgr=frame,
                roi_rectangles=roi_rectangles,
            )

            latest_frame = frame.copy()

            draw_traffic_light_rois(
                latest_frame,
                roi_rectangles,
                estimate.smoothed_state,
            )

            draw_status(
                latest_frame,
                current_frame_index,
                fps,
                estimate.raw_state,
                estimate.smoothed_state,
                estimate.red_pixels,
                estimate.yellow_pixels,
                estimate.green_pixels,
            )

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