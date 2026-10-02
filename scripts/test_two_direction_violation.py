from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rlvd.geometry import polygon_from_points
from rlvd.traffic_light import TrafficLightEstimator
from rlvd.violation import TwoDirectionViolationManager


WINDOW_NAME = "Two-Direction Red-Light Violation Test"


COLORS = {
    "near_side": (255, 180, 0),
    "far_side": (0, 255, 0),
    "near_stop": (0, 0, 255),
    "far_stop": (255, 0, 255),
    "violation_region": (0, 165, 255),
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "green": (0, 255, 0),
    "unknown": (150, 150, 150),
    "anchor_near": (255, 255, 0),
    "anchor_far": (0, 255, 0),
}


def draw_transparent_polygon(
    frame: np.ndarray,
    polygon: np.ndarray,
    label: str,
    color: tuple[int, int, int],
    alpha: float,
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
        (int(x), max(28, int(y) - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_traffic_light_rois(
    frame: np.ndarray,
    rectangles: list[list[int]],
    signal_state: str,
) -> None:
    color = COLORS.get(signal_state, COLORS["unknown"])

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
            f"COMMON TL {index}: {signal_state.upper()}",
            (x1, max(28, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.57,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_track(
    frame: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    track_id: int,
    confidence: float,
    direction: str,
    anchor: tuple[float, float],
    anchor_name: str,
    inside_region: bool,
    armed: bool,
    reported: bool,
    event_detected: bool,
) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox_xyxy]

    if event_detected:
        box_color = COLORS["red"]
    else:
        box_color = COLORS[direction]

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        box_color,
        thickness=3 if event_detected else 2,
        lineType=cv2.LINE_AA,
    )

    region_text = "IN Rv" if inside_region else "OUT Rv"

    if reported:
        status_text = "REPORTED"
    elif armed:
        status_text = "ARMED"
    else:
        status_text = "NOT ARMED"

    label = (
        f"ID {track_id} | {direction} | {region_text} | "
        f"{status_text} | car {confidence:.2f}"
    )

    cv2.putText(
        frame,
        label,
        (x1, max(25, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        box_color,
        2,
        cv2.LINE_AA,
    )

    anchor_color = (
        COLORS["anchor_near"]
        if direction == "near_side"
        else COLORS["anchor_far"]
    )

    anchor_x, anchor_y = [int(round(value)) for value in anchor]

    cv2.circle(
        frame,
        (anchor_x, anchor_y),
        radius=7,
        color=anchor_color,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        anchor_name,
        (anchor_x + 8, anchor_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        anchor_color,
        2,
        cv2.LINE_AA,
    )

    if event_detected:
        cv2.putText(
            frame,
            "RED-LIGHT VIOLATION",
            (x1, min(frame.shape[0] - 18, y2 + 26)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            COLORS["red"],
            2,
            cv2.LINE_AA,
        )


def draw_status_panel(
    frame: np.ndarray,
    frame_index: int,
    fps: float,
    raw_state: str,
    smoothed_state: str,
    red_pixels: int,
    yellow_pixels: int,
    green_pixels: int,
    near_tracks: int,
    far_tracks: int,
    violations: int,
    passage_candidates: int,
) -> None:
    lines = [
        f"Frame: {frame_index}",
        f"Time: {frame_index / fps:.2f} sec",
        f"Common TL raw: {raw_state.upper()}",
        f"Common TL smooth: {smoothed_state.upper()}",
        f"HSV pixels R:{red_pixels}  Y:{yellow_pixels}  G:{green_pixels}",
        f"Near-side tracks: {near_tracks}",
        f"Far-side tracks: {far_tracks}",
        f"Passage candidates: {passage_candidates}",
        f"Recorded violations: {violations}",
        "Controls: SPACE pause/resume | Q or ESC quit",
    ]

    overlay = frame.copy()

    panel_x1, panel_y1 = 15, 15
    panel_x2, panel_y2 = 730, 15 + 38 + len(lines) * 31

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
            COLORS.get(smoothed_state, (255, 255, 255))
            if index == 3
            else (255, 255, 255)
        )

        cv2.putText(
            frame,
            line,
            (panel_x1 + 15, panel_y1 + 35 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.63,
            color,
            2,
            cv2.LINE_AA,
        )


def write_events_csv(
    output_path: Path,
    events: list,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "video_name",
        "direction",
        "frame_index",
        "timestamp_seconds",
        "red_phase_id",
        "track_id",
        "anchor_name",
        "anchor_x",
        "anchor_y",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "traffic_light_state",
        "evidence_frame_path",
        "evidence_crop_path",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for event in events:
            writer.writerow(asdict(event))


def write_passage_review_csv(
    output_path: Path,
    passage_candidates: list[dict[str, object]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "candidate_id",
        "video_name",
        "direction",
        "raw_frame_index",
        "clip_frame_index",
        "timestamp_seconds",
        "pipeline_track_id",
        "confidence",
        "traffic_light_state",
        "was_armed",
        "was_reported",
        "anchor_name",
        "anchor_x",
        "anchor_y",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "manual_vehicle_id",
        "object_is_car",
        "entered_rv_after_red",
        "is_violation",
        "review_status",
        "notes",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(passage_candidates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Two-direction YOLOv8 + ByteTrack + common HSV "
            "red-light violation test."
        )
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video.",
    )

    parser.add_argument(
        "--config",
        default="configs/pipeline.yaml",
        help="Path to pipeline YAML configuration.",
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
        help="Number of frames to process.",
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
    roi_config = config["roi"]
    traffic_light_config = config["traffic_light"]
    violation_config = config["violation"]

    vehicle_polygons = {
        "near_side": polygon_from_points(
            roi_config["vehicle_regions"]["near_side"]["polygon"]
        ),
        "far_side": polygon_from_points(
            roi_config["vehicle_regions"]["far_side"]["polygon"]
        ),
    }

    stop_polygons = {
        "near_side": polygon_from_points(
            roi_config["stop_lines"]["near_side"]["polygon"]
        ),
        "far_side": polygon_from_points(
            roi_config["stop_lines"]["far_side"]["polygon"]
        ),
    }

    violation_polygon = polygon_from_points(
        roi_config["violation_region"]["polygon"]
    )

    common_traffic_light_rois = roi_config["traffic_light_rois"]["common"]

    weights_path = PROJECT_ROOT / model_config["weights"]
    tracker_path = PROJECT_ROOT / model_config["tracker_config"]

    if not tracker_path.exists():
        raise FileNotFoundError(
            f"ByteTrack configuration not found: {tracker_path}"
        )

    print(f"Loading model: {weights_path}")
    print(f"Using tracker: {tracker_path}")

    model = YOLO(str(weights_path))

    traffic_light_estimator = TrafficLightEstimator(
        hsv_thresholds=traffic_light_config["hsv_thresholds"],
        smoothing_window=int(traffic_light_config["smoothing_window"]),
        morphology_kernel_size=int(
            traffic_light_config["morphology_kernel_size"]
        ),
        min_active_pixels=int(traffic_light_config["min_active_pixels"]),
    )

    evidence_dir = (
        PROJECT_ROOT
        / config["project"]["output_root"]
        / "evidence"
        / f"{video_path.stem}_two_direction"
    )

    violation_manager = TwoDirectionViolationManager(
        vehicle_polygons=vehicle_polygons,
        violation_polygon=violation_polygon,
        front_anchors=violation_config["front_anchors"],
        entry_tolerance_frames=int(
            violation_config["entry_tolerance_frames"]
        ),
        red_phase_end_hold_frames=int(
            violation_config["red_phase_end_hold_frames"]
        ),
        reset_track_state_after_missing_frames=int(
            violation_config["reset_track_state_after_missing_frames"]
        ),
        save_evidence=bool(violation_config["save_evidence"]),
        evidence_dir=evidence_dir,
        evidence_padding=int(violation_config["evidence_padding"]),
    )

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))

    if fps <= 0:
        raise RuntimeError("Could not determine video FPS.")

    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1500, 850)

    processed = 0
    paused = False
    latest_frame: np.ndarray | None = None

    passage_candidates: list[dict[str, object]] = []

    previous_inside_by_track: dict[tuple[str, int], bool] = {}
    was_reported_by_track: dict[tuple[str, int], bool] = {}

    print("Two-direction violation test started.")
    print("SPACE = pause/resume | Q or ESC = quit")

    try:
        while processed < args.frames:
            if not paused:
                success, frame = capture.read()

                if not success:
                    print("Reached end of video.")
                    break

                frame_index = args.start_frame + processed

                traffic_light_estimate = traffic_light_estimator.update(
                    frame_bgr=frame,
                    roi_rectangles=common_traffic_light_rois,
                )

                signal_state = traffic_light_estimate.smoothed_state

                violation_manager.begin_frame(
                    frame_index=frame_index,
                    traffic_light_state=signal_state,
                )

                result = model.track(
                    source=frame,
                    persist=True,
                    tracker=str(tracker_path),
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
                    vehicle_polygons["near_side"],
                    "ROI_vehicle_near",
                    COLORS["near_side"],
                    alpha=0.10,
                )

                draw_transparent_polygon(
                    annotated,
                    vehicle_polygons["far_side"],
                    "ROI_vehicle_far",
                    COLORS["far_side"],
                    alpha=0.15,
                )

                draw_transparent_polygon(
                    annotated,
                    stop_polygons["near_side"],
                    "R_stop_near",
                    COLORS["near_stop"],
                    alpha=0.42,
                )

                draw_transparent_polygon(
                    annotated,
                    stop_polygons["far_side"],
                    "R_stop_far",
                    COLORS["far_stop"],
                    alpha=0.42,
                )

                draw_transparent_polygon(
                    annotated,
                    violation_polygon,
                    "R_violation",
                    COLORS["violation_region"],
                    alpha=0.12,
                )

                draw_traffic_light_rois(
                    annotated,
                    common_traffic_light_rois,
                    signal_state,
                )

                near_tracks = 0
                far_tracks = 0

                if (
                    result.boxes is not None
                    and result.boxes.id is not None
                ):
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    track_ids = result.boxes.id.int().cpu().tolist()

                    for bbox_array, confidence, track_id in zip(
                        boxes,
                        confidences,
                        track_ids,
                    ):
                        track_id = int(track_id)

                        bbox_xyxy = tuple(
                            float(value)
                            for value in bbox_array.tolist()
                        )

                        direction, event, anchor = (
                            violation_manager.update_track(
                                frame_bgr=frame,
                                video_name=video_path.name,
                                video_stem=video_path.stem,
                                frame_index=frame_index,
                                fps=fps,
                                track_id=track_id,
                                bbox_xyxy=bbox_xyxy,
                                traffic_light_state=signal_state,
                            )
                        )

                        if direction is None or anchor is None:
                            continue

                        if direction == "near_side":
                            near_tracks += 1
                        elif direction == "far_side":
                            far_tracks += 1

                        (
                            diagnostic_direction,
                            diagnostic_anchor,
                            inside_region,
                            armed,
                            reported,
                        ) = violation_manager.get_track_diagnostics(
                            track_id,
                            bbox_xyxy,
                        )

                        track_key = (direction, track_id)

                        previous_inside = previous_inside_by_track.get(
                            track_key,
                            False,
                        )

                        previously_reported = was_reported_by_track.get(
                            track_key,
                            False,
                        )

                        crossed_into_region = (
                            not previous_inside
                            and inside_region
                        )

                        newly_reported = (
                            reported
                            and not previously_reported
                        )

                        if crossed_into_region:
                            x1, y1, x2, y2 = bbox_xyxy

                            passage_candidates.append(
                                {
                                    "candidate_id": (
                                        len(passage_candidates) + 1
                                    ),
                                    "video_name": video_path.name,
                                    "direction": direction,
                                    "raw_frame_index": frame_index,
                                    "clip_frame_index": (
                                        frame_index - args.start_frame
                                    ),
                                    "timestamp_seconds": round(
                                        frame_index / fps,
                                        3,
                                    ),
                                    "pipeline_track_id": track_id,
                                    "confidence": round(
                                        float(confidence),
                                        4,
                                    ),
                                    "traffic_light_state": signal_state,
                                    "was_armed": armed,
                                    "was_reported": reported,
                                    "anchor_name": (
                                        violation_config[
                                            "front_anchors"
                                        ][direction]
                                    ),
                                    "anchor_x": round(
                                        float(diagnostic_anchor[0]),
                                        2,
                                    ),
                                    "anchor_y": round(
                                        float(diagnostic_anchor[1]),
                                        2,
                                    ),
                                    "bbox_x1": round(float(x1), 2),
                                    "bbox_y1": round(float(y1), 2),
                                    "bbox_x2": round(float(x2), 2),
                                    "bbox_y2": round(float(y2), 2),
                                    "manual_vehicle_id": "",
                                    "object_is_car": "",
                                    "entered_rv_after_red": "",
                                    "is_violation": "",
                                    "review_status": "pending",
                                    "notes": "",
                                }
                            )

                        if newly_reported and not crossed_into_region:
                            x1, y1, x2, y2 = bbox_xyxy

                            passage_candidates.append(
                                {
                                    "candidate_id": (
                                        len(passage_candidates) + 1
                                    ),
                                    "video_name": video_path.name,
                                    "direction": direction,
                                    "raw_frame_index": frame_index,
                                    "clip_frame_index": (
                                        frame_index - args.start_frame
                                    ),
                                    "timestamp_seconds": round(
                                        frame_index / fps,
                                        3,
                                    ),
                                    "pipeline_track_id": track_id,
                                    "confidence": round(
                                        float(confidence),
                                        4,
                                    ),
                                    "traffic_light_state": signal_state,
                                    "was_armed": armed,
                                    "was_reported": reported,
                                    "anchor_name": (
                                        violation_config[
                                            "front_anchors"
                                        ][direction]
                                    ),
                                    "anchor_x": round(
                                        float(diagnostic_anchor[0]),
                                        2,
                                    ),
                                    "anchor_y": round(
                                        float(diagnostic_anchor[1]),
                                        2,
                                    ),
                                    "bbox_x1": round(float(x1), 2),
                                    "bbox_y1": round(float(y1), 2),
                                    "bbox_x2": round(float(x2), 2),
                                    "bbox_y2": round(float(y2), 2),
                                    "manual_vehicle_id": "",
                                    "object_is_car": "",
                                    "entered_rv_after_red": "",
                                    "is_violation": "",
                                    "review_status": "pending",
                                    "notes": (
                                        "Pipeline reported event after "
                                        "entry was already observed"
                                    ),
                                }
                            )

                        previous_inside_by_track[track_key] = inside_region
                        was_reported_by_track[track_key] = reported

                        draw_track(
                            annotated,
                            bbox_xyxy,
                            track_id,
                            float(confidence),
                            direction,
                            anchor,
                            violation_config["front_anchors"][direction],
                            inside_region,
                            armed,
                            reported,
                            event_detected=(
                                event is not None
                                or reported
                            ),
                        )

                draw_status_panel(
                    annotated,
                    frame_index,
                    fps,
                    traffic_light_estimate.raw_state,
                    signal_state,
                    traffic_light_estimate.red_pixels,
                    traffic_light_estimate.yellow_pixels,
                    traffic_light_estimate.green_pixels,
                    near_tracks,
                    far_tracks,
                    len(violation_manager.events),
                    len(passage_candidates),
                )

                latest_frame = annotated
                processed += 1

            if latest_frame is not None:
                cv2.imshow(WINDOW_NAME, latest_frame)

            key = cv2.waitKey(1 if not paused else 30) & 0xFF

            if key == ord(" "):
                paused = not paused

            elif key == ord("q") or key == 27:
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()

    reports_dir = (
        PROJECT_ROOT
        / config["project"]["output_root"]
        / "reports"
    )

    report_path = (
        reports_dir
        / f"{video_path.stem}_two_direction_violations.csv"
    )

    passage_review_path = (
        reports_dir
        / f"{video_path.stem}_two_direction_passages_review.csv"
    )

    write_events_csv(report_path, violation_manager.events)

    write_passage_review_csv(
        passage_review_path,
        passage_candidates,
    )

    print(f"Frames processed: {processed}")
    print(
        f"Passage candidates recorded: "
        f"{len(passage_candidates)}"
    )
    print(f"Violations recorded: {len(violation_manager.events)}")
    print(f"CSV report: {report_path.resolve()}")
    print(
        "Passage review CSV: "
        f"{passage_review_path.resolve()}"
    )

    if violation_manager.events:
        print("\nRecorded events:")

        for event in violation_manager.events:
            print(
                f"  {event.direction} | "
                f"frame={event.frame_index} | "
                f"time={event.timestamp_seconds:.2f}s | "
                f"track={event.track_id} | "
                f"anchor={event.anchor_name} | "
                f"red_phase={event.red_phase_id}"
            )


if __name__ == "__main__":
    main()