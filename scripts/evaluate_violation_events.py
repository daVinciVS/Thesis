from __future__ import annotations

import argparse
import csv
from pathlib import Path


TRUE_VALUES = {"true", "1", "yes", "y"}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def clean(value: object) -> str:
    return "" if value is None else str(value).strip()


def is_true(value: object) -> bool:
    return clean(value).lower() in TRUE_VALUES


def to_int(row: dict[str, str], field: str) -> int:
    value = clean(row.get(field))
    if not value:
        raise ValueError(f"Missing '{field}' in row: {row}")
    return int(float(value))


def event_frame(
    row: dict[str, str],
    source_type: str,
    ground_truth_frame_field: str,
) -> int:
    if source_type == "ground_truth":
        return to_int(row, ground_truth_frame_field)
    return to_int(row, "frame_index")


def match_events(
    ground_truth_violations: list[dict[str, str]],
    predictions: list[dict[str, str]],
    frame_tolerance: int,
    ground_truth_frame_field: str,
) -> tuple[list[dict[str, object]], list[dict[str, str]], list[dict[str, str]]]:
    used_prediction_indices: set[int] = set()
    matched_rows: list[dict[str, object]] = []
    unmatched_ground_truth: list[dict[str, str]] = []

    for ground_truth_row in ground_truth_violations:
        ground_truth_frame = event_frame(
            ground_truth_row,
            "ground_truth",
            ground_truth_frame_field,
        )
        ground_truth_direction = clean(ground_truth_row.get("direction"))
        candidates: list[tuple[int, int]] = []

        for prediction_index, prediction_row in enumerate(predictions):
            if prediction_index in used_prediction_indices:
                continue

            prediction_frame = event_frame(
                prediction_row,
                "prediction",
                ground_truth_frame_field,
            )
            prediction_direction = clean(prediction_row.get("direction"))
            frame_difference = abs(prediction_frame - ground_truth_frame)

            direction_matches = (
                not ground_truth_direction
                or not prediction_direction
                or ground_truth_direction == prediction_direction
            )

            if direction_matches and frame_difference <= frame_tolerance:
                candidates.append((frame_difference, prediction_index))

        if not candidates:
            unmatched_ground_truth.append(ground_truth_row)
            continue

        _, best_prediction_index = min(candidates, key=lambda item: item[0])
        used_prediction_indices.add(best_prediction_index)

        prediction_row = predictions[best_prediction_index]
        prediction_frame = event_frame(
            prediction_row,
            "prediction",
            ground_truth_frame_field,
        )

        matched_rows.append(
            {
                "result": "TP",
                "ground_truth_event_id": clean(ground_truth_row.get("event_id")),
                "ground_truth_vehicle_id": clean(ground_truth_row.get("manual_vehicle_id")),
                "ground_truth_frame": ground_truth_frame,
                "ground_truth_direction": ground_truth_direction,
                "prediction_frame": prediction_frame,
                "prediction_direction": clean(prediction_row.get("direction")),
                "frame_difference": abs(prediction_frame - ground_truth_frame),
                "prediction_track_id": clean(prediction_row.get("track_id")),
            }
        )

    unmatched_predictions = [
        prediction_row
        for prediction_index, prediction_row in enumerate(predictions)
        if prediction_index not in used_prediction_indices
    ]

    return matched_rows, unmatched_ground_truth, unmatched_predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate event-level red-light violation detections."
    )
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame-tolerance", type=int, default=12)
    parser.add_argument(
        "--ground-truth-frame-field",
        default="raw_frame_index",
        choices=["raw_frame_index", "clip_frame_index"],
    )
    args = parser.parse_args()

    all_ground_truth = load_csv(args.ground_truth)
    predictions = load_csv(args.predictions)

    ground_truth_violations = [
        row for row in all_ground_truth if is_true(row.get("is_violation"))
    ]

    matched, unmatched_ground_truth, unmatched_predictions = match_events(
        ground_truth_violations,
        predictions,
        args.frame_tolerance,
        args.ground_truth_frame_field,
    )

    true_positive = len(matched)
    false_positive = len(unmatched_predictions)
    false_negative = len(unmatched_ground_truth)

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else None
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else None
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, object]] = list(matched)

    for row in unmatched_ground_truth:
        output_rows.append(
            {
                "result": "FN",
                "ground_truth_event_id": clean(row.get("event_id")),
                "ground_truth_vehicle_id": clean(row.get("manual_vehicle_id")),
                "ground_truth_frame": event_frame(
                    row,
                    "ground_truth",
                    args.ground_truth_frame_field,
                ),
                "ground_truth_direction": clean(row.get("direction")),
                "prediction_frame": "",
                "prediction_direction": "",
                "frame_difference": "",
                "prediction_track_id": "",
            }
        )

    for row in unmatched_predictions:
        output_rows.append(
            {
                "result": "FP",
                "ground_truth_event_id": "",
                "ground_truth_vehicle_id": "",
                "ground_truth_frame": "",
                "ground_truth_direction": "",
                "prediction_frame": event_frame(
                    row,
                    "prediction",
                    args.ground_truth_frame_field,
                ),
                "prediction_direction": clean(row.get("direction")),
                "frame_difference": "",
                "prediction_track_id": clean(row.get("track_id")),
            }
        )

    fieldnames = [
        "result",
        "ground_truth_event_id",
        "ground_truth_vehicle_id",
        "ground_truth_frame",
        "ground_truth_direction",
        "prediction_frame",
        "prediction_direction",
        "frame_difference",
        "prediction_track_id",
    ]

    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    def metric(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.4f}"

    print(f"Annotation rows reviewed: {len(all_ground_truth)}")
    print(f"Ground-truth violation events: {len(ground_truth_violations)}")
    print(f"Predicted events: {len(predictions)}")
    print(f"TP={true_positive}")
    print(f"FP={false_positive}")
    print(f"FN={false_negative}")
    print(f"Precision={metric(precision)}")
    print(f"Recall={metric(recall)}")
    print(f"F1={metric(f1)}")
    print(f"Event matching tolerance: ±{args.frame_tolerance} frames")
    print(f"Ground-truth frame field: {args.ground_truth_frame_field}")
    print(f"Detailed event matches: {args.output}")


if __name__ == "__main__":
    main()
