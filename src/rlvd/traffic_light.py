from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TrafficLightEstimate:
    raw_state: str
    smoothed_state: str
    red_pixels: int
    yellow_pixels: int
    green_pixels: int


class TrafficLightEstimator:
    """
    Estimates near-side traffic-light state from one or more fixed ROIs.

    Each ROI is processed independently in HSV. Pixel activations are summed
    across all signal heads, then a single raw state is selected. A temporal
    mode filter smooths the last W estimates.
    """

    def __init__(
        self,
        hsv_thresholds: dict,
        smoothing_window: int = 5,
        morphology_kernel_size: int = 3,
        min_active_pixels: int = 15,
    ) -> None:
        self.thresholds = hsv_thresholds
        self.history: deque[str] = deque(maxlen=smoothing_window)
        self.min_active_pixels = min_active_pixels

        kernel_size = max(1, int(morphology_kernel_size))
        self.kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    @staticmethod
    def _as_hsv_array(values: list[int]) -> np.ndarray:
        return np.asarray(values, dtype=np.uint8)

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Remove isolated noise and fill small gaps in illuminated lenses.
        """
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return mask

    def _count_color_pixels(self, hsv: np.ndarray) -> dict[str, int]:
        """
        Count red, yellow, and green mask pixels in one HSV traffic-light ROI.
        Red has two hue intervals because hue wraps around OpenCV's range.
        """
        red_mask_low = cv2.inRange(
            hsv,
            self._as_hsv_array(self.thresholds["red_lower_1"]),
            self._as_hsv_array(self.thresholds["red_upper_1"]),
        )

        red_mask_high = cv2.inRange(
            hsv,
            self._as_hsv_array(self.thresholds["red_lower_2"]),
            self._as_hsv_array(self.thresholds["red_upper_2"]),
        )

        red_mask = self._clean_mask(
            cv2.bitwise_or(red_mask_low, red_mask_high)
        )

        yellow_mask = self._clean_mask(
            cv2.inRange(
                hsv,
                self._as_hsv_array(self.thresholds["yellow_lower"]),
                self._as_hsv_array(self.thresholds["yellow_upper"]),
            )
        )

        green_mask = self._clean_mask(
            cv2.inRange(
                hsv,
                self._as_hsv_array(self.thresholds["green_lower"]),
                self._as_hsv_array(self.thresholds["green_upper"]),
            )
        )

        return {
            "red": int(cv2.countNonZero(red_mask)),
            "yellow": int(cv2.countNonZero(yellow_mask)),
            "green": int(cv2.countNonZero(green_mask)),
        }

    def _crop_roi(
        self,
        frame_bgr: np.ndarray,
        rect: list[int],
    ) -> np.ndarray:
        """
        Crop a rectangle safely, even if calibration coordinates touch an edge.
        Rectangle format: [x1, y1, x2, y2].
        """
        frame_height, frame_width = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(value) for value in rect]

        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(0, min(x2, frame_width))
        y2 = max(0, min(y2, frame_height))

        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=np.uint8)

        return frame_bgr[y1:y2, x1:x2]

    def _estimate_raw_state(
        self,
        frame_bgr: np.ndarray,
        roi_rectangles: list[list[int]],
    ) -> tuple[str, dict[str, int]]:
        """
        Aggregate HSV color-mask pixel counts from every configured signal head.
        """
        total_scores = {
            "red": 0,
            "yellow": 0,
            "green": 0,
        }

        for rect in roi_rectangles:
            crop_bgr = self._crop_roi(frame_bgr, rect)

            if crop_bgr.size == 0:
                continue

            hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
            roi_scores = self._count_color_pixels(hsv)

            for state, count in roi_scores.items():
                total_scores[state] += count

        best_state = max(total_scores, key=total_scores.get)

        if total_scores[best_state] < self.min_active_pixels:
            return "unknown", total_scores

        return best_state, total_scores

    def _mode_state(self) -> str:
        """
        Select the majority state over the latest W valid raw estimates.

        In a tie, choose the most recent tied state, which makes the behavior
        deterministic and responsive during traffic-light transitions.
        """
        valid_history = [
            state
            for state in self.history
            if state != "unknown"
        ]

        if not valid_history:
            return "unknown"

        counts = Counter(valid_history)
        maximum_count = max(counts.values())

        for state in reversed(valid_history):
            if counts[state] == maximum_count:
                return state

        return "unknown"

    def update(
        self,
        frame_bgr: np.ndarray,
        roi_rectangles: list[list[int]],
    ) -> TrafficLightEstimate:
        """
        Process one complete video frame and return the raw and smoothed
        traffic-light state.
        """
        raw_state, scores = self._estimate_raw_state(
            frame_bgr,
            roi_rectangles,
        )

        self.history.append(raw_state)
        smoothed_state = self._mode_state()

        return TrafficLightEstimate(
            raw_state=raw_state,
            smoothed_state=smoothed_state,
            red_pixels=scores["red"],
            yellow_pixels=scores["yellow"],
            green_pixels=scores["green"],
        )