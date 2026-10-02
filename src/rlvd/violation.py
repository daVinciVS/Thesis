from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from rlvd.geometry import point_inside_polygon


@dataclass
class ViolationEvent:
    video_name: str
    direction: str
    frame_index: int
    timestamp_seconds: float
    red_phase_id: int
    track_id: int
    anchor_name: str
    anchor_x: float
    anchor_y: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    traffic_light_state: str
    evidence_frame_path: str | None
    evidence_crop_path: str | None


@dataclass
class DirectionTrackState:
    """
    State for one tracked car assigned to one traffic direction.
    """

    last_seen_frame: int = 0
    previously_inside_violation_region: bool = False
    armed_red_phase_id: int | None = None

    # Last confirmed frame where the anchor was outside R_violation
    # during the current red phase.
    last_outside_red_frame: int | None = None


class TwoDirectionViolationManager:
    def __init__(
        self,
        vehicle_polygons: dict[str, np.ndarray],
        violation_polygon: np.ndarray,
        front_anchors: dict[str, str],
        entry_tolerance_frames: int,
        red_phase_end_hold_frames: int,
        reset_track_state_after_missing_frames: int,
        save_evidence: bool,
        evidence_dir: Path,
        evidence_padding: int,
    ) -> None:
        self.vehicle_polygons = vehicle_polygons
        self.violation_polygon = violation_polygon
        self.front_anchors = front_anchors
        self.entry_tolerance_frames = entry_tolerance_frames
        self.red_phase_end_hold_frames = red_phase_end_hold_frames
        self.non_red_frames_in_a_row = 0

        self.reset_track_state_after_missing_frames = (
            reset_track_state_after_missing_frames
        )

        self.save_evidence = save_evidence
        self.evidence_dir = evidence_dir
        self.evidence_padding = evidence_padding

        self.current_red_phase_id = 0
        self.was_red_previous_frame = False

        self.track_states: dict[tuple[str, int], DirectionTrackState] = {}

        # One reported event maximum for each car in each red phase.
        self.reported_events: set[tuple[str, int, int]] = set()

        self.events: list[ViolationEvent] = []

    def update_red_phase(self, traffic_light_state: str) -> bool:
        """
        Update a stable red-phase state with a non-red hold.

        A red phase begins immediately when the smoothed light state becomes
        red. It does not end until the light has remained non-red for
        red_phase_end_hold_frames consecutive frames. This prevents brief
        HSV/mode-filter flicker from resetting armed tracks.
        """
        observed_red = traffic_light_state == "red"

        if observed_red:
            self.non_red_frames_in_a_row = 0

            if not self.was_red_previous_frame:
                self.current_red_phase_id += 1
                self.was_red_previous_frame = True
                return True

            return False

        if not self.was_red_previous_frame:
            return False

        self.non_red_frames_in_a_row += 1

        if self.non_red_frames_in_a_row >= self.red_phase_end_hold_frames:
            self.was_red_previous_frame = False
            self.non_red_frames_in_a_row = 0

        return False

    def begin_frame(
        self,
        frame_index: int,
        traffic_light_state: str,
    ) -> None:
        """
        Must be called exactly once per video frame, before update_track().

        It updates red-phase state once for the whole frame and removes
        stale tracks. Per-track arming happens inside update_track() after
        the current anchor location is known.
        """
        self.update_red_phase(traffic_light_state)
        self.cleanup_missing_tracks(frame_index)

    def cleanup_missing_tracks(self, frame_index: int) -> None:
        expired_keys = [
            state_key
            for state_key, state in self.track_states.items()
            if frame_index - state.last_seen_frame
            > self.reset_track_state_after_missing_frames
        ]

        for state_key in expired_keys:
            del self.track_states[state_key]

    @staticmethod
    def front_anchor(
        bbox_xyxy: tuple[float, float, float, float],
        anchor_name: str,
    ) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox_xyxy
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        anchors = {
            "top_left": (x1, y1),
            "top_center": (center_x, y1),
            "top_right": (x2, y1),
            "center_left": (x1, center_y),
            "center": (center_x, center_y),
            "center_right": (x2, center_y),
            "bottom_left": (x1, y2),
            "bottom_center": (center_x, y2),
            "bottom_right": (x2, y2),
        }

        if anchor_name not in anchors:
            raise ValueError(f"Unknown anchor name: {anchor_name}")

        return anchors[anchor_name]

    @staticmethod
    def classify_direction(
        bbox_xyxy: tuple[float, float, float, float],
        vehicle_polygons: dict[str, np.ndarray],
    ) -> str | None:
        """
        Assign new tracks based on bounding-box center inside an approach ROI.
        """
        x1, y1, x2, y2 = bbox_xyxy
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        for direction, polygon in vehicle_polygons.items():
            if point_inside_polygon(center, polygon):
                return direction

        return None

    def _find_existing_track_key(
        self,
        track_id: int,
    ) -> tuple[str, int] | None:
        """
        Return the existing (direction, track_id) state key for this
        ByteTrack ID, or None if this is a newly observed track.
        """
        for state_key in self.track_states:
            direction, known_track_id = state_key

            if known_track_id == track_id:
                return direction, known_track_id

        return None

    def _save_evidence(
        self,
        frame_bgr: np.ndarray,
        video_stem: str,
        direction: str,
        frame_index: int,
        track_id: int,
        bbox_xyxy: tuple[float, float, float, float],
    ) -> tuple[str | None, str | None]:
        if not self.save_evidence:
            return None, None

        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        x1, y1, x2, y2 = [int(round(value)) for value in bbox_xyxy]
        height, width = frame_bgr.shape[:2]
        pad = self.evidence_padding

        crop_x1 = max(0, x1 - pad)
        crop_y1 = max(0, y1 - pad)
        crop_x2 = min(width, x2 + pad)
        crop_y2 = min(height, y2 + pad)

        base_name = (
            f"{video_stem}_{direction}_"
            f"frame_{frame_index:08d}_track_{track_id}"
        )

        full_frame_path = self.evidence_dir / f"{base_name}_full.jpg"
        crop_path = self.evidence_dir / f"{base_name}_car.jpg"

        cv2.imwrite(str(full_frame_path), frame_bgr)

        crop = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        cv2.imwrite(str(crop_path), crop)

        return str(full_frame_path), str(crop_path)

    def is_reported_in_current_red_phase(
        self,
        direction: str,
        track_id: int,
    ) -> bool:
        """
        True if this track was already registered as a violation during the
        current red phase.
        """
        return (
            direction,
            track_id,
            self.current_red_phase_id,
        ) in self.reported_events

    def get_track_diagnostics(
        self,
        track_id: int,
        bbox_xyxy: tuple[float, float, float, float],
    ) -> tuple[
        str | None,
        tuple[float, float] | None,
        bool,
        bool,
        bool,
    ]:
        """
        Return:
          direction,
          anchor,
          inside_violation_region,
          armed_for_current_red_phase,
          already_reported_in_current_red_phase
        """
        state_key = self._find_existing_track_key(track_id)

        if state_key is not None:
            direction = state_key[0]
            state = self.track_states[state_key]
        else:
            direction = self.classify_direction(
                bbox_xyxy,
                self.vehicle_polygons,
            )

            if direction is None:
                return None, None, False, False, False

            state = None

        anchor_name = self.front_anchors[direction]
        anchor = self.front_anchor(bbox_xyxy, anchor_name)

        inside_region = point_inside_polygon(
            anchor,
            self.violation_polygon,
        )

        reported = (
            state is not None
            and self.is_reported_in_current_red_phase(
                direction,
                track_id,
            )
        )

        armed = (
            state is not None
            and state.armed_red_phase_id == self.current_red_phase_id
            and not reported
        )

        return direction, anchor, inside_region, armed, reported

    def update_track(
        self,
        frame_bgr: np.ndarray,
        video_name: str,
        video_stem: str,
        frame_index: int,
        fps: float,
        track_id: int,
        bbox_xyxy: tuple[float, float, float, float],
        traffic_light_state: str,
    ) -> tuple[str | None, ViolationEvent | None, tuple[float, float] | None]:
        """
        Shared-R_violation event logic.

        During RED:
        - car observed outside R_violation -> armed;
        - armed car observed inside R_violation -> violation;
        - one event per direction / track / red phase.

        Cars first observed inside R_violation during red are never armed,
        which prevents flagging cars that were already inside when red began.
        """
        state_key = self._find_existing_track_key(track_id)

        if state_key is not None:
            direction = state_key[0]
            state = self.track_states[state_key]
        else:
            direction = self.classify_direction(
                bbox_xyxy,
                self.vehicle_polygons,
            )

            if direction is None:
                return None, None, None

            state_key = (direction, track_id)
            state = DirectionTrackState(last_seen_frame=frame_index)
            self.track_states[state_key] = state

        state.last_seen_frame = frame_index

        anchor_name = self.front_anchors[direction]
        anchor = self.front_anchor(bbox_xyxy, anchor_name)

        inside_violation_region = point_inside_polygon(
            anchor,
            self.violation_polygon,
        )

        if traffic_light_state != "red":
            state.previously_inside_violation_region = (
                inside_violation_region
            )
            return direction, None, anchor

        # Arm whenever this tracked car is visible outside R_violation
        # during the active red interval. This remains true even when it
        # later enters the region.
        if not inside_violation_region:
            state.armed_red_phase_id = self.current_red_phase_id
            state.last_outside_red_frame = frame_index
            state.previously_inside_violation_region = False
            return direction, None, anchor

        # The car is inside R_violation during red. It can only be a
        # violation if this same direction/track was previously armed.
        is_armed_for_current_red = (
            state.armed_red_phase_id == self.current_red_phase_id
        )

        if not is_armed_for_current_red:
            state.previously_inside_violation_region = True
            return direction, None, anchor

        event_key = (
            direction,
            track_id,
            self.current_red_phase_id,
        )

        if event_key in self.reported_events:
            state.previously_inside_violation_region = True
            return direction, None, anchor

        self.reported_events.add(event_key)

        state.previously_inside_violation_region = True

        full_frame_path, crop_path = self._save_evidence(
            frame_bgr=frame_bgr,
            video_stem=video_stem,
            direction=direction,
            frame_index=frame_index,
            track_id=track_id,
            bbox_xyxy=bbox_xyxy,
        )

        x1, y1, x2, y2 = bbox_xyxy

        event = ViolationEvent(
            video_name=video_name,
            direction=direction,
            frame_index=frame_index,
            timestamp_seconds=frame_index / fps,
            red_phase_id=self.current_red_phase_id,
            track_id=track_id,
            anchor_name=anchor_name,
            anchor_x=anchor[0],
            anchor_y=anchor[1],
            bbox_x1=x1,
            bbox_y1=y1,
            bbox_x2=x2,
            bbox_y2=y2,
            traffic_light_state=traffic_light_state,
            evidence_frame_path=full_frame_path,
            evidence_crop_path=crop_path,
        )

        self.events.append(event)

        return direction, event, anchor