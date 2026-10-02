from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


def polygon_from_points(
    points: Iterable[Iterable[int]],
) -> np.ndarray:
    """
    Convert [[x, y], ...] configuration points into OpenCV polygon format.
    """
    polygon = np.asarray(list(points), dtype=np.int32)

    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError(
            "Polygon points must have shape [[x1, y1], [x2, y2], ...]."
        )

    return polygon.reshape((-1, 1, 2))


def point_inside_polygon(
    point: tuple[float, float],
    polygon: np.ndarray,
) -> bool:
    """
    True when a point is inside or on a polygon boundary.
    """
    x, y = point

    return cv2.pointPolygonTest(
        polygon,
        (float(x), float(y)),
        False,
    ) >= 0


def bbox_bottom_center(
    bbox_xyxy: tuple[float, float, float, float],
) -> tuple[float, float]:
    """
    Road-contact proxy for an axis-aligned bounding box.
    """
    x1, _, x2, y2 = bbox_xyxy
    return ((x1 + x2) / 2.0, y2)


def polygon_mask(
    polygon: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """
    Binary uint8 mask where the polygon area equals 255.
    """
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

    cv2.fillPoly(mask, [polygon], 255)

    return mask


def bbox_polygon_iou(
    bbox_xyxy: tuple[float, float, float, float],
    polygon: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> float:
    """
    Pixel-mask IoU between a bounding box and a polygonal region.

    IoU = area(bbox ∩ polygon) / area(bbox ∪ polygon)

    This is suitable for fixed geometric road regions. It is used only for
    small ROI polygons, not large-scale arbitrary segmentation.
    """
    x1, y1, x2, y2 = bbox_xyxy

    x1 = max(0, min(frame_width - 1, int(round(x1))))
    y1 = max(0, min(frame_height - 1, int(round(y1))))
    x2 = max(0, min(frame_width - 1, int(round(x2))))
    y2 = max(0, min(frame_height - 1, int(round(y2))))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    bbox_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    region_mask = polygon_mask(
        polygon,
        frame_width,
        frame_height,
    )

    cv2.rectangle(
        bbox_mask,
        (x1, y1),
        (x2, y2),
        255,
        thickness=-1,
    )

    intersection = cv2.countNonZero(
        cv2.bitwise_and(bbox_mask, region_mask)
    )

    union = cv2.countNonZero(
        cv2.bitwise_or(bbox_mask, region_mask)
    )

    return 0.0 if union == 0 else intersection / union