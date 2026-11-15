"""Geometry helpers shared by tracking and rule modules."""
from __future__ import annotations

from typing import Sequence

Point = tuple[float, float]


def reference_point(xyxy: Sequence[float]) -> Point:
    """Bottom-centre of a box, P = ((x1 + x2) / 2, y2): where the vehicle touches the road."""
    x1, _, x2, y2 = (float(v) for v in xyxy[:4])
    return ((x1 + x2) / 2.0, y2)


def denormalize(point: Point, width: int, height: int) -> Point:
    """Normalized (0-1) point to pixel coordinates."""
    return (point[0] * width, point[1] * height)


def normalize(point: Point, width: int, height: int) -> Point:
    """Pixel point to normalized (0-1) coordinates."""
    return (point[0] / width, point[1] / height)
