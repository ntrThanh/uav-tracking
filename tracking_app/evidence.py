"""Draw a track in original-frame coordinates for visual evidence."""
from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


def draw_trajectory(image: np.ndarray, points: Sequence[dict], track_id: int) -> np.ndarray:
    canvas = image.copy()
    if not points:
        return canvas
    height, width = canvas.shape[:2]
    thickness = max(2, round(min(width, height) / 320))
    radius = max(5, thickness * 2)
    step = max(1, len(points) // 2000)
    visible = list(points[::step])
    if visible[-1] is not points[-1]:
        visible.append(points[-1])
    path = np.asarray([(round(point["x"]), round(point["y"])) for point in visible], dtype=np.int32)
    if len(path) > 1:
        cv2.polylines(canvas, [path], False, (75, 165, 30), thickness, cv2.LINE_AA)
    first = (round(points[0]["x"]), round(points[0]["y"]))
    last = (round(points[-1]["x"]), round(points[-1]["y"]))
    cv2.circle(canvas, first, radius, (0, 165, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, last, radius, (60, 190, 40), -1, cv2.LINE_AA)
    x1, y1, x2, y2 = (round(points[-1][key]) for key in ("x1", "y1", "x2", "y2"))
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 190, 40), thickness, cv2.LINE_AA)
    label = f"Track #{track_id} | last seen {points[-1]['video_time']:.1f}s"
    cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), max(1, thickness - 1), cv2.LINE_AA)
    return canvas
