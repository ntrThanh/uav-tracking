"""Drawing helpers for the annotated output video."""
from __future__ import annotations

import colorsys

import cv2
import numpy as np

from src.common.config import CameraConfig
from src.track.tracker import Track, VehicleTracker

_DIRECTION_VECTORS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def color_for_id(track_id: int) -> tuple[int, int, int]:
    """Deterministic, well-separated BGR colour per track id."""
    hue = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def _put_label(img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int]) -> None:
    (tw, th), base = cv2.getTextSize(text, _FONT, 0.5, 1)
    x, y = org
    y = max(y, th + base + 2)
    cv2.rectangle(img, (x, y - th - base - 2), (x + tw + 4, y), color, -1)
    cv2.putText(img, text, (x + 2, y - base - 1), _FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_tracks(img: np.ndarray, tracks: list[Track], tracker: VehicleTracker, trail_len: int = 30) -> np.ndarray:
    """Draw boxes, ids, reference points and recent trails in place."""
    for tr in tracks:
        x1, y1, x2, y2 = (int(round(float(v))) for v in tr.xyxy)
        color = color_for_id(tr.track_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        _put_label(img, f"#{tr.track_id} {tr.class_name} {tr.conf:.2f}", (x1, y1), color)
        px, py = tr.ref_point
        cv2.circle(img, (int(px), int(py)), 4, color, -1)
        hist = tracker.history(tr.track_id)
        if hist is not None and trail_len > 1:
            pts = [(int(p.x), int(p.y)) for p in list(hist.points)[-trail_len:]]
            for a, b in zip(pts, pts[1:]):
                cv2.line(img, a, b, color, 2)
    return img


def draw_hud(img: np.ndarray, lines: list[str]) -> np.ndarray:
    """Semi-transparent status box in the top-left corner."""
    pad, lh = 6, 20
    width = max((cv2.getTextSize(t, _FONT, 0.55, 1)[0][0] for t in lines), default=0) + 2 * pad
    height = lh * len(lines) + pad
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for i, text in enumerate(lines):
        cv2.putText(img, text, (pad, lh * (i + 1)), _FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_camera_geometry(img: np.ndarray, cam: CameraConfig) -> np.ndarray:
    """Overlay stop line (with allowed direction), zones and light ROI from the camera config."""
    h, w = img.shape[:2]
    if cam.stop_line is not None:
        (ax, ay), (bx, by) = cam.stop_line.to_pixels(w, h)
        a, b = (int(ax), int(ay)), (int(bx), int(by))
        cv2.line(img, a, b, (0, 0, 255), 3)
        mx, my = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
        dx, dy = _DIRECTION_VECTORS[cam.stop_line.allowed_direction]
        cv2.arrowedLine(img, (mx, my), (mx + 40 * dx, my + 40 * dy), (0, 0, 255), 2, tipLength=0.4)
        _put_label(img, f"stop line / allowed: {cam.stop_line.allowed_direction}", (a[0], a[1] - 6), (0, 0, 255))
    for zone in cam.zones:
        pts = np.array(zone.to_pixels(w, h), dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, (255, 160, 0), 2)
        _put_label(img, zone.name, (int(pts[0, 0, 0]), int(pts[0, 0, 1])), (255, 160, 0))
    if cam.light_roi is not None:
        x0, y0, x1, y1 = cam.light_roi.to_pixels(w, h)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 2)
        _put_label(img, "light", (x0, y0 - 6), (0, 255, 255))
    return img
