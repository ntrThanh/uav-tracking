"""Small engine registry. New engines register a factory without changing the monitor."""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Protocol

import numpy as np

from src.common.config import DetectorConfig, TrackerConfig
from src.detect.vehicle import Detections, VehicleDetector
from src.track.tracker import Track, VehicleTracker


class IoUTracker:
    """Lightweight IoU association baseline for fixed cameras."""

    def __init__(self, config: TrackerConfig) -> None:
        self.max_age = config.track_buffer
        self._next_id = 1
        self._active: dict[int, tuple[np.ndarray, int, int]] = {}

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        x1, y1 = np.maximum(a[:2], b[:2])
        x2, y2 = np.minimum(a[2:], b[2:])
        area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
        aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
        bb = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
        return area / max(aa + bb - area, 1e-9)

    def update(self, detections: Detections, frame_index: int, time_s: float) -> list[Track]:
        self._active = {key: value for key, value in self._active.items() if frame_index - value[2] <= self.max_age}
        matches: list[tuple[float, int, int]] = []
        for idx, box in enumerate(detections.xyxy):
            for track_id, (previous, cls, _) in self._active.items():
                if cls == int(detections.class_id[idx]):
                    score = self._iou(previous, box)
                    if score >= 0.3:
                        matches.append((score, track_id, idx))
        matches.sort(reverse=True)
        assigned: dict[int, int] = {}
        used: set[int] = set()
        for _, track_id, idx in matches:
            if idx not in assigned and track_id not in used:
                assigned[idx] = track_id
                used.add(track_id)
        result = []
        for idx, box in enumerate(detections.xyxy):
            track_id = assigned.get(idx)
            if track_id is None:
                track_id, self._next_id = self._next_id, self._next_id + 1
            cls = int(detections.class_id[idx])
            self._active[track_id] = (box.copy(), cls, frame_index)
            result.append(Track(track_id, box.copy(), float(detections.conf[idx]), cls, detections.class_name[idx]))
        return result


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> Detections: ...


class Tracker(Protocol):
    def update(self, detections: Detections, frame_index: int, time_s: float) -> list[Track]: ...


DETECTORS: dict[str, Callable[[DetectorConfig], Detector]] = {}
TRACKERS: dict[str, Callable[[TrackerConfig], Tracker]] = {}


def register_detector(name: str, factory: Callable[[DetectorConfig], Detector]) -> None:
    if name in DETECTORS:
        raise ValueError(f"Detector already registered: {name}")
    DETECTORS[name] = factory


def register_tracker(name: str, factory: Callable[[TrackerConfig], Tracker]) -> None:
    if name in TRACKERS:
        raise ValueError(f"Tracker already registered: {name}")
    TRACKERS[name] = factory


def create_detector(name: str, config: DetectorConfig) -> Detector:
    try:
        return DETECTORS[name](config)
    except KeyError as exc:
        raise ValueError(f"Unknown detector: {name}") from exc


def create_tracker(name: str, config: TrackerConfig) -> Tracker:
    try:
        return TRACKERS[name](config)
    except KeyError as exc:
        raise ValueError(f"Unknown tracker: {name}") from exc


register_detector("yolo11n", lambda cfg: VehicleDetector(replace(cfg, weights="weights/yolo11n.pt")))
register_detector("yolo11m", lambda cfg: VehicleDetector(replace(cfg, weights="../weights/pretrained/yolo11m.pt")))
register_tracker("bytetrack", VehicleTracker)
register_tracker("iou", IoUTracker)
