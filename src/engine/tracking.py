"""Session-local adapter for the existing five MOT algorithms."""
from __future__ import annotations

from .schema import DetectionResult, Frame, TrackingResult, detections_to_numpy, tracks_from_numpy


class TrackerAdapter:
    def __init__(self, algorithm: str, classes: dict[int, str], config_path: str | None = None,
                 frame_rate: int = 30, det_thresh: float | None = None) -> None:
        from src.trackers import available_trackers
        if algorithm not in available_trackers():
            raise ValueError(f"unknown tracker: {algorithm}")
        self.algorithm = algorithm
        self.classes = classes
        self.config_path = config_path
        self.frame_rate = frame_rate
        self.det_thresh = det_thresh
        self._tracker = None

    def load(self) -> None:
        from pathlib import Path
        from src.trackers import Tracker, load_config
        kwargs = {"frame_rate": self.frame_rate}
        if self.det_thresh is not None:
            kwargs["det_thresh"] = self.det_thresh
        config = load_config(self.algorithm, Path(self.config_path) if self.config_path else None, **kwargs)
        self._tracker = Tracker(self.algorithm, config, classes=tuple(self.classes))

    def update(self, frame: Frame, detections: DetectionResult) -> TrackingResult:
        if detections.frame != frame.meta:
            raise ValueError("detections belong to another frame")
        if self._tracker is None:
            self.load()
        assert self._tracker is not None
        rows = self._tracker.update(detections_to_numpy(detections), frame.image if self._tracker.requires_image else None)
        return tracks_from_numpy(rows, frame.meta, self.algorithm, self.classes)

    def reset(self) -> None:
        self.close()
        self.load()

    def close(self) -> None:
        self._tracker = None
