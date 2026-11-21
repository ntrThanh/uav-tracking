"""Run a plate detector inside each tracked vehicle and restore image coordinates."""
from __future__ import annotations

from dataclasses import replace

from .detectors import BaseDetector
from .schema import DetectionResult, Frame, TrackingResult


class TrackROIDetector:
    def __init__(self, detector: BaseDetector) -> None:
        self.detector = detector

    def detect(self, frame: Frame, tracks: TrackingResult) -> DetectionResult:
        regions = []
        for track in tracks.tracks:
            x0, y0, x1, y1 = (int(value) for value in track.bbox)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(frame.meta.original_width, x1), min(frame.meta.original_height, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            cropped = frame.image[y0:y1, x0:x1]
            subframe = Frame(cropped, replace(frame.meta, original_width=x1-x0,
                                             original_height=y1-y0))
            result = self.detector.detect(subframe)
            regions.extend(replace(detection,
                                   bbox=(detection.bbox[0]+x0, detection.bbox[1]+y0,
                                         detection.bbox[2]+x0, detection.bbox[3]+y0),
                                   metadata={**detection.metadata, "parent_track_id": track.track_id})
                           for detection in result.detections)
        return DetectionResult(frame.meta, tuple(regions), self.detector.model.model_id)

    def close(self) -> None:
        self.detector.close()
