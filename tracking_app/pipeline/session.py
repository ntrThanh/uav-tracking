"""Compose vehicle detection, tracking, plate detection and OCR per frame."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from tracking_app.engine.schema import Frame, TrackingResult

from tracking_app.config import Settings
from tracking_app.ocr import PlateReader, create_reader
from tracking_app.plate import PlateDetector, create_detector as create_plate_detector
from tracking_app.tracking import VehicleTracker, create_tracker
from tracking_app.vehicle import VehicleDetector, create_detector as create_vehicle_detector


@dataclass(frozen=True)
class PlateRead:
    track_id: int
    text: str
    confidence: float
    bbox: tuple[int, int, int, int] | None = field(default=None, compare=False)
    detection_confidence: float = field(default=0.0, compare=False)


class InferencePipeline:
    def __init__(self, vehicle: VehicleDetector, tracker: VehicleTracker,
                 plate: PlateDetector, reader: PlateReader, plate_interval: int) -> None:
        self.vehicle = vehicle
        self.tracker = tracker
        self.plate = plate
        self.reader = reader
        self.plate_interval = plate_interval
        self.last_timings: tuple[tuple[str, float], ...] = ()

    def process(self, frame: Frame) -> tuple[TrackingResult, tuple[PlateRead, ...]]:
        timings: list[tuple[str, float]] = []
        started = perf_counter()
        detections = self.vehicle.detect(frame)
        timings.append(("vehicle_detection", (perf_counter() - started) * 1000))
        started = perf_counter()
        tracks = self.tracker.update(frame, detections)
        timings.append(("tracking", (perf_counter() - started) * 1000))
        reads: list[PlateRead] = []
        if tracks.tracks and frame.meta.frame_id % self.plate_interval == 0:
            started = perf_counter()
            regions = self.plate.detect(frame, tracks).detections
            timings.append(("plate_detection", (perf_counter() - started) * 1000))
            for region in regions:
                track_id = region.metadata.get("parent_track_id")
                if track_id is None:
                    continue
                x1, y1, x2, y2 = map(int, region.bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.meta.original_width, x2), min(frame.meta.original_height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue 
                started = perf_counter()
                result = self.reader.read(frame.image[y1:y2, x1:x2])
                timings.append(("ocr", (perf_counter() - started) * 1000))
                reads.append(PlateRead(int(track_id), result[0] if result else "",
                                       result[1] if result else 0.0, (x1, y1, x2, y2),
                                       region.confidence))
        self.last_timings = tuple(timings)
        return tracks, tuple(reads)

    def close(self) -> None:
        first_error = None
        for component in (self.plate, self.tracker, self.vehicle):
            try:
                component.close()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error:
            raise first_error

    def __enter__(self) -> "InferencePipeline":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def create_pipeline(settings: Settings) -> InferencePipeline:
    # Construct in dependency order and release partial construction on failure.
    from tracking_app.plugins import load_plugins
    load_plugins()
    reader = create_reader(settings)
    vehicle = tracker = plate = None
    try:
        vehicle = create_vehicle_detector(settings)
        tracker = create_tracker(settings)
        plate = create_plate_detector(settings)
        return InferencePipeline(vehicle, tracker, plate, reader, settings.plate_interval)
    except Exception:
        for component in (plate, tracker, vehicle):
            if component is not None:
                component.close()
        raise
