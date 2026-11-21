"""Tracker adapter factory backed by the app's local tracker registry."""
from __future__ import annotations

from typing import Callable, Protocol

from tracking_app.engine.models import ModelRegistry
from tracking_app.engine.schema import DetectionResult, Frame, TrackingResult
from tracking_app.engine.tracking import TrackerAdapter

from tracking_app.config import CATALOG, Settings


class VehicleTracker(Protocol):
    def update(self, frame: Frame, detections: DetectionResult) -> TrackingResult: ...
    def close(self) -> None: ...


Factory = Callable[[Settings], VehicleTracker]
_BACKENDS: dict[str, Factory] = {}


def register_backend(name: str, factory: Factory) -> None:
    if name in _BACKENDS:
        raise ValueError(f"Tracker backend already registered: {name}")
    _BACKENDS[name] = factory


def available_backends() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))


def create_tracker(settings: Settings) -> VehicleTracker:
    try:
        return _BACKENDS[settings.tracker_backend](settings)
    except KeyError as exc:
        raise ValueError(f"Unknown tracker backend: {settings.tracker_backend}") from exc


def _engine(settings: Settings) -> VehicleTracker:
    model = ModelRegistry.from_json(CATALOG).get(settings.vehicle_model)
    return TrackerAdapter(settings.tracker, model.classes, frame_rate=settings.frame_rate,
                          det_thresh=settings.track_threshold)


register_backend("engine", _engine)
