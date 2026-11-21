"""Plate regions are detected inside tracked vehicle boxes."""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Protocol

from tracking_app.engine.detectors import build_detector
from tracking_app.engine.models import ModelRegistry
from tracking_app.engine.roi import TrackROIDetector
from tracking_app.engine.schema import DetectionResult, Frame, TrackingResult

from tracking_app.config import CATALOG, Settings


class PlateDetector(Protocol):
    def detect(self, frame: Frame, tracks: TrackingResult) -> DetectionResult: ...
    def close(self) -> None: ...


class EnginePlateDetector:
    def __init__(self, settings: Settings) -> None:
        catalog = ModelRegistry.from_json(CATALOG)
        model = catalog.get(settings.plate_model)
        registry = ModelRegistry(catalog.base_dir)
        registry.register(replace(model, postprocess={**model.postprocess,
                                                      "conf": settings.plate_confidence, "iou": settings.plate_iou}))
        self.node = TrackROIDetector(build_detector(settings.plate_model, registry, settings.device))

    def detect(self, frame: Frame, tracks: TrackingResult) -> DetectionResult:
        return self.node.detect(frame, tracks)

    def close(self) -> None:
        self.node.close()


Factory = Callable[[Settings], PlateDetector]
_BACKENDS: dict[str, Factory] = {}


def register_backend(name: str, factory: Factory) -> None:
    if name in _BACKENDS:
        raise ValueError(f"Plate backend already registered: {name}")
    _BACKENDS[name] = factory


def available_backends() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))


def create_detector(settings: Settings) -> PlateDetector:
    try:
        return _BACKENDS[settings.plate_backend](settings)
    except KeyError as exc:
        raise ValueError(f"Unknown plate backend: {settings.plate_backend}") from exc


register_backend("engine", EnginePlateDetector)
