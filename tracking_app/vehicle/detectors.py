"""Vehicle detection depends on a contract, not one model architecture."""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Protocol

from tracking_app.engine.detectors import build_detector
from tracking_app.engine.models import ModelRegistry
from tracking_app.engine.schema import DetectionResult, Frame

from tracking_app.config import CATALOG, Settings


class VehicleDetector(Protocol):
    def detect(self, frame: Frame) -> DetectionResult: ...
    def close(self) -> None: ...


class FilteredVehicleDetector:
    def __init__(self, detector: VehicleDetector, names: set[str]) -> None:
        self.detector = detector
        self.names = names

    def detect(self, frame: Frame) -> DetectionResult:
        result = self.detector.detect(frame)
        return replace(result, detections=tuple(d for d in result.detections
                                                 if d.class_name.lower() in self.names))

    def close(self) -> None:
        self.detector.close()


Factory = Callable[[Settings], VehicleDetector]
_BACKENDS: dict[str, Factory] = {}


def register_backend(name: str, factory: Factory) -> None:
    if name in _BACKENDS:
        raise ValueError(f"Vehicle backend already registered: {name}")
    _BACKENDS[name] = factory


def available_backends() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))


def create_detector(settings: Settings) -> VehicleDetector:
    try:
        return _BACKENDS[settings.vehicle_backend](settings)
    except KeyError as exc:
        raise ValueError(f"Unknown vehicle backend: {settings.vehicle_backend}") from exc


def _engine(settings: Settings) -> VehicleDetector:
    catalog = ModelRegistry.from_json(CATALOG)
    model = catalog.get(settings.vehicle_model)
    registry = ModelRegistry(catalog.base_dir)
    registry.register(replace(model, postprocess={**model.postprocess,
                                                  "conf": settings.confidence, "iou": settings.iou}))
    detector = build_detector(settings.vehicle_model, registry, settings.device)
    names = {name.strip().lower() for name in settings.vehicle_classes.split(",") if name.strip()}
    return FilteredVehicleDetector(detector, names)


register_backend("engine", _engine)
