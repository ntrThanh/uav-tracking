"""The three fine-tuned detectors behind one interface (base.Detector).

The id is the folder name of the detection cache, results/dets/<id>/. Each model is imported only when it is built,
so YOLO and Faster R-CNN run without rfdetr installed and the other way round.
"""
from __future__ import annotations

import importlib

from .base import Detector

_DETECTORS = {
    "yolo11m": ("yolo", "Yolo11m"),
    "frcnn_r50": ("frcnn", "FasterRCNN"),
    "rfdetr_m": ("rfdetr", "RFDETRMedium"),
}


def available_detectors() -> list[str]:
    return sorted(_DETECTORS)


def build_detector(name: str, **kwargs) -> Detector:
    if name not in _DETECTORS:
        raise KeyError(f"Unknown detector {name!r}; available: {available_detectors()}")
    module, cls = _DETECTORS[name]
    return getattr(importlib.import_module(f"{__name__}.{module}"), cls)(**kwargs)


__all__ = ["Detector", "available_detectors", "build_detector"]
