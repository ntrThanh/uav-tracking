"""License plate region detector contract and replaceable implementations."""
from .detectors import PlateDetector, available_backends, create_detector, register_backend

__all__ = ["PlateDetector", "available_backends", "create_detector", "register_backend"]
