"""Vehicle detector contract and replaceable implementations."""
from .detectors import VehicleDetector, available_backends, create_detector, register_backend

__all__ = ["VehicleDetector", "available_backends", "create_detector", "register_backend"]
