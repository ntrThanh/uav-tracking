"""Tracker contract and replaceable implementations."""
from .trackers import VehicleTracker, available_backends, create_tracker, register_backend

__all__ = ["VehicleTracker", "available_backends", "create_tracker", "register_backend"]
