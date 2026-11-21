"""OCR contract, normalization and replaceable readers."""
from .readers import PlateReader, available_backends, create_reader, normalize_plate, register_backend

__all__ = ["PlateReader", "available_backends", "create_reader", "normalize_plate", "register_backend"]
