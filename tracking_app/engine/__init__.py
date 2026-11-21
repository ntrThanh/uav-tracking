"""Frame, ONNX and tracking adapters owned by the desktop app."""

from .schema import Detection, DetectionResult, Frame, FrameMetadata, Track, TrackingResult

__all__ = ["Frame", "FrameMetadata", "Detection", "DetectionResult", "Track", "TrackingResult"]
