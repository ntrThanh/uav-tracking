"""Model-independent data contracts. Boxes are xyxy in original-image pixels."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

BBox = tuple[float, float, float, float]


def _bbox(value: Any) -> BBox:
    if len(value) != 4:
        raise ValueError("bbox must contain x1,y1,x2,y2")
    box = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"invalid xyxy bbox: {box}")
    return box  # type: ignore[return-value]


@dataclass(frozen=True)
class FrameMetadata:
    source_id: str
    frame_id: int
    timestamp: float
    original_width: int
    original_height: int
    processing_width: int | None = None
    processing_height: int | None = None
    camera: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id or self.frame_id < 0 or not math.isfinite(self.timestamp):
            raise ValueError("invalid source, frame ID or timestamp")
        if min(self.original_width, self.original_height) <= 0:
            raise ValueError("original dimensions must be positive")
        if self.processing_width is not None and self.processing_width <= 0:
            raise ValueError("processing width must be positive")
        if self.processing_height is not None and self.processing_height <= 0:
            raise ValueError("processing height must be positive")


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    meta: FrameMetadata

    def __post_init__(self) -> None:
        if self.image.ndim != 3 or self.image.shape[2] != 3 or self.image.dtype != np.uint8:
            raise ValueError("frame image must be BGR uint8 HxWx3")
        if self.image.shape[:2] != (self.meta.original_height, self.meta.original_width):
            raise ValueError("frame dimensions differ from metadata")


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str
    detector_id: str
    mask: Any = None
    keypoints: Any = None
    embedding: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox", _bbox(self.bbox))
        if not 0 <= self.confidence <= 1 or not math.isfinite(self.confidence):
            raise ValueError("detection confidence must be in [0,1]")
        if not self.class_name or not self.detector_id:
            raise ValueError("detection requires class name and detector ID")


@dataclass(frozen=True)
class DetectionResult:
    frame: FrameMetadata
    detections: tuple[Detection, ...]
    model_id: str
    latency_ms: float = 0.0


@dataclass(frozen=True)
class Track:
    track_id: int
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str
    state: str
    timestamp: float
    trajectory: tuple[tuple[float, float], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox", _bbox(self.bbox))
        if self.track_id <= 0 or not 0 <= self.confidence <= 1 or not math.isfinite(self.timestamp):
            raise ValueError("invalid track ID, confidence or timestamp")


@dataclass(frozen=True)
class TrackingResult:
    frame: FrameMetadata
    tracks: tuple[Track, ...]
    tracker_id: str
    latency_ms: float = 0.0


@dataclass(frozen=True)
class RuleContext:
    frame: Frame
    detections: tuple[Detection, ...] = ()
    tracks: tuple[Track, ...] = ()
    signals: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Event:
    event_id: str
    event_type: str
    source_id: str
    rule_id: str
    timestamp: float
    status: str = "candidate"
    track_id: int | None = None
    confidence: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    model_versions: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SystemError:
    source_id: str
    stage: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(frozen=True)
class ModelMetadata:
    model_id: str
    task: str
    architecture: str
    weights: str
    runtime: str
    input_size: tuple[int, int]
    precision: str
    classes: dict[int, str]
    version: str = "1"
    checksum: str | None = None
    preprocess: dict[str, Any] = field(default_factory=dict)
    postprocess: dict[str, Any] = field(default_factory=dict)


def detections_from_numpy(rows: np.ndarray, frame: FrameMetadata, model_id: str,
                          classes: dict[int, str]) -> DetectionResult:
    """Convert legacy Nx6 xyxy,score,class rows; class mapping is explicit."""
    array = np.asarray(rows, dtype=float)
    if array.size == 0:
        array = array.reshape(0, 6)
    if array.ndim != 2 or array.shape[1] != 6 or not np.isfinite(array).all():
        raise ValueError("detections must be finite Nx6")
    output = []
    for x1, y1, x2, y2, score, raw_class in array:
        if int(raw_class) != raw_class or int(raw_class) not in classes:
            raise ValueError(f"unmapped class ID: {raw_class}")
        class_id = int(raw_class)
        if min(x1, y1) < 0 or x2 > frame.original_width or y2 > frame.original_height:
            raise ValueError("detection box outside original image")
        output.append(Detection((x1, y1, x2, y2), float(score), class_id, classes[class_id], model_id))
    return DetectionResult(frame, tuple(output), model_id)


def detections_to_numpy(result: DetectionResult) -> np.ndarray:
    return np.asarray([(*d.bbox, d.confidence, d.class_id) for d in result.detections], dtype=float).reshape(-1, 6)


def tracks_from_numpy(rows: np.ndarray, frame: FrameMetadata, tracker_id: str,
                      classes: dict[int, str]) -> TrackingResult:
    array = np.asarray(rows, dtype=float)
    if array.size == 0:
        array = array.reshape(0, 7)
    if array.ndim != 2 or array.shape[1] != 7 or not np.isfinite(array).all():
        raise ValueError("tracks must be finite Nx7")
    output = []
    for x1, y1, x2, y2, track_id, score, raw_class in array:
        if int(raw_class) != raw_class or int(raw_class) not in classes:
            raise ValueError(f"unmapped track class ID: {raw_class}")
        if int(track_id) != track_id or track_id <= 0:
            raise ValueError("invalid track ID")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("invalid track box")
        # Motion prediction can extend beyond the image as a vehicle leaves.
        x1, y1 = max(0., x1), max(0., y1)
        x2, y2 = min(frame.original_width, x2), min(frame.original_height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cls = int(raw_class)
        output.append(Track(int(track_id), (x1, y1, x2, y2), float(score), cls,
                            classes[cls], "active", frame.timestamp))
    return TrackingResult(frame, tuple(output), tracker_id)
