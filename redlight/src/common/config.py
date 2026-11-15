"""Configuration loading for the red-light violation pipeline.

Two JSON files drive a run:

* pipeline config (``config/pipeline.json``): model paths, thresholds, runtime options.
* camera config (``config/<camera_id>.json``): stop line, zones and traffic-light ROI
  in normalized image coordinates (0-1), as drawn by the user in the web UI.

Every tunable parameter lives in one of these files; modules never hard-code them.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_DIRECTIONS = ("up", "down", "left", "right")
LIGHT_STATES = ("red", "yellow", "green", "unknown")

T = TypeVar("T")
Point = tuple[float, float]


def resolve_path(path: str | Path) -> Path:
    """Return ``path`` as an absolute path, resolving relative ones against the project root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def _build(dc: type[T], data: dict[str, Any] | None, section: str) -> T:
    """Instantiate dataclass ``dc`` from ``data``, rejecting unknown keys."""
    data = dict(data or {})
    allowed = {f.name for f in fields(dc)}  # type: ignore[arg-type]
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys in config section '{section}': {unknown}")
    return dc(**data)


def _unit(value: Any, name: str) -> float:
    """Validate a normalized coordinate in [0, 1]."""
    v = float(value)
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {v}")
    return v


def _point(value: Any, name: str) -> Point:
    """Validate a normalized [x, y] pair."""
    if len(value) != 2:
        raise ValueError(f"{name} must be [x, y], got {value!r}")
    return (_unit(value[0], f"{name}.x"), _unit(value[1], f"{name}.y"))


# --------------------------------------------------------------------------- pipeline


@dataclass
class DetectorConfig:
    """Vehicle detector (Ultralytics YOLO) parameters."""

    weights: str = "weights/yolo11n.pt"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.5
    agnostic_nms: bool = True  # merge overlapping boxes across classes (car vs truck on one vehicle)
    device: str = "auto"  # "auto" | "cpu" | "cuda:0" | ...
    half: bool = False
    classes: dict[int, str] = field(
        default_factory=lambda: {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    )

    def __post_init__(self) -> None:
        self.classes = {int(k): str(v) for k, v in self.classes.items()}
        if not self.classes:
            raise ValueError("detector.classes must not be empty")
        if not 0.0 < self.conf < 1.0:
            raise ValueError("detector.conf must be in (0, 1)")


@dataclass
class TrackerConfig:
    """ByteTrack parameters (Ultralytics implementation) plus per-track history length."""

    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    track_buffer: int = 30  # frames a lost track is kept before it is dropped
    match_thresh: float = 0.8
    fuse_score: bool = True
    history_len: int = 150  # reference points kept per track

    def __post_init__(self) -> None:
        if self.track_low_thresh > self.track_high_thresh:
            raise ValueError("tracker.track_low_thresh must be <= track_high_thresh")
        if self.history_len < 2:
            raise ValueError("tracker.history_len must be >= 2")


@dataclass
class VideoConfig:
    """Which part of the input video to process."""

    start_s: float = 0.0
    end_s: float | None = None
    stride: int = 1

    def __post_init__(self) -> None:
        if self.stride < 1:
            raise ValueError("video.stride must be >= 1")
        if self.end_s is not None and self.end_s <= self.start_s:
            raise ValueError("video.end_s must be greater than start_s")


@dataclass
class OutputConfig:
    """Annotated video and logging options."""

    write_video: bool = True
    codec: str = "mp4v"
    trail_len: int = 30
    log_every: int = 50


_PIPELINE_SECTIONS: dict[str, type] = {
    "detector": DetectorConfig,
    "tracker": TrackerConfig,
    "video": VideoConfig,
    "output": OutputConfig,
}


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration (models, thresholds, runtime)."""

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        unknown = sorted(set(data) - set(_PIPELINE_SECTIONS))
        if unknown:
            raise ValueError(f"Unknown pipeline config sections: {unknown}")
        return cls(**{name: _build(dc, data.get(name), name) for name, dc in _PIPELINE_SECTIONS.items()})

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        """Read a pipeline config JSON file."""
        path = resolve_path(path)
        with path.open("r", encoding="utf-8") as f:
            cfg = cls.from_dict(json.load(f))
        log.info("Loaded pipeline config %s", path)
        return cfg


# --------------------------------------------------------------------------- camera


@dataclass
class StopLine:
    """Stop line from A to B; ``allowed_direction`` is the travel direction allowed to cross it."""

    a: Point
    b: Point
    allowed_direction: str = "down"

    def __post_init__(self) -> None:
        self.a = _point(self.a, "stop_line.a")
        self.b = _point(self.b, "stop_line.b")
        if self.a == self.b:
            raise ValueError("stop_line endpoints must differ")
        if self.allowed_direction not in ALLOWED_DIRECTIONS:
            raise ValueError(f"stop_line.allowed_direction must be one of {ALLOWED_DIRECTIONS}")

    def to_pixels(self, width: int, height: int) -> tuple[Point, Point]:
        """Endpoints in pixel coordinates for a frame of the given size."""
        return (self.a[0] * width, self.a[1] * height), (self.b[0] * width, self.b[1] * height)

    def to_dict(self) -> dict[str, Any]:
        return {"a": list(self.a), "b": list(self.b), "allowed_direction": self.allowed_direction}


@dataclass
class Zone:
    """Named polygon (e.g. the intersection area) in normalized coordinates."""

    name: str
    polygon: list[Point]

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ValueError(f"zone '{self.name}' needs at least 3 points")
        self.polygon = [_point(p, f"zone.{self.name}.polygon[{i}]") for i, p in enumerate(self.polygon)]

    def to_pixels(self, width: int, height: int) -> list[Point]:
        return [(x * width, y * height) for x, y in self.polygon]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "polygon": [list(p) for p in self.polygon]}


@dataclass
class LightROI:
    """Rectangle around the traffic light (normalized x, y, w, h)."""

    x: float
    y: float
    w: float
    h: float

    def __post_init__(self) -> None:
        self.x, self.y = _unit(self.x, "light_roi.x"), _unit(self.y, "light_roi.y")
        self.w, self.h = float(self.w), float(self.h)
        if self.w <= 0 or self.h <= 0 or self.x + self.w > 1.0 + 1e-6 or self.y + self.h > 1.0 + 1e-6:
            raise ValueError("light_roi must be a non-empty rectangle inside the image")

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Integer (x0, y0, x1, y1) in pixels, at least 1 px wide/high."""
        x0, y0 = int(round(self.x * width)), int(round(self.y * height))
        x1, y1 = int(round((self.x + self.w) * width)), int(round((self.y + self.h) * height))
        return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)

    def to_dict(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass
class LightSegment:
    """Manual light state for [start_s, end_s) when the light is not visible in the frame."""

    start_s: float
    end_s: float
    state: str

    def __post_init__(self) -> None:
        if self.state not in LIGHT_STATES:
            raise ValueError(f"light_override.state must be one of {LIGHT_STATES}")
        if self.end_s <= self.start_s:
            raise ValueError("light_override segment must have end_s > start_s")

    def to_dict(self) -> dict[str, Any]:
        return {"start_s": self.start_s, "end_s": self.end_s, "state": self.state}


@dataclass
class CameraConfig:
    """Per-camera geometry drawn by the user on the reference frame."""

    camera_id: str = "cam01"
    reference_frame: str | None = None
    stop_line: StopLine | None = None
    zones: list[Zone] = field(default_factory=list)
    light_roi: LightROI | None = None
    light_override: list[LightSegment] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraConfig":
        allowed = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"Unknown keys in camera config: {unknown}")
        stop_line = data.get("stop_line")
        roi = data.get("light_roi")
        override = data.get("light_override")
        return cls(
            camera_id=str(data.get("camera_id", "cam01")),
            reference_frame=data.get("reference_frame"),
            stop_line=_build(StopLine, stop_line, "stop_line") if stop_line else None,
            zones=[_build(Zone, z, "zones") for z in data.get("zones") or []],
            light_roi=_build(LightROI, roi, "light_roi") if roi else None,
            light_override=[_build(LightSegment, s, "light_override") for s in override] if override else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "reference_frame": self.reference_frame,
            "stop_line": self.stop_line.to_dict() if self.stop_line else None,
            "zones": [z.to_dict() for z in self.zones],
            "light_roi": self.light_roi.to_dict() if self.light_roi else None,
            "light_override": [s.to_dict() for s in self.light_override] if self.light_override else None,
        }

    @classmethod
    def load(cls, path: str | Path) -> "CameraConfig":
        """Read a camera config JSON file."""
        path = resolve_path(path)
        with path.open("r", encoding="utf-8") as f:
            cfg = cls.from_dict(json.load(f))
        log.info("Loaded camera config %s (camera_id=%s)", path, cfg.camera_id)
        return cfg

    def save(self, path: str | Path) -> Path:
        """Write the camera config as JSON and return the path."""
        path = resolve_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path
