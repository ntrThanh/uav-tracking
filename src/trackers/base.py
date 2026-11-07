"""Contracts shared by every tracking algorithm.

Array layouts used across the package:
    detections  (N, 6)  x1, y1, x2, y2, score, class
    tracks      (M, 7)  x1, y1, x2, y2, id, score, class

An algorithm only ever sees boxes of one class. Everything that must happen once per frame for all classes
(appearance features, camera motion) lives in FrontEnd, so the algorithms stay independent of each other.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, replace
from typing import ClassVar, Protocol, TypeVar, runtime_checkable

import numpy as np

DET_COLS = 6
TRACK_COLS = 7

C = TypeVar("C", bound="TrackerConfig")


@runtime_checkable
class AppearanceEncoder(Protocol):
    """Turns image crops into L2-normalised descriptors."""

    def encode(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """frame: BGR image, boxes: (N, 4) xyxy. Returns (N, D)."""
        ...


@runtime_checkable
class CameraMotionEstimator(Protocol):
    """Estimates the global image motion between two consecutive frames."""

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """Returns a (2, 3) affine matrix mapping the previous frame onto this one."""
        ...


@dataclass(frozen=True)
class TrackerConfig:
    """Parameters common to all algorithms. Subclasses add their own fields."""

    det_thresh: float = 0.5     # the "high" detection threshold tuned per detector
    frame_rate: int = 30

    @classmethod
    def from_dict(cls: type[C], values: dict) -> C:
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"{cls.__name__}: unknown parameters {sorted(unknown)}; expected {sorted(known)}")
        return cls(**values)

    def with_det_thresh(self: C, value: float) -> C:
        """Copy with another high threshold. Override when more than one field depends on it."""
        return replace(self, det_thresh=value)


@dataclass(frozen=True)
class FrameContext:
    """Per-frame information computed once and shared by all per-class trackers."""

    embeddings: np.ndarray | None = None    # (N, D), aligned with the detections of the frame
    warp: np.ndarray | None = None          # (2, 3) camera motion
    image: np.ndarray | None = None         # BGR frame, only for algorithms that ask for it (keep_image)

    def subset(self, mask: np.ndarray) -> "FrameContext":
        return replace(self, embeddings=None if self.embeddings is None else self.embeddings[mask])


class FrontEnd:
    """Computes the FrameContext. With no encoder and no estimator it needs no image at all."""

    def __init__(self, encoder: AppearanceEncoder | None = None, motion: CameraMotionEstimator | None = None,
                 embed_min_score: float = 0.0, keep_image: bool = False):
        self.encoder = encoder
        self.motion = motion
        self.embed_min_score = embed_min_score
        self.keep_image = keep_image

    @property
    def requires_image(self) -> bool:
        return self.encoder is not None or self.motion is not None or self.keep_image

    def analyze(self, dets: np.ndarray, frame: np.ndarray | None) -> FrameContext:
        if not self.requires_image:
            return FrameContext()
        if frame is None:
            raise ValueError("This tracker needs the image of every frame, got frame=None")
        warp = self.motion.estimate(frame) if self.motion is not None else None
        return FrameContext(embeddings=self._embed(dets, frame), warp=warp, image=frame if self.keep_image else None)

    def _embed(self, dets: np.ndarray, frame: np.ndarray) -> np.ndarray | None:
        if self.encoder is None:
            return None
        keep = dets[:, 4] > self.embed_min_score    # low-score boxes are never matched by appearance
        if not keep.any():
            return np.zeros((len(dets), 0), dtype=np.float32)
        features = np.asarray(self.encoder.encode(frame, dets[keep, :4]), dtype=np.float32)
        out = np.zeros((len(dets), features.shape[1]), dtype=np.float32)
        out[keep] = features
        return out


class TrackingAlgorithm(ABC):
    """One tracker instance follows one class of one sequence.

    With handles_classes=True a single instance receives all classes and keeps them apart itself; its rows must then
    carry the class in the last column.
    """

    config_cls: ClassVar[type[TrackerConfig]] = TrackerConfig
    handles_classes: ClassVar[bool] = False

    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self._last_id = 0

    def _new_id(self) -> int:
        self._last_id += 1
        return self._last_id

    @classmethod
    def needs_encoder(cls, cfg: TrackerConfig) -> bool:
        return False

    @classmethod
    def build_front_end(cls, cfg: TrackerConfig, encoder: AppearanceEncoder | None) -> FrontEnd:
        return FrontEnd()

    @abstractmethod
    def step(self, dets: np.ndarray, ctx: FrameContext) -> np.ndarray:
        """Advance one frame. dets is (N, 6) and may be empty; returns (M, 7)."""


def track_rows(rows: list[list[float]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float64).reshape(-1, TRACK_COLS)
