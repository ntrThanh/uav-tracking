"""Video ingestion: sequential frame reading with timestamps and reference-frame extraction."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

log = logging.getLogger(__name__)

DEFAULT_FPS = 25.0


@dataclass(frozen=True)
class VideoInfo:
    """Static properties of a video file."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


@dataclass(frozen=True)
class Frame:
    """One decoded frame; ``index`` is the position in the original file."""

    index: int
    time_s: float
    image: np.ndarray  # BGR, HxWx3


def _open(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    return cap


def _info_from_capture(cap: cv2.VideoCapture, path: Path) -> VideoInfo:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(fps) or fps <= 0:
        log.warning("Video %s reports fps=%s; falling back to %.1f", path.name, fps, DEFAULT_FPS)
        fps = DEFAULT_FPS
    return VideoInfo(
        path=path,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=fps,
        frame_count=max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0),
    )


def probe_video(path: str | Path) -> VideoInfo:
    """Return size, fps and frame count without decoding frames."""
    path = Path(path)
    cap = _open(path)
    try:
        return _info_from_capture(cap, path)
    finally:
        cap.release()


def read_frame_at(path: str | Path, index: int) -> np.ndarray:
    """Seek to ``index`` and return that frame (BGR)."""
    path = Path(path)
    cap = _open(path)
    try:
        if index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, img = cap.read()
        if not ok or img is None:
            raise IndexError(f"Cannot read frame {index} from {path}")
        return img
    finally:
        cap.release()


class VideoReader:
    """Iterate over the frames of a video file as :class:`Frame` objects.

    Frames are decoded sequentially (seeking only once, to ``start_s``). ``end_s`` is
    exclusive and ``stride`` keeps every n-th frame. ``Frame.index`` and ``Frame.time_s``
    always refer to the original file, independent of the window and stride.
    """

    def __init__(self, path: str | Path, start_s: float = 0.0, end_s: float | None = None, stride: int = 1) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if end_s is not None and end_s <= start_s:
            raise ValueError("end_s must be greater than start_s")
        self.path = Path(path)
        self.stride = int(stride)
        self._cap: cv2.VideoCapture | None = _open(self.path)
        self.info = _info_from_capture(self._cap, self.path)
        self.first_index = int(round(start_s * self.info.fps))
        self.end_index: int | None = int(round(end_s * self.info.fps)) if end_s is not None else None

    @property
    def expected_frames(self) -> int | None:
        """Number of frames the iterator should yield, or None if the count is unknown."""
        if self.info.frame_count <= 0:
            return None
        end = self.info.frame_count if self.end_index is None else min(self.end_index, self.info.frame_count)
        return max(0, math.ceil((end - self.first_index) / self.stride))

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __iter__(self) -> Iterator[Frame]:
        if self._cap is None:
            raise RuntimeError("VideoReader is closed")
        cap = self._cap
        if self.first_index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.first_index)
        index = self.first_index
        while self.end_index is None or index < self.end_index:
            ok, img = cap.read()
            if not ok or img is None:
                break
            if (index - self.first_index) % self.stride == 0:
                yield Frame(index=index, time_s=index / self.info.fps, image=img)
            index += 1


def extract_reference_frame(path: str | Path, time_s: float = 0.0) -> np.ndarray:
    """Return the frame at ``time_s`` (BGR); used as the canvas for drawing lines and zones."""
    info = probe_video(path)
    return read_frame_at(path, int(round(time_s * info.fps)))


def save_reference_frame(path: str | Path, out_path: str | Path, time_s: float = 0.0) -> Path:
    """Extract the reference frame and write it as an image file."""
    img = extract_reference_frame(path, time_s)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), img):
        raise OSError(f"Cannot write reference frame to {out_path}")
    return out_path
