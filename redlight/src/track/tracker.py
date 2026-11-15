"""Multi-object tracking (ByteTrack via Ultralytics) with per-track reference-point history."""
from __future__ import annotations

import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from src.common.config import TrackerConfig
from src.common.geometry import Point, reference_point
from src.detect.vehicle import Detections

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackPoint:
    """Reference point (and box) of a track at one frame."""

    frame_index: int
    time_s: float
    x: float
    y: float
    xyxy: tuple[float, float, float, float]


@dataclass
class Track:
    """State of one tracked vehicle at the current frame."""

    track_id: int
    xyxy: np.ndarray  # (4,) float32, raw detection box
    conf: float
    class_id: int
    class_name: str

    @property
    def ref_point(self) -> Point:
        return reference_point(self.xyxy)


@dataclass
class TrackHistory:
    """Bounded history of one track id across frames."""

    track_id: int
    first_frame: int
    last_frame: int
    n_frames: int = 0
    class_votes: Counter = field(default_factory=Counter)
    points: deque = field(default_factory=deque)

    @property
    def class_name(self) -> str:
        """Majority class over the track's lifetime."""
        return self.class_votes.most_common(1)[0][0] if self.class_votes else "unknown"

    def last_two(self) -> tuple[TrackPoint, TrackPoint] | None:
        """Previous and current reference points, or None if the track is too young."""
        if len(self.points) < 2:
            return None
        return self.points[-2], self.points[-1]


class _DetectionView:
    """Minimal results-like view over :class:`Detections` for Ultralytics' BYTETracker."""

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return int(self.conf.shape[0])

    def __getitem__(self, mask: np.ndarray) -> "_DetectionView":
        return _DetectionView(self.xywh[mask], self.conf[mask], self.cls[mask])


class VehicleTracker:
    """ByteTrack wrapper that assigns stable ids and records reference points per id."""

    def __init__(self, cfg: TrackerConfig) -> None:
        from ultralytics.trackers.byte_tracker import BYTETracker  # lazy: heavy import

        self.cfg = cfg
        args = SimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=cfg.track_high_thresh,
            track_low_thresh=cfg.track_low_thresh,
            new_track_thresh=cfg.new_track_thresh,
            track_buffer=cfg.track_buffer,
            match_thresh=cfg.match_thresh,
            fuse_score=cfg.fuse_score,
        )
        self._tracker = BYTETracker(args)
        self._histories: dict[int, TrackHistory] = {}
        log.info("Tracker ByteTrack (high=%.2f low=%.2f new=%.2f buffer=%d match=%.2f)",
                 cfg.track_high_thresh, cfg.track_low_thresh, cfg.new_track_thresh,
                 cfg.track_buffer, cfg.match_thresh)

    def update(self, dets: Detections, frame_index: int, time_s: float) -> list[Track]:
        """Advance the tracker by one frame and return the confirmed tracks visible in it."""
        view = _DetectionView(dets.xywh, dets.conf, dets.class_id)
        out = self._tracker.update(view)  # (N, 8): x1, y1, x2, y2, track_id, score, cls, det_idx
        tracks: list[Track] = []
        for row in out:
            track_id, score, cls, idx = int(row[4]), float(row[5]), int(row[6]), int(row[7])
            if 0 <= idx < len(dets):
                xyxy, name = dets.xyxy[idx].copy(), dets.class_name[idx]
            else:  # defensive: should not happen, tracker rows always map to a detection
                xyxy, name = row[:4].astype(np.float32), str(cls)
            track = Track(track_id, xyxy, score, cls, name)
            self._record(track, frame_index, time_s)
            tracks.append(track)
        return tracks

    def _record(self, track: Track, frame_index: int, time_s: float) -> None:
        hist = self._histories.get(track.track_id)
        if hist is None:
            hist = TrackHistory(track.track_id, frame_index, frame_index)
            hist.points = deque(maxlen=self.cfg.history_len)
            self._histories[track.track_id] = hist
        hist.last_frame = frame_index
        hist.n_frames += 1
        hist.class_votes[track.class_name] += 1
        px, py = track.ref_point
        hist.points.append(TrackPoint(frame_index, time_s, px, py, tuple(float(v) for v in track.xyxy)))

    def history(self, track_id: int) -> TrackHistory | None:
        return self._histories.get(track_id)

    @property
    def histories(self) -> dict[int, TrackHistory]:
        return self._histories

    def reset(self) -> None:
        self._tracker.reset()
        self._histories.clear()

    def stats(self) -> dict:
        """Track-length statistics used to judge id stability."""
        lengths = np.array([h.n_frames for h in self._histories.values()], dtype=int)
        if lengths.size == 0:
            return {"n_tracks": 0}
        return {
            "n_tracks": int(lengths.size),
            "track_len_mean": round(float(lengths.mean()), 1),
            "track_len_median": float(np.median(lengths)),
            "track_len_min": int(lengths.min()),
            "track_len_max": int(lengths.max()),
            "n_tracks_shorter_than_5": int((lengths < 5).sum()),
            "tracks_by_class": dict(Counter(h.class_name for h in self._histories.values())),
        }
