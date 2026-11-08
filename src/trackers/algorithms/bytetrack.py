"""ByteTrack (Zhang et al., 2022): associate high-score boxes first, then recover lost matches with low-score boxes.

step() is a template method. BoT-SORT reuses it and only replaces the hooks marked below.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, auto

import numpy as np

from ..base import FrameContext, TrackerConfig, TrackingAlgorithm, track_rows
from ..core.association import fuse_score, iou_matrix, linear_assignment
from ..core.motion import LinearKalman, XYAHMotion
from ..core.track import KalmanTrack, boxes_of
from ..registry import register

SECOND_MATCH_THRESH = 0.5
UNCONFIRMED_MATCH_THRESH = 0.7
DUPLICATE_IOU_DISTANCE = 0.15
MAX_NEW_TRACK_THRESH = 0.95     # detector scores stay below 1, so a higher bar would never start a track


@dataclass(frozen=True)
class ByteTrackConfig(TrackerConfig):
    det_thresh: float = 0.5         # track_high_thresh
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    match_thresh: float = 0.8
    track_buffer: int = 30

    def with_det_thresh(self, value: float) -> "ByteTrackConfig":
        """new_track_thresh keeps its distance to the high threshold, capped at MAX_NEW_TRACK_THRESH."""
        margin = self.new_track_thresh - self.det_thresh
        new_track = max(value, min(value + margin, MAX_NEW_TRACK_THRESH))
        return replace(self, det_thresh=value, new_track_thresh=new_track)


class State(Enum):
    TRACKED = auto()
    LOST = auto()
    REMOVED = auto()


class ByteTrackItem(KalmanTrack):
    FROZEN_VELOCITIES: tuple[int, ...] = (7,)   # velocity dims reset while a track is not observed

    def __init__(self, motion: LinearKalman, det: np.ndarray, track_id: int, frame_id: int):
        super().__init__(motion, det, track_id)
        self.state = State.TRACKED
        self.activated = frame_id == 1
        self.start_frame = frame_id
        self.last_frame = frame_id

    def predict(self) -> None:
        if self.state is not State.TRACKED:
            self.mean = self.mean.copy()
            self.mean[list(self.FROZEN_VELOCITIES)] = 0.0
        super().predict()

    def observe(self, det: np.ndarray, frame_id: int) -> None:
        self.correct(det)
        self.state = State.TRACKED
        self.activated = True
        self.last_frame = frame_id


@register("bytetrack")
class ByteTrack(TrackingAlgorithm):
    config_cls = ByteTrackConfig
    motion_cls: type[LinearKalman] = XYAHMotion

    def __init__(self, cfg: ByteTrackConfig):
        super().__init__(cfg)
        self.motion = self.motion_cls()
        self.tracked: list[ByteTrackItem] = []
        self.lost: list[ByteTrackItem] = []
        self.frame_id = 0
        self.max_lost = int(cfg.frame_rate / 30.0 * cfg.track_buffer)

    # ------------------------------------------------------------- hooks
    def _create(self, det: np.ndarray, feature: np.ndarray | None) -> ByteTrackItem:
        return ByteTrackItem(self.motion, det, self._new_id(), self.frame_id)

    def _observe(self, track: ByteTrackItem, det: np.ndarray, feature: np.ndarray | None) -> None:
        track.observe(det, self.frame_id)

    def _compensate(self, tracks: list[ByteTrackItem], ctx: FrameContext) -> None:
        """Camera motion correction, nothing to do here."""

    def _high_cost(self, tracks: list[ByteTrackItem], dets: np.ndarray, features: np.ndarray | None) -> np.ndarray:
        """Cost of the first stage and of the unconfirmed stage."""
        return fuse_score(1.0 - iou_matrix(boxes_of(tracks), dets[:, :4]), dets[:, 4])

    # ------------------------------------------------------------- algorithm
    def step(self, dets: np.ndarray, ctx: FrameContext) -> np.ndarray:
        self.frame_id += 1
        scores = dets[:, 4]
        is_high = scores > self.cfg.det_thresh
        is_low = (scores > self.cfg.track_low_thresh) & (scores < self.cfg.det_thresh)
        high, low = dets[is_high], dets[is_low]
        features = None if ctx.embeddings is None else ctx.embeddings[is_high]

        confirmed = [t for t in self.tracked if t.activated]
        unconfirmed = [t for t in self.tracked if not t.activated]
        pool = confirmed + self.lost
        for track in pool:
            track.predict()
        self._compensate(pool + unconfirmed, ctx)

        left_tracks, left_high = self._match_high(pool, high, features)
        self._match_low(left_tracks, low)
        left_high = self._match_unconfirmed(unconfirmed, high, features, left_high)
        self._start_tracks(high, features, left_high)
        self._refresh(pool + unconfirmed)
        return track_rows([t.row() for t in self.tracked if t.activated])

    def _match_high(self, pool, high, features):
        cost = self._high_cost(pool, high, features)
        matches, left_tracks, left_dets = linear_assignment(cost, self.cfg.match_thresh)
        for t, d in matches:
            self._observe(pool[t], high[d], self._feature(features, d))
        return [pool[t] for t in left_tracks], left_dets

    def _match_low(self, left_tracks, low) -> None:
        candidates = [t for t in left_tracks if t.state is State.TRACKED]
        cost = 1.0 - iou_matrix(boxes_of(candidates), low[:, :4])
        matches, missed, _ = linear_assignment(cost, SECOND_MATCH_THRESH)
        for t, d in matches:
            self._observe(candidates[t], low[d], None)
        for t in missed:
            candidates[t].state = State.LOST

    def _match_unconfirmed(self, unconfirmed, high, features, left_high: list[int]) -> list[int]:
        rest = None if features is None else features[left_high]
        cost = self._high_cost(unconfirmed, high[left_high], rest)
        matches, dropped, free = linear_assignment(cost, UNCONFIRMED_MATCH_THRESH)
        for t, d in matches:
            self._observe(unconfirmed[t], high[left_high[d]], self._feature(rest, d))
        for t in dropped:
            unconfirmed[t].state = State.REMOVED
        return [left_high[d] for d in free]

    def _start_tracks(self, high, features, left_high: list[int]) -> None:
        for d in left_high:
            if high[d, 4] >= self.cfg.new_track_thresh:
                self.tracked.append(self._create(high[d], self._feature(features, d)))

    def _refresh(self, seen: list[ByteTrackItem]) -> None:
        for track in self.lost:
            if self.frame_id - track.last_frame > self.max_lost:
                track.state = State.REMOVED
        everyone = dict.fromkeys(seen + self.tracked)   # keeps order, drops repeats
        tracked = [t for t in everyone if t.state is State.TRACKED]
        lost = [t for t in everyone if t.state is State.LOST]
        self.tracked, self.lost = self._remove_duplicates(tracked, lost)

    def _remove_duplicates(self, tracked, lost):
        """When an active and a lost track sit on the same box, keep the one that lived longer."""
        close = (1.0 - iou_matrix(boxes_of(tracked), boxes_of(lost))) < DUPLICATE_IOU_DISTANCE
        drop_tracked, drop_lost = set(), set()
        for p, q in zip(*np.nonzero(close)):
            age_p = tracked[p].last_frame - tracked[p].start_frame
            age_q = lost[q].last_frame - lost[q].start_frame
            (drop_lost.add(q) if age_p > age_q else drop_tracked.add(p))
        return ([t for i, t in enumerate(tracked) if i not in drop_tracked],
                [t for i, t in enumerate(lost) if i not in drop_lost])

    @staticmethod
    def _feature(features: np.ndarray | None, index: int) -> np.ndarray | None:
        return None if features is None else features[index]
