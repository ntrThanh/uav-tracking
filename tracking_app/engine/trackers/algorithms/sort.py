"""SORT (Bewley et al., 2016): Kalman prediction and Hungarian matching on IoU."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..base import FrameContext, TrackerConfig, TrackingAlgorithm, track_rows
from ..core.association import iou_matrix, linear_assignment
from ..core.motion import XYSRMotion
from ..core.track import KalmanTrack, boxes_of
from ..registry import register


@dataclass(frozen=True)
class SortConfig(TrackerConfig):
    det_thresh: float = 0.4
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3


class SortTrack(KalmanTrack):
    def __init__(self, motion, det, track_id):
        super().__init__(motion, det, track_id)
        self.time_since_update = 0
        self.hit_streak = 0

    def predict(self) -> None:
        super().predict()
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

    def correct(self, det: np.ndarray) -> None:
        super().correct(det)
        self.time_since_update = 0
        self.hit_streak += 1


@register("sort")
class Sort(TrackingAlgorithm):
    config_cls = SortConfig

    def __init__(self, cfg: SortConfig):
        super().__init__(cfg)
        self.motion = XYSRMotion()
        self.tracks: list[SortTrack] = []
        self.frame_count = 0

    def step(self, dets: np.ndarray, ctx: FrameContext) -> np.ndarray:
        self.frame_count += 1
        dets = dets[dets[:, 4] >= self.cfg.det_thresh]
        for track in self.tracks:
            track.predict()

        cost = 1.0 - iou_matrix(boxes_of(self.tracks), dets[:, :4])
        matches, _, new = linear_assignment(cost, 1.0 - self.cfg.iou_threshold)
        for t, d in matches:
            self.tracks[t].correct(dets[d])
        self.tracks += [SortTrack(self.motion, dets[d], self._new_id()) for d in new]

        rows = [t.row() for t in self.tracks if self._visible(t)]
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.cfg.max_age]
        return track_rows(rows)

    def _visible(self, track: SortTrack) -> bool:
        warmed_up = track.hit_streak >= self.cfg.min_hits or self.frame_count <= self.cfg.min_hits
        return track.time_since_update == 0 and warmed_up
