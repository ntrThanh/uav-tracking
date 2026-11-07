"""DeepSORT (Wojke et al., 2017): matching cascade on appearance with a Mahalanobis gate, then IoU."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from ..base import AppearanceEncoder, FrameContext, FrontEnd, TrackerConfig, TrackingAlgorithm, track_rows
from ..core.association import cosine_distance, iou_matrix, linear_assignment
from ..core.motion import XYAHMotion
from ..core.track import KalmanTrack, boxes_of
from ..registry import register

CHI2_95_4DOF = 9.4877       # gate for a four-dimensional measurement
INFEASIBLE = 1e5


@dataclass(frozen=True)
class DeepSortConfig(TrackerConfig):
    det_thresh: float = 0.4
    max_age: int = 30
    n_init: int = 3
    max_iou_distance: float = 0.7
    max_cosine_distance: float = 0.2
    nn_budget: int = 100
    reid_weights: str = "weights/pretrained/osnet_x0_25_msmt17.pt"


class State(Enum):
    TENTATIVE = auto()
    CONFIRMED = auto()
    DELETED = auto()


class DeepSortTrack(KalmanTrack):
    def __init__(self, motion, det, track_id, feature: np.ndarray, cfg: DeepSortConfig):
        super().__init__(motion, det, track_id)
        self.cfg = cfg
        self.state = State.TENTATIVE
        self.hits = 1
        self.time_since_update = 0
        self.gallery = [feature]

    def predict(self) -> None:
        super().predict()
        self.time_since_update += 1

    def correct_with(self, det: np.ndarray, feature: np.ndarray) -> None:
        self.correct(det)
        self.gallery = (self.gallery + [feature])[-self.cfg.nn_budget:]
        self.hits += 1
        self.time_since_update = 0
        if self.state is State.TENTATIVE and self.hits >= self.cfg.n_init:
            self.state = State.CONFIRMED

    def mark_missed(self) -> None:
        if self.state is State.TENTATIVE or self.time_since_update > self.cfg.max_age:
            self.state = State.DELETED

    def appearance_cost(self, features: np.ndarray) -> np.ndarray:
        """Nearest-neighbour distance from the gallery of this track to each feature."""
        return cosine_distance(np.asarray(self.gallery), features).min(axis=0)


@register("deepsort")
class DeepSort(TrackingAlgorithm):
    config_cls = DeepSortConfig

    def __init__(self, cfg: DeepSortConfig):
        super().__init__(cfg)
        self.motion = XYAHMotion()
        self.tracks: list[DeepSortTrack] = []

    @classmethod
    def needs_encoder(cls, cfg: DeepSortConfig) -> bool:
        return True

    @classmethod
    def build_front_end(cls, cfg: DeepSortConfig, encoder: AppearanceEncoder | None) -> FrontEnd:
        return FrontEnd(encoder=encoder, embed_min_score=np.nextafter(cfg.det_thresh, -1.0))

    def step(self, dets: np.ndarray, ctx: FrameContext) -> np.ndarray:
        keep = dets[:, 4] >= self.cfg.det_thresh
        dets, features = dets[keep], ctx.embeddings[keep]
        for track in self.tracks:
            track.predict()

        matches, unmatched_tracks, unmatched_dets = self._match(dets, features)
        for t, d in matches:
            self.tracks[t].correct_with(dets[d], features[d])
        for t in unmatched_tracks:
            self.tracks[t].mark_missed()
        self.tracks += [DeepSortTrack(self.motion, dets[d], self._new_id(), features[d], self.cfg)
                        for d in unmatched_dets]
        self.tracks = [t for t in self.tracks if t.state is not State.DELETED]
        return track_rows([t.row() for t in self.tracks
                           if t.state is State.CONFIRMED and t.time_since_update == 0])

    def _match(self, dets: np.ndarray, features: np.ndarray):
        confirmed = [i for i, t in enumerate(self.tracks) if t.state is State.CONFIRMED]
        tentative = [i for i, t in enumerate(self.tracks) if t.state is not State.CONFIRMED]
        matches, left_dets = self._cascade(confirmed, dets, features)

        matched = {t for t, _ in matches}
        left_confirmed = [i for i in confirmed if i not in matched]
        recent = [i for i in left_confirmed if self.tracks[i].time_since_update == 1]
        old = [i for i in left_confirmed if self.tracks[i].time_since_update != 1]
        iou_matches, left_tracks, left_dets = self._match_iou(tentative + recent, left_dets, dets)
        return matches + iou_matches, old + left_tracks, left_dets

    def _cascade(self, confirmed: list[int], dets: np.ndarray, features: np.ndarray):
        """Tracks seen most recently choose first."""
        matches, left = [], list(range(len(dets)))
        for age in range(1, self.cfg.max_age + 1):
            group = [i for i in confirmed if self.tracks[i].time_since_update == age]
            if not left:
                break
            if not group:
                continue
            found, _, free = linear_assignment(self._gated_cost(group, left, dets, features),
                                               self.cfg.max_cosine_distance)
            matches += [(group[t], left[d]) for t, d in found]
            left = [left[d] for d in free]
        return matches, left

    def _gated_cost(self, group: list[int], cols: list[int], dets: np.ndarray, features: np.ndarray) -> np.ndarray:
        measurements = np.array([self.motion.encode(dets[d, :4]) for d in cols])
        cost = np.zeros((len(group), len(cols)))
        for row, i in enumerate(group):
            track = self.tracks[i]
            cost[row] = track.appearance_cost(features[cols])
            gate = self.motion.gating_distance(track.mean, track.cov, measurements)
            cost[row, gate > CHI2_95_4DOF] = INFEASIBLE
        return cost

    def _match_iou(self, rows: list[int], cols: list[int], dets: np.ndarray):
        boxes = boxes_of([self.tracks[i] for i in rows])
        cost = 1.0 - iou_matrix(boxes, dets[cols, :4])
        found, left_rows, left_cols = linear_assignment(cost, self.cfg.max_iou_distance)
        return ([(rows[t], cols[d]) for t, d in found], [rows[t] for t in left_rows], [cols[d] for d in left_cols])
