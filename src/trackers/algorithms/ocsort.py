"""OC-SORT (Cao et al., 2023): SORT with three observation-centric additions.

    OCM  the direction of motion joins the association cost
    OCR  a second try with unmatched tracks at their last observed position
    ORU  after a gap, the Kalman filter is replayed along a straight line between the two observations
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..base import FrameContext, TrackerConfig, TrackingAlgorithm, track_rows
from ..core.association import iou_matrix, linear_assignment
from ..core.motion import XYSRMotion
from ..core.track import KalmanTrack, boxes_of
from ..registry import register

NO_BOX = np.full(4, -1.0)


@dataclass(frozen=True)
class OcSortConfig(TrackerConfig):
    det_thresh: float = 0.5
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    delta_t: int = 3
    inertia: float = 0.2
    asso_func: str = "iou"

    def __post_init__(self):
        if self.asso_func != "iou":
            raise ValueError(f"Only asso_func=iou is implemented, got {self.asso_func}")


def direction(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Unit vector (dy, dx) between two box centres."""
    delta = (end[:2] + end[2:4] - start[:2] - start[2:4]) / 2.0
    delta = delta[::-1]
    return delta / (np.linalg.norm(delta) + 1e-6)


class OcTrack(KalmanTrack):
    def __init__(self, motion, det, track_id, delta_t: int):
        super().__init__(motion, det, track_id)
        self.delta_t = delta_t
        self.age = 0
        self.time_since_update = 0
        self.hit_streak = 0
        self.observations: dict[int, np.ndarray] = {}
        self.last_box = NO_BOX
        self.velocity = np.zeros(2)
        self._anchor = (self.age, motion.encode(det[:4]))   # last real measurement
        self._frozen = None                                 # filter state just after the last real measurement

    def predict(self) -> None:
        super().predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

    def miss(self) -> None:
        if self._frozen is None:
            self._frozen = (self.mean.copy(), self.cov.copy())

    def observe(self, det: np.ndarray) -> None:
        box = det[:4]
        if self.observations:
            self.velocity = direction(self.reference_box(), box)
        self.last_box = box.copy()
        self.observations[self.age] = box.copy()
        self.time_since_update = 0
        self.hit_streak += 1
        z = self.motion.encode(box)
        if self._frozen is not None:
            self._replay(z)
        self.correct(det)
        self._anchor = (self.age, z)

    def reference_box(self) -> np.ndarray:
        """The observation delta_t frames ago, or the closest available one. Needs at least one observation."""
        for dt in range(self.delta_t, 0, -1):
            if self.age - dt in self.observations:
                return self.observations[self.age - dt]
        return self.observations[max(self.observations)]

    def _replay(self, z: np.ndarray) -> None:
        """ORU: restore the filter and feed it virtual measurements on the line from the anchor to z."""
        self.mean, self.cov = self._frozen
        self._frozen = None
        anchor_age, start = self._anchor
        gap = self.age - anchor_age
        size_start, size_end = self._size(start), self._size(z)
        for i in range(1, gap):
            ratio = i / gap
            cx, cy = start[:2] + ratio * (z[:2] - start[:2])
            w, h = size_start + ratio * (size_end - size_start)
            virtual = np.array([cx, cy, w * h, w / h])
            self.mean, self.cov = self.motion.update(self.mean, self.cov, virtual)
            self.mean, self.cov = self.motion.predict(self.mean, self.cov)

    @staticmethod
    def _size(z: np.ndarray) -> np.ndarray:
        """Width and height from area and aspect ratio."""
        w = np.sqrt(z[2] * z[3])
        return np.array([w, z[2] / w])


@register("ocsort")
class OcSort(TrackingAlgorithm):
    config_cls = OcSortConfig

    def __init__(self, cfg: OcSortConfig):
        super().__init__(cfg)
        self.motion = XYSRMotion()
        self.tracks: list[OcTrack] = []
        self.frame_count = 0

    def step(self, dets: np.ndarray, ctx: FrameContext) -> np.ndarray:
        self.frame_count += 1
        dets = dets[dets[:, 4] > self.cfg.det_thresh]
        for track in self.tracks:
            track.predict()

        matches, left_tracks, left_dets = self._associate(dets)
        for t, d in matches:
            self.tracks[t].observe(dets[d])
        left_tracks, left_dets = self._recover(dets, left_tracks, left_dets)
        for t in left_tracks:
            self.tracks[t].miss()
        self.tracks += [OcTrack(self.motion, dets[d], self._new_id(), self.cfg.delta_t) for d in left_dets]

        rows = [self._output(t) for t in self.tracks if self._visible(t)]
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.cfg.max_age]
        return track_rows(rows)

    def _associate(self, dets: np.ndarray):
        """OCM: IoU plus the agreement between track direction and track-to-detection direction."""
        iou = iou_matrix(boxes_of(self.tracks), dets[:, :4])
        if iou.size == 0:
            return [], list(range(len(self.tracks))), list(range(len(dets)))
        overlapping = iou > self.cfg.iou_threshold
        if overlapping.sum(axis=0).max() <= 1 and overlapping.sum(axis=1).max() <= 1:
            found = [(int(t), int(d)) for t, d in zip(*np.nonzero(overlapping))]    # no conflict to resolve
            left_tracks = [t for t in range(len(self.tracks)) if t not in {m[0] for m in found}]
            left_dets = [d for d in range(len(dets)) if d not in {m[1] for m in found}]
            return found, left_tracks, left_dets
        cost = -(iou + self._direction_score(dets))
        found, left_tracks, left_dets = linear_assignment(cost, np.inf)
        matches = [(t, d) for t, d in found if iou[t, d] >= self.cfg.iou_threshold]
        rejected = [m for m in found if iou[m] < self.cfg.iou_threshold]
        return matches, left_tracks + [t for t, _ in rejected], left_dets + [d for _, d in rejected]

    def _direction_score(self, dets: np.ndarray) -> np.ndarray:
        references = np.array([t.reference_box() if t.observations else NO_BOX for t in self.tracks])
        velocities = np.array([t.velocity for t in self.tracks])
        ref_centres = (references[:, :2] + references[:, 2:]) / 2.0
        det_centres = (dets[:, :2] + dets[:, 2:4]) / 2.0
        delta = det_centres[None, :, :] - ref_centres[:, None, :]
        delta /= np.linalg.norm(delta, axis=2, keepdims=True) + 1e-6
        cos = np.clip(velocities[:, None, 1] * delta[..., 0] + velocities[:, None, 0] * delta[..., 1], -1.0, 1.0)
        score = (np.pi / 2 - np.abs(np.arccos(cos))) / np.pi
        valid = np.array([bool(t.observations) for t in self.tracks])[:, None]
        return score * valid * self.cfg.inertia * dets[None, :, 4]

    def _recover(self, dets: np.ndarray, left_tracks: list[int], left_dets: list[int]):
        """OCR: match what is left against the last observation of each track."""
        if not left_tracks or not left_dets:
            return left_tracks, left_dets
        last = np.array([self.tracks[t].last_box for t in left_tracks])
        iou = iou_matrix(last, dets[left_dets, :4])
        found, _, _ = linear_assignment(-iou, -self.cfg.iou_threshold)
        for t, d in found:
            self.tracks[left_tracks[t]].observe(dets[left_dets[d]])
        used_t = {t for t, _ in found}
        used_d = {d for _, d in found}
        return ([t for i, t in enumerate(left_tracks) if i not in used_t],
                [d for i, d in enumerate(left_dets) if i not in used_d])

    def _visible(self, track: OcTrack) -> bool:
        warmed_up = track.hit_streak >= self.cfg.min_hits or self.frame_count <= self.cfg.min_hits
        return track.time_since_update == 0 and warmed_up

    @staticmethod
    def _output(track: OcTrack) -> list[float]:
        return track.row(track.last_box if track.observations else None)
