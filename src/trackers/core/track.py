"""State common to every track: identity, last detection and a Kalman filter."""
from __future__ import annotations

import numpy as np

from .motion import LinearKalman


class KalmanTrack:
    def __init__(self, motion: LinearKalman, det: np.ndarray, track_id: int):
        self.motion = motion
        self.track_id = track_id
        self.score = float(det[4])
        self.mean, self.cov = motion.initiate(motion.encode(det[:4]))

    @property
    def box(self) -> np.ndarray:
        return self.motion.decode(self.mean)

    def predict(self) -> None:
        self.mean, self.cov = self.motion.predict(self.mean, self.cov)

    def correct(self, det: np.ndarray) -> None:
        self.mean, self.cov = self.motion.update(self.mean, self.cov, self.motion.encode(det[:4]))
        self.score = float(det[4])

    def row(self, box: np.ndarray | None = None) -> list[float]:
        """One output line. The class column is filled in by Tracker."""
        x1, y1, x2, y2 = self.box if box is None else box
        return [x1, y1, x2, y2, self.track_id, self.score, -1]


def boxes_of(tracks: list[KalmanTrack]) -> np.ndarray:
    return np.array([t.box for t in tracks], dtype=np.float64).reshape(-1, 4)
