"""Camera motion estimation used by BoT-SORT."""
from __future__ import annotations

import cv2
import numpy as np

IDENTITY = np.eye(2, 3)


class SparseOpticalFlow:
    """Tracks corners between frames and fits a similarity transform with RANSAC ("sof" in BoT-SORT)."""

    def __init__(self, downscale: int = 2):
        self.downscale = max(int(downscale), 1)
        self._previous: np.ndarray | None = None

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.downscale > 1:
            h, w = gray.shape
            gray = cv2.resize(gray, (w // self.downscale, h // self.downscale))
        return gray

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        gray = self._prepare(frame)
        previous, self._previous = self._previous, gray
        if previous is None or previous.shape != gray.shape:
            return IDENTITY.copy()

        corners = cv2.goodFeaturesToTrack(previous, maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3)
        if corners is None:
            return IDENTITY.copy()
        moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, corners, None)
        ok = status.ravel() == 1
        if ok.sum() < 5:
            return IDENTITY.copy()
        matrix, _ = cv2.estimateAffinePartial2D(corners[ok], moved[ok], method=cv2.RANSAC)
        if matrix is None:
            return IDENTITY.copy()
        matrix = matrix.astype(np.float64)
        matrix[:, 2] *= self.downscale
        return matrix


def warp_gaussian(mean: np.ndarray, cov: np.ndarray, warp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Move an eight-state Kalman state by the camera motion: each (x, y)-like pair is rotated, the centre shifted."""
    rotation = np.kron(np.eye(4), warp[:, :2])
    mean = rotation @ mean
    mean[:2] += warp[:, 2]
    return mean, rotation @ cov @ rotation.T
