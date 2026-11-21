"""Constant-velocity Kalman filters behind one interface.

LinearKalman holds the filter equations. A concrete model only says how a box maps to a measurement and how large
the noise is, so the same track code runs on any of them:
    XYSRMotion  centre, area, aspect ratio   SORT, OC-SORT
    XYAHMotion  centre, aspect ratio, height DeepSORT, ByteTrack
    XYWHMotion  centre, width, height        BoT-SORT
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

Gaussian = tuple[np.ndarray, np.ndarray]


class LinearKalman(ABC):
    def __init__(self, transition: np.ndarray, observation: np.ndarray):
        self.F = transition
        self.H = observation

    @abstractmethod
    def encode(self, box: np.ndarray) -> np.ndarray:
        """xyxy box -> measurement."""

    @abstractmethod
    def decode(self, mean: np.ndarray) -> np.ndarray:
        """State -> xyxy box."""

    @abstractmethod
    def initiate(self, z: np.ndarray) -> Gaussian:
        """First state from one measurement."""

    @abstractmethod
    def process_noise(self, mean: np.ndarray) -> np.ndarray:
        """Q for this state."""

    @abstractmethod
    def measurement_noise(self, mean: np.ndarray) -> np.ndarray:
        """R for this state."""

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> Gaussian:
        q = self.process_noise(mean)
        return self.F @ mean, self.F @ cov @ self.F.T + q

    def project(self, mean: np.ndarray, cov: np.ndarray) -> Gaussian:
        return self.H @ mean, self.H @ cov @ self.H.T + self.measurement_noise(mean)

    def update(self, mean: np.ndarray, cov: np.ndarray, z: np.ndarray) -> Gaussian:
        projected, s = self.project(mean, cov)
        gain = np.linalg.solve(s, self.H @ cov).T
        return mean + gain @ (z - projected), cov - gain @ s @ gain.T

    def gating_distance(self, mean: np.ndarray, cov: np.ndarray, zs: np.ndarray) -> np.ndarray:
        """Squared Mahalanobis distance from the state to each measurement."""
        projected, s = self.project(mean, cov)
        chol = np.linalg.cholesky(s)
        whitened = np.linalg.solve(chol, (zs - projected).T)
        return np.sum(whitened ** 2, axis=0)


def _constant_velocity(position_dims: int, velocity_dims: int) -> tuple[np.ndarray, np.ndarray]:
    size = position_dims + velocity_dims
    transition = np.eye(size)
    transition[np.arange(velocity_dims), position_dims + np.arange(velocity_dims)] = 1.0
    return transition, np.eye(position_dims, size)


class XYSRMotion(LinearKalman):
    """SORT state: u, v, s, r and the velocities of u, v, s. Noise is fixed."""

    def __init__(self):
        super().__init__(*_constant_velocity(4, 3))

    def encode(self, box: np.ndarray) -> np.ndarray:
        w, h = box[2] - box[0], box[3] - box[1]
        return np.array([box[0] + w / 2, box[1] + h / 2, w * h, w / h])

    def decode(self, mean: np.ndarray) -> np.ndarray:
        area, ratio = max(mean[2], 1e-6), max(mean[3], 1e-6)
        w = np.sqrt(area * ratio)
        h = area / w
        return np.array([mean[0] - w / 2, mean[1] - h / 2, mean[0] + w / 2, mean[1] + h / 2])

    def initiate(self, z: np.ndarray) -> Gaussian:
        mean = np.zeros(7)
        mean[:4] = z
        return mean, np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])

    def process_noise(self, mean: np.ndarray) -> np.ndarray:
        return np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 1e-4])

    def measurement_noise(self, mean: np.ndarray) -> np.ndarray:
        return np.diag([1.0, 1.0, 10.0, 10.0])

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> Gaussian:
        if mean[2] + mean[6] <= 0:      # the area must not shrink below zero
            mean = mean.copy()
            mean[6] = 0.0
        return super().predict(mean, cov)


class _ScaledNoiseMotion(LinearKalman):
    """Eight-state filters whose noise grows with the box size."""

    POSITION_WEIGHT = 1.0 / 20
    VELOCITY_WEIGHT = 1.0 / 160

    def __init__(self):
        super().__init__(*_constant_velocity(4, 4))

    @abstractmethod
    def _position_std(self, values: np.ndarray, weight: float) -> np.ndarray:
        """Per-dimension standard deviation of the four measured values."""

    @abstractmethod
    def _velocity_std(self, values: np.ndarray, weight: float) -> np.ndarray:
        """Per-dimension standard deviation of the four velocities."""

    def _measure_std(self, values: np.ndarray) -> np.ndarray:
        return self._position_std(values, self.POSITION_WEIGHT)

    def initiate(self, z: np.ndarray) -> Gaussian:
        std = np.r_[self._position_std(z, 2 * self.POSITION_WEIGHT), self._velocity_std(z, 10 * self.VELOCITY_WEIGHT)]
        return np.r_[z, np.zeros(4)], np.diag(std ** 2)

    def process_noise(self, mean: np.ndarray) -> np.ndarray:
        std = np.r_[self._position_std(mean, self.POSITION_WEIGHT), self._velocity_std(mean, self.VELOCITY_WEIGHT)]
        return np.diag(std ** 2)

    def measurement_noise(self, mean: np.ndarray) -> np.ndarray:
        return np.diag(self._measure_std(mean) ** 2)


class XYAHMotion(_ScaledNoiseMotion):
    def encode(self, box: np.ndarray) -> np.ndarray:
        w, h = box[2] - box[0], box[3] - box[1]
        return np.array([box[0] + w / 2, box[1] + h / 2, w / h, h])

    def decode(self, mean: np.ndarray) -> np.ndarray:
        h = max(mean[3], 1e-6)
        w = max(mean[2], 1e-6) * h
        return np.array([mean[0] - w / 2, mean[1] - h / 2, mean[0] + w / 2, mean[1] + h / 2])

    def _position_std(self, values: np.ndarray, weight: float) -> np.ndarray:
        h = values[3]
        return np.array([weight * h, weight * h, 1e-2, weight * h])

    def _velocity_std(self, values: np.ndarray, weight: float) -> np.ndarray:
        h = values[3]
        return np.array([weight * h, weight * h, 1e-5, weight * h])

    def _measure_std(self, values: np.ndarray) -> np.ndarray:
        h = values[3]
        return np.array([self.POSITION_WEIGHT * h, self.POSITION_WEIGHT * h, 1e-1, self.POSITION_WEIGHT * h])


class XYWHMotion(_ScaledNoiseMotion):
    def encode(self, box: np.ndarray) -> np.ndarray:
        w, h = box[2] - box[0], box[3] - box[1]
        return np.array([box[0] + w / 2, box[1] + h / 2, w, h])

    def decode(self, mean: np.ndarray) -> np.ndarray:
        w, h = max(mean[2], 1e-6), max(mean[3], 1e-6)
        return np.array([mean[0] - w / 2, mean[1] - h / 2, mean[0] + w / 2, mean[1] + h / 2])

    def _position_std(self, values: np.ndarray, weight: float) -> np.ndarray:
        w, h = values[2], values[3]
        return np.array([weight * w, weight * h, weight * w, weight * h])

    def _velocity_std(self, values: np.ndarray, weight: float) -> np.ndarray:
        return self._position_std(values, weight)
