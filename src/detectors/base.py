"""Common interface of the three fine-tuned detectors.

Every detector returns boxes in the original image, in the same layout as the tracker input:
    (N, 6)  x1, y1, x2, y2, score, class      class is the VisDrone id 1-10
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NUM_CLASSES = 10


class Detector(ABC):
    """One model on one device. Subclasses only load the model and run it on a single BGR image."""

    default_weights: str = ""

    def __init__(self, weights: str | None = None, device: str = "cuda:0", conf: float = 0.05):
        path = Path(weights or self.default_weights)
        self.weights = path if path.is_absolute() else ROOT / path
        if not self.weights.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.weights}")
        self.device = device
        self.conf = conf

    @abstractmethod
    def _predict(self, image: np.ndarray) -> np.ndarray:
        """image: BGR (H, W, 3). Returns (N, 6) with the class already mapped to 1-10."""

    def predict(self, image: np.ndarray) -> np.ndarray:
        dets = np.asarray(self._predict(image), dtype=np.float64).reshape(-1, 6)
        h, w = image.shape[:2]
        dets[:, [0, 2]] = dets[:, [0, 2]].clip(0, w)
        dets[:, [1, 3]] = dets[:, [1, 3]].clip(0, h)
        keep = ((dets[:, 2] > dets[:, 0]) & (dets[:, 3] > dets[:, 1]) & (dets[:, 4] >= self.conf)
                & (dets[:, 5] >= 1) & (dets[:, 5] <= NUM_CLASSES))
        return dets[keep]

    def settings(self) -> dict:
        """Inference parameters written next to the cache."""
        return {}

    def describe(self) -> dict:
        return {"weights": str(self.weights.relative_to(ROOT) if self.weights.is_relative_to(ROOT) else self.weights),
                "weights_sha256": sha256(self.weights), "conf": self.conf, **self.settings()}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 23), b""):
            h.update(chunk)
    return h.hexdigest()
