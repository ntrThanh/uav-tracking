"""OSNet appearance encoder, shared by DeepSORT and BoT-SORT.

The network and its weights come from boxmot. Only this file imports it, and only the model is used: all tracking
logic is in this package. The algorithms depend on the AppearanceEncoder protocol, not on this class.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class OsnetEncoder:
    def __init__(self, weights: str, device: str | None = None):
        import torch
        from boxmot import ReIDConfig
        from boxmot.reid.factory import create_reid_encoder

        path = Path(weights) if Path(weights).is_absolute() else ROOT / weights
        if not path.is_file():
            raise FileNotFoundError(f"ReID weights not found: {path}")
        device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self._encoder = create_reid_encoder(ReIDConfig(path, device=device, allow_download=False))

    def encode(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        import torch
        from boxmot.structures import Boxes, Detections, Frame

        n = len(boxes)
        rgb_chw = np.ascontiguousarray(frame.transpose(2, 0, 1)[::-1])
        image = Frame(image=torch.from_numpy(rgb_chw), sample_id="frame")
        crops = Detections(geometry=Boxes(torch.from_numpy(np.ascontiguousarray(boxes, dtype=np.float32))),
                           scores=torch.ones(n), class_ids=torch.zeros(n, dtype=torch.int64), sample_id="frame")
        features = self._encoder.encode((image,), (crops,))[0].numpy()
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return features / np.maximum(norms, 1e-12)
