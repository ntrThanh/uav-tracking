"""T3-det: RF-DETR Medium 576 fine-tuned on VisDrone2019-DET. Model classes 0-9 are VisDrone 1-10, no NMS,
at most 300 boxes per image (num_select)."""
from __future__ import annotations

import numpy as np

from .base import Detector, NUM_CLASSES


class RFDETRMedium(Detector):
    default_weights = "runs/det/T3det_rfdetr_m_576_det/checkpoint_best_total.pth"

    def __init__(self, weights=None, device="cuda:0", conf=0.05, resolution=576):
        super().__init__(weights, device, conf)
        import rfdetr
        from PIL import Image
        self.resolution = resolution
        self.model = rfdetr.RFDETRMedium(pretrain_weights=str(self.weights), num_classes=NUM_CLASSES,
                                         resolution=resolution, device=device)
        self._image = Image

    def _predict(self, image: np.ndarray) -> np.ndarray:
        det = self.model.predict(self._image.fromarray(np.ascontiguousarray(image[:, :, ::-1])), threshold=self.conf)
        return np.c_[det.xyxy, det.confidence, det.class_id + 1]

    def settings(self) -> dict:
        return {"resolution": self.resolution, "nms": None, "max_det": 300, "precision": "fp32"}
