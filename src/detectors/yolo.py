"""T1: YOLO11m fine-tuned on VisDrone2019-DET (Ultralytics). Model classes 0-9 are VisDrone 1-10."""
from __future__ import annotations

import numpy as np

from .base import Detector

VISDRONE_NAMES = ["pedestrian", "people", "bicycle", "car", "van",
                  "truck", "tricycle", "awning-tricycle", "bus", "motor"]


class Yolo11m(Detector):
    default_weights = "weights/turning/T1_yolo11m_640_det/best_yolo11m.pt"

    def __init__(self, weights=None, device="cuda:0", conf=0.05, imgsz=640, iou=0.7, max_det=1000):
        super().__init__(weights, device, conf)
        from ultralytics import YOLO
        self.imgsz, self.iou, self.max_det = imgsz, iou, max_det
        self.model = YOLO(str(self.weights))
        names = [str(self.model.names[i]).lower().strip() for i in range(len(self.model.names))]
        if names != VISDRONE_NAMES:
            raise ValueError(f"{self.weights.name}: class names {names} do not match VisDrone order")

    def _predict(self, image: np.ndarray) -> np.ndarray:
        r = self.model.predict(image, imgsz=self.imgsz, conf=self.conf, iou=self.iou, max_det=self.max_det,
                               device=self.device, verbose=False)[0]
        b = r.boxes
        return np.c_[b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy() + 1]

    def settings(self) -> dict:
        return {"imgsz": self.imgsz, "nms_iou": self.iou, "max_det": self.max_det, "precision": "fp32"}
