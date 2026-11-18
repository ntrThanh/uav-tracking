"""Plate detection with an optional OCR adapter.

The plate detector ships in weights/; text reading requires installing the separate
PaddleOCR runtime. Results keep detection and reading status separate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.common.config import PROJECT_ROOT


@dataclass
class PlateResult:
    status: str
    text: str | None = None
    confidence: float | None = None
    crop: np.ndarray | None = None


class PlateRecognizer:
    def __init__(self) -> None:
        from ultralytics import YOLO
        weights = PROJECT_ROOT / "weights/plate_yolo11s.pt"
        if not weights.is_file():
            raise FileNotFoundError(f"Thiếu mô hình biển số: {weights}")
        self.detector = YOLO(str(weights))
        self.ocr: Any = None
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
        except ImportError:
            pass

    def recognize(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> PlateResult:
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return PlateResult("invalid_vehicle_crop")
        vehicle = frame[y0:y1, x0:x1]
        results = self.detector.predict(vehicle, conf=0.25, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return PlateResult("plate_not_found")
        index = int(boxes.conf.argmax().item())
        px0, py0, px1, py1 = [int(v) for v in boxes.xyxy[index].tolist()]
        crop = vehicle[max(0, py0):min(vehicle.shape[0], py1), max(0, px0):min(vehicle.shape[1], px1)]
        if crop.size == 0:
            return PlateResult("invalid_plate_crop")
        confidence = float(boxes.conf[index].item())
        if self.ocr is None:
            return PlateResult("ocr_unavailable", confidence=confidence, crop=crop)
        try:
            output = self.ocr.ocr(crop, cls=False)
            pieces = []
            for line in output or []:
                for detection in line or []:
                    if isinstance(detection, (list, tuple)) and len(detection) > 1:
                        value = detection[1]
                        if isinstance(value, (list, tuple)) and value:
                            pieces.append(str(value[0]))
            text = re.sub(r"[^A-Z0-9]", "", "".join(pieces).upper())
            return PlateResult("read" if text else "unreadable", text or None, confidence, crop)
        except Exception:
            return PlateResult("ocr_error", confidence=confidence, crop=crop)
