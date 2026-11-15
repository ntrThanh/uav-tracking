"""Vehicle detection with an Ultralytics YOLO model (COCO vehicle classes by default)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.common.config import DetectorConfig, resolve_path

log = logging.getLogger(__name__)


@dataclass
class Detections:
    """Per-frame detections in pixel coordinates."""

    xyxy: np.ndarray  # (N, 4) float32
    conf: np.ndarray  # (N,) float32
    class_id: np.ndarray  # (N,) int32, model class ids
    class_name: list[str]  # (N,) human-readable names from the config

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])

    @classmethod
    def empty(cls) -> "Detections":
        return cls(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int32), [])

    @property
    def xywh(self) -> np.ndarray:
        """Boxes as (centre x, centre y, width, height)."""
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1).astype(np.float32)


def resolve_device(device: str) -> str:
    """Map ``"auto"`` to ``cuda:0`` when available, else ``cpu``; pass other values through."""
    if device != "auto":
        return device
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - torch missing
        return "cpu"


class VehicleDetector:
    """Thin wrapper around ``ultralytics.YOLO`` that keeps only the configured vehicle classes."""

    def __init__(self, cfg: DetectorConfig) -> None:
        from ultralytics import YOLO  # imported lazily: slow, and tests may not need it

        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        weights = resolve_path(cfg.weights)
        if not weights.exists():
            log.warning("Weights %s not found; Ultralytics will try to download them", weights)
        self.model = YOLO(str(weights))
        model_names: dict[int, str] = self.model.names
        missing = [c for c in cfg.classes if c not in model_names]
        if missing:
            raise ValueError(f"Configured class ids {missing} are not in the model's classes {model_names}")
        self.class_ids = sorted(cfg.classes)
        # Only pass FP16 when requested: Ultralytics >= 8.4 warns on every call otherwise.
        self._precision_kwargs: dict = {"half": True} if cfg.half and self.device != "cpu" else {}
        log.info(
            "Detector %s on %s (imgsz=%d, conf=%.2f, iou=%.2f, agnostic_nms=%s, classes=%s)",
            weights.name, self.device, cfg.imgsz, cfg.conf, cfg.iou, cfg.agnostic_nms, cfg.classes,
        )

    def detect(self, frame_bgr: np.ndarray) -> Detections:
        """Run the model on one BGR frame and return vehicle detections."""
        results = self.model.predict(
            frame_bgr,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            agnostic_nms=self.cfg.agnostic_nms,
            classes=self.class_ids,
            device=self.device,
            verbose=False,
            **self._precision_kwargs,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return Detections.empty()
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = boxes.conf.cpu().numpy().astype(np.float32)
        cls = boxes.cls.cpu().numpy().astype(np.int32)
        names = [self.cfg.classes.get(int(c), str(int(c))) for c in cls]
        return Detections(xyxy, conf, cls, names)
