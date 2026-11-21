"""ONNX detector lifecycle for the tracking desktop app."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any

import cv2
import numpy as np

from .models import ModelRegistry
from .runtime import RuntimeChoice, open_onnx
from .schema import Detection, DetectionResult, Frame, ModelMetadata


class BaseDetector(ABC):
    def __init__(self, model: ModelMetadata, registry: ModelRegistry, device: str = "auto",
                 fallback: tuple[str, ...] = ("cpu",)) -> None:
        self.model = model
        self.registry = registry
        self.device = device
        self.fallback = fallback
        self.runtime_choice: RuntimeChoice | None = None
        self.loaded = False
        self.load_ms = 0.0

    @abstractmethod
    def load(self) -> None: ...

    def preprocess(self, frame: Frame) -> Any:
        return frame.image

    @abstractmethod
    def predict(self, input_data: Any) -> Any: ...

    @abstractmethod
    def postprocess(self, raw: Any, frame: Frame) -> DetectionResult: ...

    def detect(self, frame: Frame) -> DetectionResult:
        if not self.loaded:
            load_start = time.perf_counter()
            self.load()
            self.load_ms = (time.perf_counter() - load_start) * 1000
        start = time.perf_counter()
        result = self.postprocess(self.predict(self.preprocess(frame)), frame)
        return replace(result, latency_ms=(time.perf_counter() - start) * 1000)

    def close(self) -> None:
        self.loaded = False


class OnnxDetector(BaseDetector):
    """ONNX Runtime for decoded Nx6 output or Ultralytics YOLO head output."""

    def load(self) -> None:
        if self.model.runtime != "onnxruntime":
            raise ValueError("ONNX adapter requires onnxruntime")
        self.session, self.runtime_choice = open_onnx(str(self.registry.path(self.model.model_id)),
                                                      self.device, self.fallback)
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError("ONNX adapter requires exactly one image input")
        self.input_name = inputs[0].name
        self.loaded = True

    def preprocess(self, frame: Frame) -> np.ndarray:
        height, width = self.model.input_size
        image = frame.image
        scale = min(width / image.shape[1], height / image.shape[0])
        scaled_w, scaled_h = round(image.shape[1] * scale), round(image.shape[0] * scale)
        resized = cv2.resize(image, (scaled_w, scaled_h))
        pad_x, pad_y = (width - scaled_w) // 2, (height - scaled_h) // 2
        canvas = np.full((height, width, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + scaled_h, pad_x:pad_x + scaled_w] = resized
        self._transform = (scale, pad_x, pad_y)
        return np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0

    def predict(self, input_data: np.ndarray) -> list[np.ndarray]:
        return self.session.run(None, {self.input_name: input_data})

    def _original(self, box: np.ndarray, frame: Frame) -> tuple[float, float, float, float]:
        scale, pad_x, pad_y = self._transform
        x1, y1, x2, y2 = box
        return (float(np.clip((x1 - pad_x) / scale, 0, frame.meta.original_width)),
                float(np.clip((y1 - pad_y) / scale, 0, frame.meta.original_height)),
                float(np.clip((x2 - pad_x) / scale, 0, frame.meta.original_width)),
                float(np.clip((y2 - pad_y) / scale, 0, frame.meta.original_height)))

    def postprocess(self, raw: list[np.ndarray], frame: Frame) -> DetectionResult:
        if not raw:
            raise ValueError("ONNX model returned no outputs")
        format_name = self.model.postprocess.get("format", "decoded_nx6")
        threshold = float(self.model.postprocess.get("conf", 0.25))
        detections = []
        if format_name == "decoded_nx6":
            rows = np.asarray(raw[0]).reshape(-1, 6)
            for x1, y1, x2, y2, conf, cls in rows:
                if conf < threshold:
                    continue
                class_id = int(cls)
                if class_id != cls or class_id not in self.model.classes:
                    raise ValueError(f"unmapped ONNX class: {cls}")
                box = self._original(np.array([x1, y1, x2, y2]), frame)
                if box[2] > box[0] and box[3] > box[1]:
                    detections.append(Detection(box, float(conf), class_id,
                                                self.model.classes[class_id], self.model.model_id))
        elif format_name == "ultralytics_yolo":
            output = np.asarray(raw[0])
            if output.ndim != 3 or output.shape[0] != 1:
                raise ValueError("expected YOLO output (1,4+classes,N)")
            output = output[0].T
            if output.shape[1] != 4 + len(self.model.classes):
                raise ValueError("YOLO output class count differs from model metadata")
            native_ids = sorted(self.model.classes)
            candidates = []
            for row in output:
                cls_index = int(np.argmax(row[4:]))
                conf = float(row[4 + cls_index])
                if conf < threshold:
                    continue
                x, y, w, h = row[:4]
                box = self._original(np.array([x - w/2, y - h/2, x + w/2, y + h/2]), frame)
                if box[2] > box[0] and box[3] > box[1]:
                    candidates.append((box, conf, native_ids[cls_index]))
            # Class-wise NMS; OpenCV returns kept positions for xywh rectangles.
            for cls in native_ids:
                group = [(box, score) for box, score, label in candidates if label == cls]
                if not group:
                    continue
                rectangles = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b, _ in group]
                indices = cv2.dnn.NMSBoxes(rectangles, [s for _, s in group], threshold,
                                           float(self.model.postprocess.get("iou", 0.7)))
                for index in np.asarray(indices).reshape(-1):
                    box, score = group[int(index)]
                    detections.append(Detection(box, score, cls, self.model.classes[cls], self.model.model_id))
        else:
            raise ValueError(f"unknown ONNX output format: {format_name}")
        return DetectionResult(frame.meta, tuple(detections), self.model.model_id)

    def close(self) -> None:
        if hasattr(self, "session"):
            del self.session
        super().close()


def build_detector(model_id: str, registry: ModelRegistry, device: str = "auto",
                   fallback: tuple[str, ...] = ("cpu",)) -> BaseDetector:
    model = registry.get(model_id)
    if model.runtime != "onnxruntime":
        raise ValueError("The tracking app requires an ONNX Runtime detector")
    return OnnxDetector(model, registry, device, fallback)
