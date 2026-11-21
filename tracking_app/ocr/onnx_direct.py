"""Local ONNX Runtime adapter for exported PaddleOCR DB/CTC models.

This keeps the desktop app usable when the PaddleOCR Python package is absent.
The official PaddleOCR adapter remains available as ``paddleocr_onnx``.
"""
from __future__ import annotations

import cv2
import numpy as np
import yaml

from tracking_app.config import Settings
from .readers import normalize_plate


class PaddleONNXReader:
    def __init__(self, settings: Settings) -> None:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if settings.device == "cuda" and "CUDAExecutionProvider" not in providers:
            raise RuntimeError("ONNX Runtime has no CUDAExecutionProvider")
        selected = ["CUDAExecutionProvider", "CPUExecutionProvider"] if settings.device == "cuda" else ["CPUExecutionProvider"]
        self.detector = ort.InferenceSession(str(settings.ocr_detection_dir + "/inference.onnx"), providers=selected)
        self.recognizer = ort.InferenceSession(str(settings.ocr_model_dir + "/inference.onnx"), providers=selected)
        with open(settings.ocr_model_dir + "/inference.yml", encoding="utf-8") as stream:
            metadata = yaml.safe_load(stream)
        self.characters = ["", *metadata["PostProcess"]["character_dict"], " "]
        self.minimum = settings.ocr_confidence
        self.text_threshold = settings.ocr_text_threshold
        self.box_threshold = settings.ocr_box_threshold
        self.unclip_ratio = settings.ocr_unclip_ratio

    @staticmethod
    def _prepare_detection(image: np.ndarray) -> tuple[np.ndarray, float, float]:
        height, width = image.shape[:2]
        scale = min(1.0, 960 / max(height, width))
        resized_width = max(32, int(round(width * scale / 32)) * 32)
        resized_height = max(32, int(round(height * scale / 32)) * 32)
        resized = cv2.resize(image, (resized_width, resized_height)).astype(np.float32) / 255.0
        mean = np.array([.485, .456, .406], np.float32)
        std = np.array([.229, .224, .225], np.float32)
        tensor = ((resized - mean) / std).transpose(2, 0, 1)[None]
        return np.ascontiguousarray(tensor), width / resized_width, height / resized_height

    def _regions(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        tensor, sx, sy = self._prepare_detection(image)
        output = self.detector.run(None, {self.detector.get_inputs()[0].name: tensor})[0]
        probability = np.asarray(output)[0, 0]
        mask = (probability > self.text_threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if width < 3 or height < 3:
                continue
            if float(probability[y:y+height, x:x+width].mean()) < self.box_threshold:
                continue
            cx, cy = (x + width/2) * sx, (y + height/2) * sy
            # Rectangular expansion is sufficient for plate crops; the official SDK handles polygons.
            expanded_width, expanded_height = width * sx * self.unclip_ratio, height * sy * self.unclip_ratio
            x1 = max(0, int(cx - expanded_width/2)); y1 = max(0, int(cy - expanded_height/2))
            x2 = min(image.shape[1], int(cx + expanded_width/2)); y2 = min(image.shape[0], int(cy + expanded_height/2))
            if x2 > x1 and y2 > y1:
                regions.append((x1, y1, x2, y2))
        return sorted(regions, key=lambda box: (box[1], box[0]))[:12]

    def _recognize(self, crop: np.ndarray) -> tuple[str, float] | None:
        height, width = crop.shape[:2]
        target_width = min(320, max(1, int(np.ceil(48 * width / height))))
        resized = cv2.resize(crop, (target_width, 48)).astype(np.float32) / 255.0
        tensor = np.zeros((48, 320, 3), np.float32)
        tensor[:, :target_width] = (resized - .5) / .5
        tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])
        scores = np.asarray(self.recognizer.run(None, {self.recognizer.get_inputs()[0].name: tensor})[0])[0]
        indices = scores.argmax(axis=1)
        confidence = scores.max(axis=1)
        chars, values = [], []
        previous = -1
        for index, score in zip(indices, confidence):
            index = int(index)
            if index != 0 and index != previous and index < len(self.characters):
                chars.append(self.characters[index])
                values.append(float(score))
            previous = index
        value = normalize_plate("".join(chars))
        if not value or not values:
            return None
        return value, float(np.mean(values))

    def read(self, crop: np.ndarray) -> tuple[str, float] | None:
        if crop.size == 0:
            return None
        candidates = []
        for x1, y1, x2, y2 in self._regions(crop):
            result = self._recognize(crop[y1:y2, x1:x2])
            if result and result[1] >= self.minimum:
                candidates.append(result)
        if not candidates:
            fallback = self._recognize(crop)
            if fallback and fallback[1] >= self.minimum:
                candidates.append(fallback)
        text = "".join(item[0] for item in candidates)
        if not 5 <= len(text) <= 12:
            return None
        return text, min(item[1] for item in candidates)
