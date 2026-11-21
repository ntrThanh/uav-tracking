"""OCR extension point. Add a reader without changing the session runner."""
from __future__ import annotations

import re
from typing import Callable, Protocol

import numpy as np

from tracking_app.config import Settings


class PlateReader(Protocol):
    def read(self, crop: np.ndarray) -> tuple[str, float] | None: ...


class PreprocessedReader:
    """Apply a replaceable preprocessor before any OCR backend."""

    def __init__(self, reader: PlateReader, preprocessor) -> None:
        self.reader = reader
        self.preprocessor = preprocessor

    def read(self, crop: np.ndarray) -> tuple[str, float] | None:
        if crop.size == 0:
            return None
        for variant in self.preprocessor.variants(crop):
            result = self.reader.read(variant)
            if result:
                return result
        return None


def normalize_plate(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


class NoOCR:
    def read(self, crop: np.ndarray) -> tuple[str, float] | None:
        return None


class PaddleOCRReader:
    """PaddleOCR text detection + recognition, both run by ONNX Runtime."""

    def __init__(self, settings: Settings) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError("PaddleOCR is missing. Install the app requirements to read license plates.") from exc
        options = {
            "engine": "onnxruntime",
            "text_detection_model_name": settings.ocr_detection_model_name,
            "text_recognition_model_name": settings.ocr_model_name,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
        if settings.device != "auto":
            options["device"] = "gpu" if settings.device == "cuda" else "cpu"
        if settings.ocr_model_dir:
            options["text_recognition_model_dir"] = settings.ocr_model_dir
        if settings.ocr_detection_dir:
            options["text_detection_model_dir"] = settings.ocr_detection_dir
        self.reader = PaddleOCR(**options)
        self.minimum = settings.ocr_confidence

    def read(self, crop: np.ndarray) -> tuple[str, float] | None:
        if crop.size == 0:
            return None
        results = self.reader.predict(crop)
        candidates = []
        for result in results:
            payload = result.json.get("res", result.json)
            boxes = payload.get("rec_boxes", [])
            for index, (text, score) in enumerate(zip(payload.get("rec_texts", []), payload.get("rec_scores", []))):
                score = float(score)
                if score < self.minimum:
                    continue
                box = boxes[index] if index < len(boxes) else (0, index, 0, 0)
                candidates.append((float(box[1]), float(box[0]), normalize_plate(text), score))
        candidates.sort(key=lambda item: (item[0], item[1]))
        text = "".join(item[2] for item in candidates)
        if not 5 <= len(text) <= 12:
            return None
        return text, min(item[3] for item in candidates)


Factory = Callable[[Settings], PlateReader]
_BACKENDS: dict[str, Factory] = {}


def register_backend(name: str, factory: Factory) -> None:
    if name in _BACKENDS:
        raise ValueError(f"OCR backend already registered: {name}")
    _BACKENDS[name] = factory


def available_backends() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))


def create_reader(settings: Settings) -> PlateReader:
    try:
        reader = _BACKENDS[settings.ocr_backend](settings)
    except KeyError as exc:
        raise ValueError(f"Unsupported OCR backend: {settings.ocr_backend}") from exc
    if settings.ocr_backend == "none":
        return reader
    from .preprocess import create_preprocessor
    return PreprocessedReader(reader, create_preprocessor(settings))


register_backend("none", lambda settings: NoOCR())
register_backend("paddleocr_onnx", PaddleOCRReader)
from .onnx_direct import PaddleONNXReader  # noqa: E402
register_backend("paddle_onnx_direct", PaddleONNXReader)
