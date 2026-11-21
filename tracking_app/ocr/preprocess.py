"""Replaceable image preparation before plate OCR."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

import cv2
import numpy as np

if TYPE_CHECKING:
    from tracking_app.config import Settings


class PlatePreprocessor(Protocol):
    def variants(self, image: np.ndarray) -> Iterable[np.ndarray]: ...


@dataclass(frozen=True)
class PreprocessStage:
    label: str
    image: np.ndarray
    ocr_attempt: int | None = None


class StandardPlatePreprocessor:
    def __init__(self, settings: Settings, mode: str) -> None:
        self.settings = settings
        self.mode = mode

    def stages(self, image: np.ndarray) -> Iterable[PreprocessStage]:
        steps = [PreprocessStage("Original crop", image)]
        height, width = image.shape[:2]
        scale = min(self.settings.ocr_upscale, 960 / max(width, height))
        if scale > 1:
            image = cv2.resize(image, (round(width * scale), round(height * scale)),
                               interpolation=cv2.INTER_CUBIC)
            steps.append(PreprocessStage(f"Resize x{scale:.2f}", image))
        if self.settings.ocr_sharpen:
            blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
            amount = self.settings.ocr_sharpen
            image = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
            steps.append(PreprocessStage(f"Sharpen {amount:g}", image))
        if self.mode in ("none", "auto"):
            steps[-1] = replace(steps[-1], ocr_attempt=1)
        yield from steps
        if self.mode in ("clahe", "auto"):
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            light, a, b = cv2.split(lab)
            light = cv2.createCLAHE(clipLimit=self.settings.ocr_clahe_clip,
                                    tileGridSize=(4, 4)).apply(light)
            yield PreprocessStage(f"CLAHE {self.settings.ocr_clahe_clip:g}",
                                  cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2BGR),
                                  2 if self.mode == "auto" else 1)

    def variants(self, image: np.ndarray) -> Iterable[np.ndarray]:
        for stage in self.stages(image):
            if stage.ocr_attempt is not None:
                yield stage.image


Factory = Callable[["Settings"], PlatePreprocessor]
_PREPROCESSORS: dict[str, Factory] = {}


def register_preprocessor(name: str, factory: Factory) -> None:
    if name in _PREPROCESSORS:
        raise ValueError(f"OCR preprocessor already registered: {name}")
    _PREPROCESSORS[name] = factory


def available_preprocessors() -> tuple[str, ...]:
    return tuple(sorted(_PREPROCESSORS))


def create_preprocessor(settings: Settings) -> PlatePreprocessor:
    return _PREPROCESSORS[settings.ocr_preprocess_mode](settings)


for _mode in ("none", "clahe", "auto"):
    register_preprocessor(_mode, lambda settings, mode=_mode: StandardPlatePreprocessor(settings, mode))
