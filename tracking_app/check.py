"""Validate the local desktop runtime and all bundled ONNX files."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from tracking_app.engine.models import ModelRegistry

from .config import CATALOG, Settings


def check() -> list[str]:
    problems = []
    for name in ("tkinter", "PIL", "cv2", "numpy", "scipy", "yaml", "onnxruntime"):
        if importlib.util.find_spec(name) is None:
            problems.append(f"Missing Python module: {name}")
    settings = Settings.from_dict({})
    registry = ModelRegistry.from_json(CATALOG)
    for model_id in (settings.vehicle_model, settings.plate_model):
        try:
            registry.path(model_id)
        except FileNotFoundError as exc:
            problems.append(f"Missing model: {exc.filename or str(exc)}")
    for directory in (settings.ocr_detection_dir, settings.ocr_model_dir):
        for name in ("inference.onnx", "inference.yml"):
            if not (Path(directory) / name).is_file():
                problems.append(f"Missing OCR model: {Path(directory) / name}")
    return problems


if __name__ == "__main__":
    issues = check()
    if issues:
        for issue in issues:
            print(issue)
        raise SystemExit(1)
    print("Tracking app: dependencies and ONNX files are ready.")
