"""Validated model catalog; architecture and class mapping are explicit metadata."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .schema import ModelMetadata


class ModelRegistry:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = Path(base_dir or Path.cwd()).resolve()
        self._models: dict[str, ModelMetadata] = {}

    def register(self, model: ModelMetadata) -> None:
        if model.model_id in self._models:
            raise ValueError(f"duplicate model ID: {model.model_id}")
        if not model.model_id or not model.task or not model.architecture or not model.weights:
            raise ValueError("model ID, task, architecture and weights are required")
        if model.runtime not in {"pytorch", "onnxruntime", "tflite", "tensorrt", "qnn"}:
            raise ValueError(f"unknown runtime: {model.runtime}")
        if min(model.input_size) <= 0 or not model.classes:
            raise ValueError("input size and class mapping are required")
        if len(set(model.classes.values())) != len(model.classes):
            raise ValueError("class names must be unique")
        self._models[model.model_id] = model

    def get(self, model_id: str) -> ModelMetadata:
        try:
            return self._models[model_id]
        except KeyError as exc:
            raise KeyError(f"unknown model {model_id!r}") from exc

    def path(self, model_id: str, verify_checksum: bool = True) -> Path:
        model = self.get(model_id)
        path = Path(model.weights)
        if not path.is_absolute():
            path = self.base_dir / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if verify_checksum and model.checksum:
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
            if digest.lower() != model.checksum.lower():
                raise ValueError(f"checksum mismatch for {model_id}")
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelRegistry":
        path = Path(path).resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("models"), list):
            raise ValueError("model catalog needs a models list")
        registry = cls(path.parent)
        for item in data["models"]:
            values = dict(item)
            values["classes"] = {int(k): str(v) for k, v in values["classes"].items()}
            values["input_size"] = tuple(values["input_size"])
            registry.register(ModelMetadata(**values))
        return registry
