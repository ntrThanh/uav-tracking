"""ONNX Runtime provider selection for the desktop app."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class UnsupportedBackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeChoice:
    runtime: str
    device: str
    provider: str | None
    attempts: tuple[str, ...]
    fallback_used: bool


def device_candidates(device: str, fallback: tuple[str, ...] = ("cpu",)) -> tuple[str, ...]:
    if device == "auto":
        ordered = ("cuda", *fallback)
    else:
        ordered = (device, *fallback)
    return tuple(dict.fromkeys(ordered))


def open_onnx(path: str, device: str, fallback: tuple[str, ...] = ("cpu",)) -> tuple[Any, RuntimeChoice]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise UnsupportedBackendError("ONNX Runtime is not installed") from exc
    available = set(ort.get_available_providers())
    attempted = []
    errors = []
    for candidate in device_candidates(device, fallback):
        provider = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}.get(candidate.split(":")[0])
        attempted.append(candidate)
        if provider not in available:
            errors.append(f"{candidate}: provider unavailable")
            continue
        try:
            options = [{"device_id": int(candidate.split(":", 1)[1])}] if candidate.startswith("cuda:") else [{}]
            session = ort.InferenceSession(path, providers=[(provider, options[0])])
            if session.get_providers()[0] != provider:
                raise RuntimeError(f"provider changed to {session.get_providers()}")
            return session, RuntimeChoice("onnxruntime", candidate, provider, tuple(attempted), len(attempted) > 1)
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
    raise UnsupportedBackendError("ONNX model could not load: " + "; ".join(errors))
