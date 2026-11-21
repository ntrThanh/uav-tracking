"""Runtime capability checks and explicit device/provider selection."""
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


def discover() -> dict[str, Any]:
    report: dict[str, Any] = {"cpu": True, "cuda": False, "pytorch": False,
                              "onnxruntime": False, "onnx_providers": []}
    try:
        import torch
        report["pytorch"] = True
        report["cuda"] = bool(torch.cuda.is_available())
        if report["cuda"]:
            report["gpu_name"] = torch.cuda.get_device_name(0)
            try:
                report["cuda_free_bytes"] = torch.cuda.mem_get_info()[0]
            except Exception:
                pass
    except ImportError:
        pass
    try:
        import onnxruntime as ort
        report["onnxruntime"] = True
        report["onnx_providers"] = ort.get_available_providers()
    except ImportError:
        pass
    return report


def device_candidates(device: str, fallback: tuple[str, ...] = ("cpu",)) -> tuple[str, ...]:
    if device == "auto":
        ordered = ("cuda", *fallback)
    else:
        ordered = (device, *fallback)
    return tuple(dict.fromkeys(ordered))


def choose_pytorch(device: str, fallback: tuple[str, ...] = ("cpu",)) -> RuntimeChoice:
    capabilities = discover()
    if not capabilities["pytorch"]:
        raise UnsupportedBackendError("PyTorch is not installed")
    attempted = []
    for candidate in device_candidates(device, fallback):
        attempted.append(candidate)
        if candidate == "cpu":
            return RuntimeChoice("pytorch", "cpu", None, tuple(attempted), len(attempted) > 1)
        if candidate.startswith("cuda") and capabilities["cuda"]:
            import torch
            try:
                if ":" in candidate and int(candidate.split(":", 1)[1]) >= torch.cuda.device_count():
                    continue
            except ValueError:
                continue
            return RuntimeChoice("pytorch", candidate if ":" in candidate else "cuda:0", None,
                                 tuple(attempted), len(attempted) > 1)
    raise UnsupportedBackendError(f"no PyTorch device available among {attempted}")


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


def require_supported(runtime: str) -> None:
    if runtime not in {"pytorch", "onnxruntime"}:
        raise UnsupportedBackendError(f"{runtime} backend has no verified executable adapter")
