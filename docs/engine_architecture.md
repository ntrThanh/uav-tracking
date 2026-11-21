# Modular inference engine: architecture and migration

This document describes the research `src/engine` package. The desktop tracking app now uses its own runtime under `tracking_app/engine`; see `tracking_app/engine/README.md` for its modules and ownership.

The reusable package lives in `src/engine`. Existing training, detection cache, TrackEval, and tracker implementations retain their public interfaces. `src/engine` adapts them; it does not copy their algorithms.

## Data flow and ownership

```text
Source.read() -> Frame -> Pipeline(DAG)
                         | detector node -> DetectionResult
                         | tracker node  -> TrackingResult (one instance per stream)
                         | rule node     -> Event candidates
                         ` sink node     -> JSON / callback / evidence adapter
```

`Frame` owns BGR pixels and metadata. All boxes in the engine are `xyxy` pixels of the **original** image. Each `Detection` carries its model ID and explicit class mapping; model-native indices never leak into tracker or rule nodes. The legacy NumPy layouts are converted only at adapter boundaries. `SystemError` is separate from business `Event`.

Each pipeline is synchronous and processes one frame at a time. `PipelineGraph` validates dependencies, cycles, node kinds and contracts before loading models. `PipelineSession` owns node instances and closes them in reverse order. Tracking and rule instances belong to one session per camera, preventing cross-camera state sharing. A graph can contain parallel branches and conditional nodes; no queue or distributed scheduler is introduced.

## Components

| Component | Responsibility | Existing implementation reused |
|---|---|---|
| `schema.py` | Validated frame, detection, track, event contracts and NumPy conversion | `src/detectors/base.py`, `src/trackers/base.py` layouts |
| `models.py` | Model metadata, IDs, class maps, checksum and config loading | Detector defaults in `src/detectors/` |
| `runtime.py` | Device discovery, explicit provider selection and fallback reports | PyTorch; optional ONNX Runtime |
| `detectors.py` | Lifecycle and decoding adapters | `src/detectors` YOLO, RF-DETR, Faster R-CNN; ONNX output adapter |
| `tracking.py` | Session-local wrappers for five existing trackers | `src/trackers` registry/factory |
| `rules.py` | Stateful rules and registry | Stop-line crossing geometry |
| `graph.py` | Validated deterministic DAG and node execution | New, intentionally small |
| `sources.py` | Image/video/USB/RTSP lifecycle and timestamping | OpenCV video capture |
| `events.py`, `metrics.py` | JSON/callback sinks and per-stage timings | New |

## Runtime policy

PyTorch adapters use the existing model loaders and require architecture metadata for `.pth`. ONNX Runtime requires an installed package and compatible ONNX output contract. The available ONNX adapter accepts decoded `Nx6` boxes and Ultralytics export output with explicit metadata. Provider and fallback are recorded. An accelerator being listed does not prove the model loads: session creation is the final capability check. TFLite, TensorRT and QNN report explicit unsupported errors until executable adapters and tests exist.

## Migration

1. New CLI `python -m src.engine.cli --config ...` runs the reusable package. Existing CLIs continue unchanged.
2. UAV example selects a legacy detector adapter and a tracker from `src/trackers`.
3. Traffic monitoring remains available as a standalone pipeline example. Each source creates its own session, so tracking and rule state stays separate across cameras.

## Deployment limits

The engine executes locally and synchronously. Per-camera parallelism remains the responsibility of the caller. GPU memory allocation, long-running RTSP load, OCR quality, automatic event review and hardware runtimes beyond tested PyTorch/ONNX are outside this core. No real-world speed is derived from pixel displacement.

The UAV example was exercised on one video frame with the existing YOLO11m checkpoint. The traffic graph was exercised on two frames with the vehicle checkpoint and light classifier; no tracks were produced on that sample, so its conditional plate and evidence branches did not run. The plate checkpoint was loaded and run separately on one frame. These smoke checks establish execution and resource cleanup, not detection quality or live-camera reliability.
