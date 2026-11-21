# Reusable inference engine

Start with the [architecture and migration note](../../docs/engine_architecture.md). The engine runs synchronously and keeps one `PipelineSession` per source. Existing detector, tracker, training and evaluation CLIs remain unchanged.

## Contracts

`Frame.image` is BGR `uint8`; `FrameMetadata` carries source ID, frame ID, timestamp and original dimensions. `Detection` and `Track` boxes are `xyxy` pixels in the original frame. Every detection includes model ID, explicit class ID/name, and optional mask, keypoints, embedding and metadata. `detections_from_numpy` and `tracks_from_numpy` validate legacy `(N,6)` / `(N,7)` layouts. Class IDs are taken from model metadata rather than assuming VisDrone indexing.

`SystemError` records runtime failures. A rule emits an `Event` candidate, which is a business observation and not an exception.

## Run the examples

From the repository root, with dependencies and weights installed:

```bash
# UAV: video -> YOLO11m -> ByteTrack -> JSONL tracking output
python -m src.engine.cli --config src/engine/examples/uav_tracking.json \
  --source path/to/drone.mp4 --source-id drone_1 --device cpu --max-frames 100

# Traffic: video -> vehicle tracking + plate branch + traffic light ROI -> rules -> events
python -m src.engine.cli --config src/engine/examples/traffic_monitoring.json \
  --source path/to/traffic.mp4 --source-id road_1 --device cpu --max-frames 100
```

The ROI and stop line in the traffic example are placeholders. Draw real geometry per fixed camera before using its events. The VisDrone vehicle model is for research; its domain shift on local road cameras must be measured.

Use `--kind image|video|usb|rtsp`; for USB, `--source 0` selects device 0. File timestamps are seconds from the beginning of the file; USB/RTSP timestamps are wall time. `--stride N` processes every Nth frame without an unbounded queue. RTSP reconnects with capped exponential backoff. One source/session must not be shared between cameras.

## Configuration

`models.json` is a catalog. Each model needs ID, task, architecture, weights, runtime, `(height,width)`, precision, explicit class map, version, and optional SHA-256, preprocessing and postprocessing metadata. Paths are relative to the catalog. `.pth` files need a matching architecture (`rfdetr_m` or `frcnn_r50`); extension alone is insufficient.

Pipeline JSON has `models` and `nodes`. Node types: `detector`, `tracker`, `traffic_light`, `signal`, `rule`, `sink`, `track_sink`. Each node has `id`, `type`, optional `inputs`, `config`, `when` (`always`, `has_tracks`, `has_events`) and `on_error` (`raise`, `skip`). The graph checks references, contracts and cycles before model loading. Detector nodes may have normalized `roi`, or `track_rois:true` with one tracker input to run on vehicle crops. Conditional nodes skip when prerequisites are absent; failures in one branch skip dependents while independent branches continue under `on_error:skip`.

To switch detectors, change the detector node's `config.model`; to switch trackers, change the tracker node's `config.algorithm`. The existing tracker YAML files remain authoritative. A new source creates a new session and independent tracker/rule state.

## Add components

- **Detector:** register model metadata. For a new architecture, subclass `BaseDetector` with `load`, `predict` and `postprocess`, then add one factory branch in `build_detector`. Keep resize, decoding and NMS inside the adapter.
- **Tracker:** add an algorithm to the existing `src/trackers` registry and YAML config. `TrackerAdapter` discovers it automatically; it passes explicit classes and a fresh instance per session.
- **Rule:** subclass `BaseRule`, implement `evaluate(RuleContext)`, call `_allow` for cooldown, and register with `register_rule`. Rule state belongs to a session. The built-ins are line crossing, region entry/exit, counting, red-light candidate, wrong-way and stopped object. Unknown light produces no red-light event and records an insufficient-data outcome.
- **Sink:** implement `EventSink.write(event, frame)` and `close()`. Built-ins provide JSONL, callbacks and original-frame evidence. Event status remains `candidate` until reviewed by an application.

## Runtime and hardware

`runtime.discover()` reports CPU, CUDA availability and ONNX execution providers. PyTorch and ONNX Runtime are executable backends. Device selection records attempted devices, chosen provider and whether fallback occurred; ONNX session construction verifies provider and model loading. A provider listing does not prove every operator runs on that provider. Unsupported TFLite, TensorRT and QNN configurations raise `UnsupportedBackendError`; these are extension points rather than working backends.

ONNX Runtime is optional (`pip install onnxruntime` or a matching GPU build). The adapter supports output metadata `format: decoded_nx6` or `format: ultralytics_yolo`; metadata must match the exported model. Actual YOLO ONNX export parity is not yet benchmarked. The synthetic ONNX test runs an executable model on CPU.

## Tests and limits

```bash
PYTHONPATH=. python -m pytest -q tests/test_engine.py
PYTHONPATH=. python -m pytest -q tests/test_engine_onnx.py  # optional onnx + onnxruntime
```

The core suite uses fake frames and adapters. RF-DETR's real checkpoint needs the RF-DETR package and its dependencies; the adapter contract is tested with a mocked legacy detector in the default environment. No TensorRT/QNN/TFLite execution, calibrated speed estimation, multi-camera throughput or real-world violation accuracy is claimed.
