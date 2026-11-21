# Tracking app engine

This package contains the inference runtime used by the desktop app. App code imports
`tracking_app.engine`; it does not import the research packages under `src/`.

| Module | Role |
|---|---|
| `schema.py` | Frame, detection, track, and model metadata contracts; all boxes use original-image xyxy pixels. |
| `sources.py` | Video, RTSP, and USB capture with frame stride and RTSP reconnection. |
| `models.py` | ONNX model catalog and weight-path validation. |
| `runtime.py` | ONNX Runtime CPU/CUDA provider selection and fallback. |
| `detectors.py` | ONNX image preprocessing, inference, and detection postprocessing. |
| `roi.py` | Run the plate detector inside each tracked vehicle and translate boxes to full-frame coordinates. |
| `tracking.py` | Adapt frame detections to the local multi-object tracker package. |
| `trackers/` | Tracker algorithms, registry, YAML configurations, and optional ReID adapter. |

`tracking_app.pipeline.InferencePipeline` composes these pieces with the app's
vehicle, plate, tracking, and OCR interfaces. `tracking_app.runner.SessionRunner`
owns the active source, persistence, preview, and UI snapshot. The research
`src/engine` remains separate for its CLI and rule pipelines.

The default vehicle and plate ONNX files, the optional VisDrone ONNX file, and
the optional ReID weights live under `tracking_app/models/`. Model artifacts are
excluded from Git and must be present when running the corresponding backend.
