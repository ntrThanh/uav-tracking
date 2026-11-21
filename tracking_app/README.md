# Vehicle Tracking desktop app

This Python app opens a desktop window with Tkinter. Select a local video, enter an RTSP URL, or select a USB camera index. Choose the models, tracker, and thresholds, then click **Start AI**. A video session ends when the file ends; RTSP and USB sessions continue until you click **Stop**. Only one session runs at a time. The app stores license plates and image coordinates by track ID. You can search tracks, view trajectories, and export a session to Excel.

The sidebar has three pages: **Live monitor** for source selection and video preview, **Session history** for searching vehicles and plates, viewing trajectories, and exporting Excel, and **Model settings** for selecting backends, models, trackers, and thresholds. Every settings card has **Save** and **Reset defaults** controls. **Save** persists only that card's fields; edits in other cards remain unsaved. **Reset defaults** changes only that card's visible fields, and you can click **Save** to persist the defaults. **Start AI** uses and saves all currently visible settings.

**Live monitor** shows processing FPS, average inference time per processed frame, and cumulative inference time for the current session. Click **Show details** to open the inference breakdown over the video, and **Hide details** to close it. The breakdown lists vehicle detection, tracking, plate detection, OCR, and remaining pipeline work, with time spent on the last frame, average time per call, and call count. OCR can run more than once per frame, while plate detection and OCR run only when needed. FPS includes frame processing and preview work; inference time measures the pipeline only. The values remain visible when a video finishes and reset when a new session starts.

Selecting a track in **Session history** shows its trajectory drawn on the last frame where that vehicle was seen, plus a vehicle snapshot and a plate snapshot when the plate detector found one. The trajectory image marks the starting point, ending point, and final vehicle box in original-frame coordinates. Evidence is saved as JPEG files under `outputs/tracking_app/evidence/<session_id>/`. Plate images are retained even when OCR cannot read the text. Excel export copies the trajectory and crop images into a folder beside the workbook and lists their relative paths in the Vehicles sheet. Sessions recorded before evidence capture was added have no snapshots; run the source again to capture them.

## Install and run

Use a Python installation with Tkinter. From the project root:

```bash
python -m pip install -r tracking_app/requirements.txt
./run_app.sh
```

`run_app.sh` checks dependencies and model files, then opens the window. On this machine it uses `.venv-tracking`; choose another Python with `TRACKING_PYTHON=/path/to/python ./run_app.sh`. You can run the preflight check alone with `python -m tracking_app.check`.

Tkinter is usually included with Python. Some Linux distributions require the `python3-tk` system package. The app uses Pillow for the preview. It does not require a browser or web server.

Videos are read directly from the selected file path. RTSP sources use `rtsp://...` URLs. USB cameras are devices on the computer running the app. Settings are stored in `outputs/tracking_app/settings.json`; sessions and trajectories are stored in `outputs/tracking_app/sessions.sqlite3`.

The model list comes from `tracking_app/models.json`; tracking algorithms and YAML configurations live in `tracking_app/engine/trackers`. The app catalog accepts ONNX Runtime models that follow the output contract in `tracking_app/engine/detectors.py`. Vehicle models use the `object_detection` task; plate models use `license_plate_detection`. The default vehicle model is YOLO11n COCO for street cameras. YOLO11m VisDrone is available for drone videos. Vehicle classes, separate vehicle and plate confidence/IoU thresholds, tracker, OCR, frame stride, plate detection interval, and preview width are configurable in the window. Input dimensions come from the exported ONNX model and its catalog metadata.

Three YOLO models have been exported to ONNX: `yolo11n.onnx`, `best_yolo11m.onnx`, and `plate_yolo11s.onnx`. These large artifacts are excluded by `.gitignore`. To export them again from `.pt` checkpoints on another machine:

```bash
python -m pip install -r tracking_app/requirements-export.txt
python -m tracking_app.export_onnx
```

Exporting requires a compatible PyTorch installation. The research project's RF-DETR and Faster R-CNN checkpoints have not been converted to ONNX and are not in the app catalog.

## Modules and replacement points

| Directory | Responsibility | Contract |
|---|---|---|
| `vehicle/` | Detect vehicles | `VehicleDetector.detect(frame)` |
| `tracking/` | Assign IDs and maintain tracks | `VehicleTracker.update(frame, detections)` |
| `plate/` | Find plates within tracked vehicles | `PlateDetector.detect(frame, tracks)` |
| `ocr/` | Read text from plate crops | `PlateReader.read(crop)` |
| `pipeline/` | Connect the components | `InferencePipeline.process(frame)` |
| `engine/` | App-owned ONNX runtime, frame source, ROI adapter, and tracker implementations | `FrameSource`, `OnnxDetector`, `TrackerAdapter` |
| `ui/` | Tkinter desktop window | Calls `SessionRunner` and `SessionStore` |

`runner.py` manages the video or stream lifecycle, persistence, and preview. It accepts a `pipeline_factory` constructor argument so the whole pipeline can be replaced. The desktop app uses its own engine in `tracking_app/engine`, including ONNX adapters, frame sources, track ROI logic, tracker algorithms, and tracker configurations. It does not import `src/engine` or `src/trackers` at runtime. The research packages remain available to other project scripts. To add a backend, create a Python file in `tracking_app/plugins/`, implement its contract, and call the corresponding module's `register_backend(name, factory)`. Plugin files load when the app starts, so `runner.py` and `pipeline/session.py` do not need changes. New backend names appear in the interface. For example:

```python
# tracking_app/plugins/my_ocr.py
from tracking_app.ocr import register_backend

class MyReader:
    def __init__(self, settings):
        self.minimum = settings.ocr_confidence

    def read(self, crop):
        # Return (plate_text, confidence) or None.
        return None

register_backend("my_ocr", MyReader)
```

Models and algorithms for the `engine` backend are validated against the catalog. A custom backend should validate its own settings.
Custom pipelines can provide a plate bounding box in `PlateRead.bbox` to enable plate snapshots; vehicle snapshots use each track's bounding box. The built-in pipeline supplies the plate box even when OCR returns no text.

The default OCR backend runs two locally exported PaddleOCR models directly with ONNX Runtime: `tracking_app/models/ocr/PP-OCRv5_mobile_det/inference.onnx` and `tracking_app/models/ocr/latin_PP-OCRv5_mobile_rec/inference.onnx`. Each directory also contains `inference.yml`. This backend does not download models or require the PaddleOCR package. To use the official PaddleOCR package instead, install `tracking_app/requirements-paddle-sdk.txt` and select `paddleocr_onnx` in the app. Both models were tested with PaddleOCR and ONNX Runtime on `server22`. The YOLO plate model finds plate regions; the OCR models locate and read text in those regions. Selecting the `none` OCR backend leaves the plate text empty. Excel export uses Python's standard library. See the [PaddleOCR ONNX Runtime documentation](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html).

OCR preprocessing is configurable in **Model settings**. The default resizes plate crops by 2x before OCR; `auto` tries that image first and retries with CLAHE contrast enhancement only if the first pass fails. Resize factor, contrast mode, CLAHE limit, sharpen strength, and the direct ONNX text detection thresholds are editable. A larger resize factor can improve tiny plates but costs more inference time. After changing settings, select a track with a saved plate image in **Session history** and click **Retry selected OCR** to apply the current settings without rerunning the video. This action replaces that track's previous OCR text if a new result is found.

Trajectories use the bottom center of each detection box, measured in pixels in the original video. They are not GPS coordinates. If a vehicle loses its track and gets a new ID, the app does not automatically join the two segments. OCR can be wrong for blurred or occluded plates.
