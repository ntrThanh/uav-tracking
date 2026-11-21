"""Session lifecycle and persistence with a tiny synthetic video; no model weights required."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types
import zipfile
from dataclasses import asdict

import cv2
import numpy as np

from tracking_app.engine.schema import Detection, DetectionResult, Frame, Track, TrackingResult
from tracking_app.config import Settings
from tracking_app.ocr import normalize_plate
from tracking_app.ocr.readers import PaddleOCRReader
from tracking_app.ocr.readers import PreprocessedReader
from tracking_app.ocr.preprocess import create_preprocessor
from tracking_app.pipeline import InferencePipeline, PlateRead
from tracking_app.pipeline import create_pipeline
from tracking_app.vehicle import register_backend as register_vehicle
from tracking_app.plate import register_backend as register_plate
from tracking_app.tracking import register_backend as register_tracker
from tracking_app.ocr import register_backend as register_ocr
from tracking_app.runner import SessionRunner
from tracking_app.storage import SessionStore


def test_app_engine_imports_without_research_src(tmp_path):
    package = Path(__file__).resolve().parents[1] / "tracking_app"
    (tmp_path / "tracking_app").symlink_to(package, target_is_directory=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(tmp_path)
    code = """
import importlib.util
from tracking_app.config import CATALOG, Settings
from tracking_app.engine.models import ModelRegistry
from tracking_app.engine.trackers import available_trackers, load_config
from tracking_app.pipeline import create_pipeline

assert importlib.util.find_spec('src') is None
settings = Settings.from_dict({'ocr_backend': 'none', 'device': 'cpu'})
models = ModelRegistry.from_json(CATALOG)
assert models.path(settings.vehicle_model).is_file()
assert models.path(settings.plate_model).is_file()
for name in available_trackers():
    load_config(name)
with create_pipeline(settings):
    pass
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_app_plate_roi_restores_full_frame_boxes():
    from tracking_app.engine.roi import TrackROIDetector
    from tracking_app.engine.schema import FrameMetadata

    class Detector:
        model = types.SimpleNamespace(model_id="plate")
        def detect(self, crop):
            assert crop.image.shape == (20, 20, 3)
            region = Detection((2, 3, 8, 9), .8, 0, "plate", "plate")
            return DetectionResult(crop.meta, (region,), "plate")
        def close(self): pass

    frame = Frame(np.zeros((60, 80, 3), np.uint8), FrameMetadata("cam", 0, 0., 80, 60))
    track = Track(7, (10, 20, 30, 40), .9, 2, "car", "active", 0.)
    result = TrackROIDetector(Detector()).detect(frame, TrackingResult(frame.meta, (track,), "test"))
    assert result.frame == frame.meta
    assert result.detections[0].bbox == (12, 23, 18, 29)
    assert result.detections[0].metadata["parent_track_id"] == 7


def test_video_completes_and_trajectory_persists(tmp_path, monkeypatch):
    video = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48))
    assert writer.isOpened()
    for value in (20, 60, 100):
        writer.write(np.full((48, 64, 3), value, np.uint8))
    writer.release()

    class FakePipeline:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def process(self, frame: Frame):
            time.sleep(.002)
            self.last_timings = (("vehicle_detection", .5), ("tracking", .3),
                                 ("plate_detection", .2), ("ocr", .1), ("ocr", .1))
            track = Track(1, (10, 10, 30, 30), 0.9, 4, "car", "active", frame.meta.timestamp)
            return TrackingResult(frame.meta, (track,), "fake"), (
                PlateRead(1, "59A12345", .9, (12, 20, 24, 27), .95),)

    store = SessionStore(tmp_path / "sessions.sqlite3")
    runner = SessionRunner(store, pipeline_factory=lambda settings: FakePipeline())
    settings = Settings.from_dict({"ocr_backend": "none", "plate_interval": 1})
    store.start("test", "video", "tiny.avi", "2026-01-01T00:00:00Z", json.dumps({}))
    runner._run("test", "video", str(video), settings)
    assert runner.snapshot()["status"] == "completed"
    assert runner.snapshot()["frames"] == 3
    assert runner.snapshot()["plates"] == 1
    metrics = runner.snapshot()
    assert metrics["processing_fps"] > 0
    assert metrics["last_inference_ms"] > 0
    assert metrics["inference_total_s"] > 0
    assert abs(metrics["avg_inference_ms"] * 3 / 1000 - metrics["inference_total_s"]) < 1e-9
    timings = metrics["stage_timings"]
    assert timings["vehicle_detection"]["calls"] == 3
    assert timings["vehicle_detection"]["avg_ms"] == .5
    assert timings["ocr"]["calls"] == 6
    assert timings["ocr"]["last_calls"] == 2
    assert abs(timings["ocr"]["last_ms"] - .2) < 1e-9
    assert timings["other"]["calls"] == 3
    assert timings["other"]["last_ms"] > 0
    assert store.tracks("test", "59A")[0]["points"] == 3
    assert len(store.points("test", 1)) == 3
    evidence = SessionStore(tmp_path / "sessions.sqlite3").evidence("test", 1)
    assert set(evidence) == {"vehicle", "plate", "trajectory"}
    assert cv2.imread(str(evidence["vehicle"]["path"])).shape[:2] == (20, 20)
    assert cv2.imread(str(evidence["plate"]["path"])).shape[:2] == (7, 12)
    assert evidence["trajectory"]["frame_id"] == 2
    trajectory_image = cv2.imread(str(evidence["trajectory"]["path"]))
    assert trajectory_image.shape[:2] == (48, 64)
    assert trajectory_image[0, 0].mean() > 75  # Background comes from the final video frame.
    assert not np.array_equal(trajectory_image[30, 20], trajectory_image[0, 0])
    workbook = store.export_excel("test", tmp_path / "tracking.xlsx")
    assert (tmp_path / "tracking_evidence/1_trajectory.jpg").is_file()
    assert (tmp_path / "tracking_evidence/1_vehicle.jpg").is_file()
    assert (tmp_path / "tracking_evidence/1_plate.jpg").is_file()
    with zipfile.ZipFile(workbook) as archive:
        assert archive.testzip() is None
        assert b"59A12345" in archive.read("xl/worksheets/sheet1.xml")
        assert b"tracking_evidence/1_plate.jpg" in archive.read("xl/worksheets/sheet1.xml")
        assert b"Trajectory" in archive.read("xl/workbook.xml")
    store.set_plate("test", 1, "65K155292", .8, 2)
    assert store.tracks("test")[0]["plate"] == "65K155292"


def test_plate_normalization():
    assert normalize_plate("59-A 123.45") == "59A12345"


def test_ocr_preprocessing_resizes_and_retries_with_clahe():
    settings = Settings.from_dict({"ocr_backend": "none", "ocr_upscale": 2,
                                   "ocr_preprocess_mode": "auto"})
    seen = []
    class Reader:
        def read(self, image):
            seen.append(image.copy())
            return ("65K155292", .86) if len(seen) == 2 else None
    crop = np.full((46, 68, 3), 90, np.uint8)
    cv2.putText(crop, "65K1", (3, 21), cv2.FONT_HERSHEY_SIMPLEX, .45, (20, 20, 20), 1)
    reader = PreprocessedReader(Reader(), create_preprocessor(settings))
    assert reader.read(crop) == ("65K155292", .86)
    assert [image.shape[:2] for image in seen] == [(92, 136), (92, 136)]
    assert not np.array_equal(seen[0], seen[1])
    import pytest
    with pytest.raises(ValueError, match="ocr_upscale"):
        Settings.from_dict({"ocr_upscale": 5})


def test_each_settings_section_saves_and_resets_only_its_fields(monkeypatch):
    from tracking_app.ui import desktop
    from tracking_app.ui.layout import GROUP_FIELDS

    class Var:
        def __init__(self, value): self.value = str(value)
        def get(self): return self.value
        def set(self, value): self.value = str(value)

    fields = [field for group in GROUP_FIELDS.values() for field in group]
    assert len(fields) == len(set(fields)) == len(asdict(Settings()))
    base = Settings.from_dict({})
    app = types.SimpleNamespace(settings=base,
                                vars={key: Var(value) for key, value in asdict(base).items()},
                                message=Var(""))
    saved = []
    monkeypatch.setattr(desktop, "save_settings", saved.append)
    app.vars["confidence"].set("0.48")
    app.vars["plate_iou"].set("0.42")
    app.vars["ocr_upscale"].set("3")  # Unsaved change in another section.
    desktop.DesktopApp._save_group(app, "detection")
    assert saved[0].confidence == .48 and saved[0].plate_iou == .42
    assert saved[0].ocr_upscale == base.ocr_upscale
    assert app.vars["ocr_upscale"].get() == "3"
    desktop.DesktopApp._reset_group(app, "detection")
    assert app.vars["confidence"].get() == str(Settings().confidence)
    assert app.vars["plate_iou"].get() == str(Settings().plate_iou)
    assert app.vars["ocr_upscale"].get() == "3"


def test_vehicle_and_plate_iou_settings_are_independent(monkeypatch):
    import tracking_app.vehicle.detectors as vehicle_module
    import tracking_app.plate.detectors as plate_module
    captured = {}
    def vehicle_builder(model_id, registry, device):
        captured["vehicle"] = registry.get(model_id).postprocess
        return object()
    def plate_builder(model_id, registry, device):
        captured["plate"] = registry.get(model_id).postprocess
        return object()
    monkeypatch.setattr(vehicle_module, "build_detector", vehicle_builder)
    monkeypatch.setattr(plate_module, "build_detector", plate_builder)
    settings = Settings.from_dict({"iou": .7, "plate_iou": .4})
    vehicle_module._engine(settings)
    plate_module.EnginePlateDetector(settings)
    assert captured["vehicle"]["iou"] == .7
    assert captured["plate"]["iou"] == .4


def test_existing_settings_keep_their_previous_shared_iou(tmp_path, monkeypatch):
    import tracking_app.config as config
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"iou": .73}), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS", path)
    loaded = config.load_settings()
    assert loaded.iou == loaded.plate_iou == .73


def test_replaceable_pipeline_components_and_plate_interval():
    class Vehicle:
        def detect(self, frame):
            return DetectionResult(frame.meta, (Detection((10, 10, 30, 30), .9, 4, "car", "fake"),), "fake")
        def close(self): pass
    class Tracker:
        def update(self, frame, detections):
            assert detections.detections[0].class_name == "car"
            return TrackingResult(frame.meta, (Track(1, (10, 10, 30, 30), .9, 4, "car", "active", frame.meta.timestamp),), "fake")
        def close(self): pass
    class Plate:
        calls = 0
        def detect(self, frame, tracks):
            self.calls += 1
            region = Detection((12, 20, 24, 27), .9, 0, "plate", "fake", metadata={"parent_track_id": 1})
            return DetectionResult(frame.meta, (region,), "fake")
        def close(self): pass
    class OCR:
        def read(self, crop):
            assert crop.shape == (7, 12, 3)
            return ("59A12345", .9)

    plate = Plate()
    pipeline = InferencePipeline(Vehicle(), Tracker(), plate, OCR(), plate_interval=2)
    image = np.zeros((48, 64, 3), np.uint8)
    from tracking_app.engine.schema import FrameMetadata
    with pipeline:
        first, reads = pipeline.process(Frame(image, FrameMetadata("cam", 0, 0., 64, 48)))
        assert first.tracks[0].track_id == 1 and reads == (PlateRead(1, "59A12345", .9),)
        assert reads[0].bbox == (12, 20, 24, 27)
        assert [name for name, _ in pipeline.last_timings] == [
            "vehicle_detection", "tracking", "plate_detection", "ocr"]
        _, reads = pipeline.process(Frame(image, FrameMetadata("cam", 1, .2, 64, 48)))
        assert reads == ()
        assert [name for name, _ in pipeline.last_timings] == ["vehicle_detection", "tracking"]
    assert plate.calls == 1


def test_plate_evidence_is_available_when_ocr_cannot_read_text():
    from tracking_app.engine.schema import FrameMetadata
    class Vehicle:
        def detect(self, frame):
            return DetectionResult(frame.meta, (), "fake")
        def close(self): pass
    class Tracker:
        def update(self, frame, detections):
            return TrackingResult(frame.meta, (Track(1, (10, 10, 30, 30), .9, 4, "car", "active",
                                                     frame.meta.timestamp),), "fake")
        def close(self): pass
    class Plate:
        def detect(self, frame, tracks):
            region = Detection((12, 20, 24, 27), .8, 0, "plate", "fake",
                               metadata={"parent_track_id": 1})
            return DetectionResult(frame.meta, (region,), "fake")
        def close(self): pass
    class OCR:
        def read(self, crop): return None
    frame = Frame(np.zeros((48, 64, 3), np.uint8), FrameMetadata("cam", 0, 0., 64, 48))
    with InferencePipeline(Vehicle(), Tracker(), Plate(), OCR(), 1) as pipeline:
        _, reads = pipeline.process(frame)
    assert len(reads) == 1
    assert reads[0].text == "" and reads[0].bbox == (12, 20, 24, 27)
    assert reads[0].detection_confidence == .8


def test_backend_registry_accepts_replacements_without_engine_model_ids():
    class Vehicle:
        def detect(self, frame): return DetectionResult(frame.meta, (), "custom")
        def close(self): pass
    class Tracker:
        def update(self, frame, detections): return TrackingResult(frame.meta, (), "custom")
        def close(self): pass
    class Plate:
        def detect(self, frame, tracks): return DetectionResult(frame.meta, (), "custom")
        def close(self): pass
    class OCR:
        def read(self, crop): return None
    register_vehicle("test_vehicle", lambda settings: Vehicle())
    register_tracker("test_tracker", lambda settings: Tracker())
    register_plate("test_plate", lambda settings: Plate())
    register_ocr("test_ocr", lambda settings: OCR())
    settings = Settings.from_dict({"vehicle_backend": "test_vehicle", "vehicle_model": "custom_vehicle",
                                   "plate_backend": "test_plate", "plate_model": "custom_plate",
                                   "tracker_backend": "test_tracker", "tracker": "custom_tracker",
                                   "ocr_backend": "test_ocr"})
    from tracking_app.engine.schema import FrameMetadata
    frame = Frame(np.zeros((48, 64, 3), np.uint8), FrameMetadata("cam", 0, 0., 64, 48))
    with create_pipeline(settings) as pipeline:
        tracks, reads = pipeline.process(frame)
    assert tracks.tracks == () and reads == ()


def test_app_model_catalog_is_onnx_only():
    from tracking_app.engine.models import ModelRegistry
    from tracking_app.config import CATALOG
    registry = ModelRegistry.from_json(CATALOG)
    for model in registry._models.values():
        assert model.runtime == "onnxruntime"
        assert registry.path(model.model_id).suffix == ".onnx"


def test_paddle_ocr_uses_onnx_runtime_and_combines_plate_lines(monkeypatch):
    captured = {}
    class FakeResult:
        json = {"res": {"rec_texts": ["123.45", "59-A"], "rec_scores": [.95, .9],
                        "rec_boxes": [[2, 25, 40, 40], [1, 2, 30, 18]]}}
    class FakePaddle:
        def __init__(self, **kwargs): captured.update(kwargs)
        def predict(self, image): return [FakeResult()]
    monkeypatch.setitem(sys.modules, "paddleocr", types.SimpleNamespace(PaddleOCR=FakePaddle))
    reader = PaddleOCRReader(Settings.from_dict({"device": "cpu"}))
    assert captured["engine"] == "onnxruntime"
    assert captured["text_recognition_model_name"] == "latin_PP-OCRv5_mobile_rec"
    assert captured["text_detection_model_dir"].endswith("PP-OCRv5_mobile_det")
    assert reader.read(np.zeros((48, 64, 3), np.uint8)) == ("59A12345", .9)


def test_local_paddle_onnx_reads_synthetic_plate():
    import pytest
    pytest.importorskip("onnxruntime")
    from tracking_app.ocr import create_reader
    settings = Settings.from_dict({"device": "cpu"})
    assert settings.ocr_backend == "paddle_onnx_direct"
    image = np.full((100, 280, 3), 255, np.uint8)
    cv2.putText(image, "59A-123.45", (6, 67), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    assert create_reader(settings).read(image)[0] == "59A12345"
