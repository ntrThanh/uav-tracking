"""Operations tests without requiring a camera, GPU or model weights."""
import json

import numpy as np
import pytest

from src.common.config import CameraConfig
from src.detect.vehicle import Detections
from src.operations.engines import IoUTracker
from src.operations.monitor import CameraProfile, CameraStore, EventStore
from src.operations.rules import RedLightRule, crossed_stop_line
from src.common.config import TrackerConfig
from src.track.tracker import Track
from src.api.app import create_app


def geometry():
    return CameraConfig.from_dict({
        "camera_id": "cam1",
        "stop_line": {"a": [0.2, 0.5], "b": [0.8, 0.5], "allowed_direction": "down"},
        "zones": [{"name": "approach", "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.49], [0.1, 0.49]]}],
        "light_roi": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1},
    })


def test_crossing_requires_direction_segment_zone_and_red():
    cfg = geometry()
    assert crossed_stop_line((0.5, 0.4), (0.5, 0.6), cfg)
    assert not crossed_stop_line((0.5, 0.6), (0.5, 0.4), cfg)
    assert not crossed_stop_line((0.95, 0.4), (0.95, 0.6), cfg)
    rule = RedLightRule(cfg)
    assert not rule.update(4, (0.5, 0.4), "red")
    assert not rule.update(4, (0.5, 0.6), "green")
    assert not rule.update(4, (0.5, 0.4), "red")
    assert rule.update(4, (0.5, 0.6), "red")
    assert not rule.update(4, (0.5, 0.4), "red")
    assert not rule.update(4, (0.5, 0.6), "red")


def test_camera_store_masks_credentials_and_restricts_video_path(tmp_path):
    store = CameraStore(tmp_path / "cameras")
    profile = CameraProfile.from_dict({
        "name": "Ngã tư", "source_type": "rtsp", "source": "rtsp://user:secret@127.0.0.1:554/live",
        "detector": "yolo11n", "tracker": "iou",
    }, "cam1")
    store.save(profile)
    assert store.get("cam1") == profile
    assert "secret" not in json.dumps(store.list()[0].public())
    assert (tmp_path / "cameras" / "cam1.json").stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError):
        CameraProfile.from_dict({"source_type": "video", "source": str(tmp_path / "x.mp4")}, "cam2")


def test_iou_tracker_keeps_id_for_adjacent_boxes():
    tracker = IoUTracker(TrackerConfig())
    def det(x):
        return Detections(np.array([[x, 10, x + 20, 30]], np.float32),
                          np.array([0.9], np.float32), np.array([2], np.int32), ["car"])
    first = tracker.update(det(10), 1, 0.0)
    second = tracker.update(det(12), 2, 0.1)
    assert first[0].track_id == second[0].track_id


def test_event_store_records_image_and_metadata(tmp_path):
    store = EventStore(tmp_path / "events")
    frame = np.zeros((100, 100, 3), np.uint8)
    track = Track(7, np.array([10, 10, 30, 30], np.float32), 0.9, 2, "car")
    event = store.record("cam1", frame, track, "red")
    assert event["type"] == "red_light"
    assert len(store.list("cam1")) == 1
    assert store.image_path("cam1", event["event_id"]).exists()
    reviewed = store.review("cam1", event["event_id"], "confirmed")
    assert reviewed["review_status"] == "confirmed"
    assert store.list("cam1")[0]["review_status"] == "confirmed"


def test_camera_api_wiring_without_network(tmp_path):
    app = create_app(config_dir=tmp_path, web_dir=tmp_path)
    def endpoint(path, method):
        return next(r.endpoint for r in app.routes if getattr(r, "path", "") == path
                    and method in getattr(r, "methods", set()))
    save = endpoint("/api/managed-cameras/{camera_id}", "PUT")
    listed = endpoint("/api/managed-cameras", "GET")
    engines = endpoint("/api/engines", "GET")
    result = save("cam1", {"name": "Camera 1", "source_type": "rtsp",
                           "source": "rtsp://user:pass@127.0.0.1/live",
                           "detector": "yolo11n", "tracker": "iou"})
    assert result["source"] == "rtsp://127.0.0.1/live"
    assert listed()[0]["camera_id"] == "cam1"
    assert "iou" in engines()["trackers"]
