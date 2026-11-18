import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from src.api.app import create_app  # noqa: E402
from src.common.config import PROJECT_ROOT  # noqa: E402

SAMPLE = PROJECT_ROOT / "data/samples/person-bicycle-car-detection.mp4"
CAM = {
    "camera_id": "cam01",
    "reference_frame": None,
    "stop_line": {"a": [0.2, 0.7], "b": [0.8, 0.7], "allowed_direction": "down"},
    "zones": [{"name": "intersection", "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.6], [0.1, 0.6]]}],
    "light_roi": {"x": 0.45, "y": 0.05, "w": 0.04, "h": 0.08},
    "light_override": None,
}


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(config_dir=tmp_path, web_dir=PROJECT_ROOT / "web"))


def test_index_and_static_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Trung tâm giám sát" in r.text
    assert client.get("/editor").status_code == 200
    assert client.get("/static/dashboard.js").status_code == 200


def test_missing_camera_is_404_and_list_empty(client):
    assert client.get("/api/cameras/cam09").status_code == 404
    assert client.get("/api/cameras").json() == []


def test_put_get_roundtrip(client, tmp_path):
    r = client.put("/api/cameras/cam01", json=CAM)
    assert r.status_code == 200, r.text
    saved = r.json()
    assert saved == CAM
    assert (tmp_path / "cam01.json").is_file()
    assert client.get("/api/cameras/cam01").json() == saved
    listed = client.get("/api/cameras").json()
    assert listed == [{"camera_id": "cam01", "has_reference": False, "has_stop_line": True, "n_zones": 1, "has_light_roi": True}]


def test_put_invalid_geometry_is_422(client):
    bad = dict(CAM, stop_line={"a": [0.1, 2.0], "b": [0.5, 0.5], "allowed_direction": "down"})
    r = client.put("/api/cameras/cam01", json=bad)
    assert r.status_code == 422 and "stop_line" in r.json()["detail"]
    r = client.put("/api/cameras/cam01", json=dict(CAM, unknown_key=1))
    assert r.status_code == 422


def test_bad_camera_id(client):
    assert client.get("/api/cameras/cam%2001").status_code == 400
    assert client.put("/api/cameras/pipeline", json=CAM).status_code == 400


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample video missing")
def test_reference_from_video_and_autofill(client, tmp_path):
    r = client.post("/api/cameras/cam01/reference/from-video", json={"video_path": str(SAMPLE), "time_s": 1.0})
    assert r.status_code == 200, r.text
    assert (r.json()["width"], r.json()["height"]) == (768, 432)
    assert (tmp_path / "cam01_ref.jpg").is_file()
    assert client.get("/api/cameras/cam01/reference.jpg").headers["content-type"] == "image/jpeg"
    saved = client.put("/api/cameras/cam01", json=CAM).json()
    assert saved["reference_frame"] == "cam01_ref.jpg"


def test_reference_from_missing_video(client):
    r = client.post("/api/cameras/cam01/reference/from-video", json={"video_path": "nope.mp4"})
    assert r.status_code == 404


def test_reference_upload(client):
    ok, buf = cv2.imencode(".png", np.zeros((40, 60, 3), np.uint8))
    assert ok
    r = client.post("/api/cameras/cam02/reference/upload", files={"file": ("ref.png", buf.tobytes(), "image/png")})
    assert r.status_code == 200 and r.json()["width"] == 60
    r = client.post("/api/cameras/cam02/reference/upload", files={"file": ("x.bin", b"not an image", "application/octet-stream")})
    assert r.status_code == 400


def test_detect_light_endpoint(client, tmp_path):
    # Upload reference frame
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[10:30, 40:60] = (0, 0, 255)  # red box
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    client.post("/api/cameras/cam01/reference/upload", files={"file": ("ref.jpg", buf.tobytes(), "image/jpeg")})
    client.put("/api/cameras/cam01", json=CAM)

    # Call detect-light with custom ROI
    res = client.post(
        "/api/cameras/cam01/detect-light",
        json={"roi": {"x": 0.4, "y": 0.1, "w": 0.2, "h": 0.2}},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["state"] == "red"
    assert data["crop_base64"] is not None

