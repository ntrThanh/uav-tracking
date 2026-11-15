import json

import pytest

from src.common.config import CameraConfig, PipelineConfig, StopLine, resolve_path


def test_pipeline_defaults_from_repo_config():
    cfg = PipelineConfig.load("config/pipeline.json")
    assert cfg.detector.classes == {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    assert cfg.tracker.track_buffer == 30
    assert cfg.video.stride == 1


def test_pipeline_rejects_unknown_key():
    with pytest.raises(ValueError, match="Unknown keys"):
        PipelineConfig.from_dict({"detector": {"wieghts": "x.pt"}})


def test_camera_roundtrip(tmp_path):
    data = {
        "camera_id": "cam01",
        "reference_frame": "ref.jpg",
        "stop_line": {"a": [0.2, 0.7], "b": [0.8, 0.7], "allowed_direction": "down"},
        "zones": [{"name": "intersection", "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.6]]}],
        "light_roi": {"x": 0.45, "y": 0.05, "w": 0.04, "h": 0.08},
        "light_override": [{"start_s": 0.0, "end_s": 12.5, "state": "red"}],
    }
    cam = CameraConfig.from_dict(data)
    path = cam.save(tmp_path / "cam.json")
    assert json.loads(path.read_text()) == data
    again = CameraConfig.load(path)
    assert again == cam
    assert again.stop_line.to_pixels(1000, 500) == ((200.0, 350.0), (800.0, 350.0))
    assert again.light_roi.to_pixels(1000, 500) == (450, 25, 490, 65)


def test_camera_empty_geometry():
    cam = CameraConfig.load("config/cam01.json")
    assert cam.stop_line is None and cam.zones == [] and cam.light_roi is None


@pytest.mark.parametrize("bad", [
    {"a": [0.2, 1.7], "b": [0.8, 0.7]},
    {"a": [0.2, 0.7], "b": [0.2, 0.7]},
    {"a": [0.2, 0.7], "b": [0.8, 0.7], "allowed_direction": "south"},
])
def test_stop_line_validation(bad):
    with pytest.raises(ValueError):
        StopLine(**bad)


def test_resolve_path_relative_to_project_root():
    assert resolve_path("config/pipeline.json").is_file()
