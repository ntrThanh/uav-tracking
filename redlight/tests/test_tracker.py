import numpy as np
import pytest

pytest.importorskip("ultralytics")

from src.common.config import TrackerConfig  # noqa: E402
from src.detect.vehicle import Detections  # noqa: E402
from src.track.tracker import VehicleTracker  # noqa: E402


def dets(boxes, confs=None, names=None) -> Detections:
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    n = len(boxes)
    confs = np.full((n,), 0.9, np.float32) if confs is None else np.asarray(confs, np.float32)
    names = ["car"] * n if names is None else names
    return Detections(boxes, confs, np.full((n,), 2, np.int32), names)


def test_two_vehicles_keep_their_ids():
    tracker = VehicleTracker(TrackerConfig())
    ids_per_frame = []
    for f in range(40):
        a = [100, 100 + 5 * f, 160, 140 + 5 * f]  # moving down
        b = [400, 500 - 5 * f, 470, 540 - 5 * f]  # moving up
        tracks = tracker.update(dets([a, b]), frame_index=f, time_s=f / 25.0)
        ids_per_frame.append(tuple(sorted(t.track_id for t in tracks)))
    assert all(len(ids) == 2 for ids in ids_per_frame[1:])
    assert len(set(ids_per_frame[1:])) == 1, "ids changed during a clean two-object sequence"
    stats = tracker.stats()
    assert stats["n_tracks"] == 2 and stats["track_len_min"] >= 39
    assert stats["tracks_by_class"] == {"car": 2}


def test_reference_point_and_bounded_history():
    tracker = VehicleTracker(TrackerConfig(history_len=5))
    track_id = None
    for f in range(12):
        tracks = tracker.update(dets([[10, 10 + f, 50, 50 + f]]), f, f / 25.0)
        if tracks:
            track_id = tracks[0].track_id
            assert tracks[0].ref_point == (30.0, 50.0 + f)
    hist = tracker.history(track_id)
    assert len(hist.points) == 5 and hist.points[-1].frame_index == 11
    assert hist.n_frames >= 11
    prev, cur = hist.last_two()
    assert (prev.frame_index, cur.frame_index) == (10, 11)


def test_id_survives_short_occlusion():
    tracker = VehicleTracker(TrackerConfig(track_buffer=30))
    seen = []
    for f in range(40):
        box = [200, 100 + 4 * f, 260, 140 + 4 * f]
        visible = not 15 <= f < 22  # 7 frames occluded
        tracks = tracker.update(dets([box]) if visible else Detections.empty(), f, f / 25.0)
        seen.extend(t.track_id for t in tracks)
    assert len(set(seen)) == 1, f"expected one id across the gap, got {sorted(set(seen))}"


def test_empty_detections():
    tracker = VehicleTracker(TrackerConfig())
    assert tracker.update(Detections.empty(), 0, 0.0) == []
    assert tracker.stats() == {"n_tracks": 0}
