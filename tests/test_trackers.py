"""Behaviour every tracker must have. New algorithms are picked up automatically through the registry.

Run from the project root: python -m pytest tests
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.trackers import FrameContext, Tracker, TrackerConfig, TrackingAlgorithm, available_trackers, load_config, register
from src.trackers.base import track_rows
from src.trackers.core.association import iou_matrix, linear_assignment
from src.trackers.core.motion import XYAHMotion, XYSRMotion, XYWHMotion
from src.trackers.factory import validate_detections

FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)
FIVE = {"sort", "deepsort", "bytetrack", "ocsort", "botsort"}


class FakeEncoder:
    """The same object always gets the same feature: the feature depends only on the box height."""

    def encode(self, frame, boxes):
        h = boxes[:, 3] - boxes[:, 1]
        features = np.c_[np.sin(h / 7.0), np.cos(h / 7.0), np.ones(len(boxes))]
        return features / np.linalg.norm(features, axis=1, keepdims=True)


def make(name, **overrides):
    return Tracker(name, load_config(name, **overrides), encoder=FakeEncoder())


def box(x, y, w, h, score=0.9, cls=4):
    return [x, y, x + w, y + h, score, cls]


def two_objects(t):
    return np.array([box(100 + 4 * t, 100, 40, 80), box(700 - 4 * t, 400, 60, 50)])


@pytest.fixture(params=sorted(available_trackers()))
def name(request):
    return request.param


def test_the_five_planned_trackers_exist():
    assert FIVE <= set(available_trackers())


def test_output_contract(name):
    tracker = make(name)
    for t in range(10):
        out = tracker.update(two_objects(t), FRAME)
        assert out.ndim == 2 and out.shape[1] == 7
        assert len(set(out[:, 4])) == len(out)
    assert len(out) == 2


def test_ids_stay_stable_on_clean_motion(name):
    tracker = make(name)
    seen = [tracker.update(two_objects(t), FRAME) for t in range(30)]
    assert {len(s) for s in seen[5:]} == {2}
    ids = [tuple(sorted(s[:, 4])) for s in seen[5:]]
    assert len(set(ids)) == 1


def test_empty_frames_are_accepted(name):
    tracker = make(name)
    assert tracker.update(np.zeros((0, 6)), FRAME).shape == (0, 7)
    tracker.update(two_objects(0), FRAME)
    assert tracker.update(np.zeros((0, 6)), FRAME).shape[1] == 7


def test_track_survives_a_short_gap(name):
    tracker = make(name)
    for t in range(10):
        before = tracker.update(two_objects(t), FRAME)
    for t in range(10, 13):
        tracker.update(np.zeros((0, 6)), FRAME)
    after = None
    for t in range(13, 20):
        after = tracker.update(two_objects(t), FRAME)
    assert set(after[:, 4]) == set(before[:, 4])


def test_classes_never_share_a_track(name):
    """Two classes at the same place must give two different tracks."""
    tracker = make(name)
    for t in range(10):
        dets = np.array([box(100 + t, 100, 50, 50, cls=4), box(100 + t, 100, 50, 50, cls=1)])
        out = tracker.update(dets, FRAME)
    assert sorted(out[:, 6]) == [1, 4]
    assert len(set(out[:, 4])) == 2


def test_classes_outside_the_evaluation_are_ignored(name):
    tracker = make(name)
    for t in range(6):
        out = tracker.update(np.array([box(10 + t, 10, 30, 30, cls=10)]), FRAME)
    assert len(out) == 0


def test_runs_are_reproducible(name):
    rng = np.random.default_rng(0)
    frames = [np.c_[rng.uniform(0, 500, (8, 2)), rng.uniform(510, 900, (8, 2)), rng.uniform(0.2, 1, 8), np.full(8, 4)]
              for _ in range(20)]
    first, second = make(name), make(name)
    for dets in frames:
        np.testing.assert_array_equal(first.update(dets, FRAME), second.update(dets, FRAME))


def test_every_yaml_loads_and_accepts_a_new_threshold(name):
    assert load_config(name, det_thresh=0.3).det_thresh == 0.3


def test_unknown_parameter_is_rejected(name):
    with pytest.raises(ValueError):
        load_config(name, not_a_parameter=1)


def test_bytetrack_recovers_a_low_score_box():
    """The core idea of ByteTrack: a box below the high threshold keeps its track alive."""
    tracker = make("bytetrack")
    for t in range(6):
        ids = tracker.update(np.array([box(100 + 2 * t, 100, 40, 80)]))[:, 4]
    out = tracker.update(np.array([box(112, 100, 40, 80, score=0.2)]))
    assert list(out[:, 4]) == list(ids)


def test_sort_drops_a_low_score_box():
    tracker = make("sort")
    for t in range(6):
        tracker.update(np.array([box(100 + 2 * t, 100, 40, 80)]))
    assert len(tracker.update(np.array([box(112, 100, 40, 80, score=0.2)]))) == 0


def test_bytetrack_threshold_moves_new_track_threshold():
    cfg = load_config("bytetrack", det_thresh=0.3)
    assert cfg.new_track_thresh == pytest.approx(0.4)


@pytest.mark.parametrize("name", ["bytetrack", "botsort"])
def test_new_track_threshold_stays_reachable_at_high_det_threshold(name):
    assert load_config(name, det_thresh=0.85).new_track_thresh == pytest.approx(0.95)
    assert load_config(name, det_thresh=0.9).new_track_thresh == pytest.approx(0.95)
    assert load_config(name, det_thresh=0.97).new_track_thresh == pytest.approx(0.97)


def test_image_is_only_required_when_used():
    assert not make("sort").requires_image
    assert not make("bytetrack").requires_image
    assert not make("ocsort").requires_image
    assert make("deepsort").requires_image
    assert make("botsort").requires_image
    with pytest.raises(ValueError):
        make("botsort").update(two_objects(0), None)


def test_config_type_is_checked():
    with pytest.raises(TypeError):
        Tracker("sort", load_config("bytetrack"))


def test_a_new_algorithm_plugs_in_without_touching_other_code():
    @dataclass(frozen=True)
    class EchoConfig(TrackerConfig):
        pass

    @register("echo_for_test")
    class Echo(TrackingAlgorithm):
        config_cls = EchoConfig

        def step(self, dets: np.ndarray, ctx: FrameContext) -> np.ndarray:
            return track_rows([[*d[:4], self._new_id(), d[4], -1] for d in dets])

    out = Tracker("echo_for_test", EchoConfig()).update(two_objects(0))
    assert len(out) == 2 and len(set(out[:, 4])) == 2


@pytest.mark.parametrize("motion", [XYSRMotion(), XYAHMotion(), XYWHMotion()])
def test_motion_models_round_trip_a_box(motion):
    b = np.array([10.0, 20.0, 50.0, 100.0])
    np.testing.assert_allclose(motion.decode(motion.initiate(motion.encode(b))[0]), b, atol=1e-6)


@pytest.mark.parametrize("motion", [XYSRMotion(), XYAHMotion(), XYWHMotion()])
def test_motion_models_learn_a_constant_velocity(motion):
    mean, cov = motion.initiate(motion.encode(np.array([0.0, 0.0, 40.0, 80.0])))
    for t in range(1, 30):
        mean, cov = motion.predict(mean, cov)
        mean, cov = motion.update(mean, cov, motion.encode(np.array([5.0 * t, 0.0, 40.0 + 5.0 * t, 80.0])))
    mean, _ = motion.predict(mean, cov)
    np.testing.assert_allclose(motion.decode(mean), [150.0, 0.0, 190.0, 80.0], atol=2.0)


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10], [100, 100, 110, 110]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10]])
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 1 / 3], [0.0, 0.0]])
    assert iou_matrix(a, np.zeros((0, 4))).shape == (2, 0)


def test_linear_assignment_rejects_expensive_pairs():
    matches, rows, cols = linear_assignment(np.array([[0.1, 0.9], [0.9, 0.95]]), 0.5)
    assert matches == [(0, 0)] and rows == [1] and cols == [1]


def test_invalid_detections():
    with pytest.raises(ValueError):
        validate_detections(np.zeros((3, 5)))
    with pytest.raises(ValueError):
        validate_detections(np.array([[0, 0, 1, 1, np.nan, 1]]))
    assert len(validate_detections(np.array([[5, 5, 5, 9, 0.9, 1]]))) == 0     # zero width
