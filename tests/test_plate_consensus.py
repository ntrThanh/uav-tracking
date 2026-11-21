"""Online OCR decisions stay stable while allowing later corrections."""
from __future__ import annotations

import json

import cv2
import numpy as np

from tracking_app.config import Settings
from tracking_app.engine.schema import Track, TrackingResult
from tracking_app.ocr.consensus import PlateConsensus
from tracking_app.pipeline import PlateRead
from tracking_app.runner import SessionRunner
from tracking_app.storage import SessionStore


def test_one_wrong_read_does_not_replace_an_accepted_plate():
    consensus = PlateConsensus()
    assert consensus.observe("59A12345", .82, 0).text == "59A12345"
    assert consensus.observe("59A12845", .98, 5).text == "59A12345"
    assert consensus.observe("59A12345", .75, 10).text == "59A12345"
    assert consensus.observe("", 0, 15).text == "59A12345"


def test_repeated_lower_confidence_reads_correct_an_early_error():
    consensus = PlateConsensus()
    consensus.observe("59A12845", .98, 0)
    assert consensus.observe("59A12345", .65, 5).text == "59A12845"
    assert consensus.observe("59A12345", .68, 10).text == "59A12845"
    selected = consensus.observe("59A12345", .70, 15)
    assert (selected.text, selected.confidence, selected.frame_id) == ("59A12345", .70, 15)
    assert consensus.observe("59A12845", .99, 20).text == "59A12345"


def test_two_plate_regions_in_one_frame_count_once():
    consensus = PlateConsensus()
    consensus.observe("59A12345", .70, 0)
    consensus.observe("59A12845", .90, 5)
    consensus.observe("59A12845", .95, 5)
    consensus.observe("59A12845", .96, 10)
    assert consensus.selection.text == "59A12345"
    assert len(consensus._reads) == 3


def test_runner_persists_online_correction_and_matching_evidence(tmp_path):
    video = tmp_path / "plates.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48))
    assert writer.isOpened()
    for value in (20, 40, 60, 80, 100):
        writer.write(np.full((48, 64, 3), value, np.uint8))
    writer.release()

    readings = (("59A12845", .98), ("59A12345", .65), ("59A12345", .68),
                ("59A12345", .70), ("59A12845", .99))

    class FakePipeline:
        last_timings = ()

        def __enter__(self): return self
        def __exit__(self, *args): pass

        def process(self, frame):
            track = Track(1, (10, 10, 40, 35), .9, 4, "car", "active", frame.meta.timestamp)
            text, confidence = readings[frame.meta.frame_id]
            return TrackingResult(frame.meta, (track,), "fake"), (
                PlateRead(1, text, confidence, (12, 20, 30, 29), .9),)

    store = SessionStore(tmp_path / "sessions.sqlite3")
    store.start("test", "video", "plates.avi", "2026-01-01T00:00:00Z", json.dumps({}))
    runner = SessionRunner(store, pipeline_factory=lambda settings: FakePipeline())
    runner._run("test", "video", str(video), Settings.from_dict({"ocr_backend": "none"}))

    assert runner.snapshot()["status"] == "completed"
    assert store.tracks("test")[0]["plate"] == "59A12345"
    assert store.evidence("test", 1)["plate"]["frame_id"] == 3
