"""The inspection view must display the same images sent to OCR."""
from __future__ import annotations

import json

import numpy as np

from tracking_app.config import Settings
from tracking_app.ocr.preprocess import create_preprocessor
from tracking_app.storage import SessionStore


def test_inspection_stages_match_ocr_inputs():
    crop = np.arange(40 * 80 * 3, dtype=np.uint8).reshape(40, 80, 3)
    for mode, attempts in (("none", 1), ("clahe", 1), ("auto", 2)):
        settings = Settings.from_dict({"ocr_backend": "none", "ocr_preprocess_mode": mode,
                                       "ocr_upscale": 2, "ocr_sharpen": .5})
        preprocessor = create_preprocessor(settings)
        stages = list(preprocessor.stages(crop))
        images_sent_to_ocr = list(preprocessor.variants(crop))
        assert stages[0].label == "Original crop"
        assert [stage.ocr_attempt for stage in stages if stage.ocr_attempt] == list(range(1, attempts + 1))
        assert [stage.label.split()[0] for stage in stages[:3]] == ["Original", "Resize", "Sharpen"]
        assert len(images_sent_to_ocr) == attempts
        for stage, ocr_input in zip((stage for stage in stages if stage.ocr_attempt), images_sent_to_ocr):
            assert np.array_equal(stage.image, ocr_input)


def test_inspection_uses_saved_session_settings(tmp_path):
    store = SessionStore(tmp_path / "sessions.sqlite3")
    saved = {"ocr_preprocess_mode": "clahe", "ocr_upscale": 3.0}
    store.start("session", "video", "clip.avi", "2026-01-01T00:00:00Z", json.dumps(saved))
    assert store.session_settings("session") == saved
    assert store.session_settings("missing") is None
