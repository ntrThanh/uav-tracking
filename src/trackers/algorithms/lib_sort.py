"""Library version of SORT: the reference code of Bewley, packaged as sort-tracker-py."""
from __future__ import annotations

import numpy as np

from ..base import FrameContext, TrackingAlgorithm, track_rows
from ..registry import register
from .sort import SortConfig


@register("sort_lib")
class SortLibrary(TrackingAlgorithm):
    config_cls = SortConfig

    def __init__(self, cfg: SortConfig):
        super().__init__(cfg)
        from sort_tracker import Sort
        self._tracker = Sort(max_age=cfg.max_age, min_hits=cfg.min_hits, iou_threshold=cfg.iou_threshold)

    def step(self, dets: np.ndarray, ctx: FrameContext) -> np.ndarray:
        kept = dets[dets[:, 4] >= self.cfg.det_thresh, :5]
        return track_rows([[x1, y1, x2, y2, track_id, score, -1]
                           for x1, y1, x2, y2, score, track_id in self._tracker.update(kept)])
