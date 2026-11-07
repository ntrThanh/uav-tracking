"""Library version of DeepSORT: deep_sort_realtime, fed with the same OSNet features as our DeepSORT."""
from __future__ import annotations

import numpy as np

from ..base import AppearanceEncoder, FrameContext, FrontEnd, TrackingAlgorithm, track_rows
from ..registry import register
from .deepsort import DeepSortConfig


@register("deepsort_lib")
class DeepSortLibrary(TrackingAlgorithm):
    config_cls = DeepSortConfig

    def __init__(self, cfg: DeepSortConfig):
        super().__init__(cfg)
        from deep_sort_realtime.deepsort_tracker import DeepSort
        self._tracker = DeepSort(max_age=cfg.max_age, n_init=cfg.n_init, max_iou_distance=cfg.max_iou_distance,
                                 max_cosine_distance=cfg.max_cosine_distance, nn_budget=cfg.nn_budget, embedder=None)

    @classmethod
    def needs_encoder(cls, cfg: DeepSortConfig) -> bool:
        return True

    @classmethod
    def build_front_end(cls, cfg: DeepSortConfig, encoder: AppearanceEncoder | None) -> FrontEnd:
        return FrontEnd(encoder=encoder, embed_min_score=np.nextafter(cfg.det_thresh, -1.0))

    def step(self, dets: np.ndarray, ctx: FrameContext) -> np.ndarray:
        keep = dets[:, 4] >= self.cfg.det_thresh
        raw = [([x1, y1, x2 - x1, y2 - y1], score, int(cls)) for x1, y1, x2, y2, score, cls in dets[keep]]
        tracks = self._tracker.update_tracks(raw, embeds=list(ctx.embeddings[keep]))
        return track_rows([[*t.to_ltrb(), int(t.track_id), t.get_det_conf() or 0.0, -1]
                           for t in tracks if t.is_confirmed() and t.time_since_update == 0])
