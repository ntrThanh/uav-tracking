"""BoT-SORT (Aharon et al., 2022): ByteTrack plus camera motion compensation and appearance matching.

Only the hooks of ByteTrack change; the two-stage association itself is inherited.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..base import AppearanceEncoder, FrameContext, FrontEnd
from ..camera_motion import SparseOpticalFlow, warp_gaussian
from ..core.association import cosine_distance, fuse_score, iou_matrix
from ..core.motion import XYWHMotion
from ..core.track import boxes_of
from ..registry import register
from .bytetrack import ByteTrack, ByteTrackConfig, ByteTrackItem

FEATURE_MOMENTUM = 0.9


@dataclass(frozen=True)
class BotSortConfig(ByteTrackConfig):
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.25
    with_reid: bool = True
    cmc_method: str = "sof"         # "sof" or "none"
    cmc_downscale: int = 2
    reid_weights: str = "weights/pretrained/osnet_x0_25_msmt17.pt"


class BotTrack(ByteTrackItem):
    FROZEN_VELOCITIES = (6, 7)

    def __init__(self, motion, det, track_id, frame_id, feature: np.ndarray | None):
        super().__init__(motion, det, track_id, frame_id)
        self.feature = None
        self.remember(feature)

    def remember(self, feature: np.ndarray | None) -> None:
        """Exponential moving average of the appearance, skipped for boxes that were not encoded."""
        if feature is None:
            return
        if self.feature is not None:
            feature = FEATURE_MOMENTUM * self.feature + (1 - FEATURE_MOMENTUM) * feature
        self.feature = feature / max(np.linalg.norm(feature), 1e-12)


@register("botsort")
class BotSort(ByteTrack):
    config_cls = BotSortConfig
    motion_cls = XYWHMotion

    @classmethod
    def needs_encoder(cls, cfg: BotSortConfig) -> bool:
        return cfg.with_reid

    @classmethod
    def build_front_end(cls, cfg: BotSortConfig, encoder: AppearanceEncoder | None) -> FrontEnd:
        motion = SparseOpticalFlow(cfg.cmc_downscale) if cfg.cmc_method == "sof" else None
        if cfg.cmc_method not in ("sof", "none"):
            raise ValueError(f"Unknown cmc_method: {cfg.cmc_method}")
        return FrontEnd(encoder=encoder if cfg.with_reid else None, motion=motion, embed_min_score=cfg.det_thresh)

    def _create(self, det, feature) -> BotTrack:
        return BotTrack(self.motion, det, self._new_id(), self.frame_id, feature)

    def _observe(self, track: BotTrack, det, feature) -> None:
        super()._observe(track, det, feature)
        track.remember(feature)

    def _compensate(self, tracks, ctx: FrameContext) -> None:
        if ctx.warp is None:
            return
        for track in tracks:
            track.mean, track.cov = warp_gaussian(track.mean, track.cov, ctx.warp)

    def _high_cost(self, tracks, dets, features) -> np.ndarray:
        iou_cost = 1.0 - iou_matrix(boxes_of(tracks), dets[:, :4])
        too_far = iou_cost > self.cfg.proximity_thresh
        fused = fuse_score(iou_cost, dets[:, 4])
        if features is None or not self.cfg.with_reid or iou_cost.size == 0:
            return fused
        appearance = self._appearance_cost(tracks, features) / 2.0
        appearance[appearance > self.cfg.appearance_thresh] = 1.0
        appearance[too_far] = 1.0
        return np.minimum(fused, appearance)

    @staticmethod
    def _appearance_cost(tracks, features: np.ndarray) -> np.ndarray:
        cost = np.ones((len(tracks), len(features)))
        known = [i for i, t in enumerate(tracks) if t.feature is not None]
        if known:
            cost[known] = cosine_distance(np.array([tracks[i].feature for i in known]), features)
        return cost
