"""Library versions of ByteTrack, OC-SORT and BoT-SORT, all from boxmot.

boxmot separates classes itself, and for BoT-SORT computes camera motion once per frame, so one instance serves all
classes. Its ByteTrack has a single threshold for matching and for new tracks, hence no new_track_thresh here.
"""
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ..base import TRACK_COLS, AppearanceEncoder, FrameContext, FrontEnd, TrackerConfig, TrackingAlgorithm
from ..registry import register
from .botsort import BotSortConfig
from .bytetrack import SECOND_MATCH_THRESH, UNCONFIRMED_MATCH_THRESH
from .ocsort import OcSortConfig

# boxmot reports a track from its first activated frame. min_hits only filters the output, which the original
# ByteTrack and BoT-SORT do not do.
NO_OUTPUT_FILTER = 1


class BoxmotAdapter(TrackingAlgorithm):
    handles_classes = True
    tracker_name: ClassVar[str]

    def __init__(self, cfg: TrackerConfig):
        super().__init__(cfg)
        from boxmot import create_tracker
        self._tracker = create_tracker(self.tracker_name, config=self._native_config(cfg), per_class=True)

    @abstractmethod
    def _native_config(self, cfg: TrackerConfig):
        """Translate our parameters to the boxmot config of this tracker."""

    def step(self, dets: np.ndarray, ctx: FrameContext) -> np.ndarray:
        return np.asarray(self._tracker.update(dets.astype(np.float32), ctx.image)[:, :7], dtype=np.float64)


@dataclass(frozen=True)
class ByteTrackLibConfig(TrackerConfig):
    det_thresh: float = 0.5
    track_low_thresh: float = 0.1
    match_thresh: float = 0.8
    track_buffer: int = 30


@register("bytetrack_lib")
class ByteTrackLibrary(BoxmotAdapter):
    config_cls = ByteTrackLibConfig
    tracker_name = "bytetrack"

    def _native_config(self, cfg: ByteTrackLibConfig):
        from boxmot import ByteTrackConfig
        return ByteTrackConfig(track_thresh=cfg.det_thresh, min_conf=cfg.track_low_thresh, match_thresh=cfg.match_thresh,
                               max_lost_frames=int(cfg.frame_rate / 30.0 * cfg.track_buffer), min_hits=NO_OUTPUT_FILTER)


@register("ocsort_lib")
class OcSortLibrary(BoxmotAdapter):
    config_cls = OcSortConfig
    tracker_name = "ocsort"

    def _native_config(self, cfg: OcSortConfig):
        from boxmot import OcSortConfig as NativeConfig
        return NativeConfig(det_thresh=cfg.det_thresh, max_age=cfg.max_age, min_hits=cfg.min_hits,
                            iou_threshold=cfg.iou_threshold, delta_t=cfg.delta_t, inertia=cfg.inertia,
                            asso_func=cfg.asso_func)


@register("botsort_lib")
class BotSortLibrary(BoxmotAdapter):
    """Appearance features come from our FrontEnd, so both BoT-SORT versions use the same encoder."""

    config_cls = BotSortConfig
    tracker_name = "botsort"

    def __init__(self, cfg: BotSortConfig):
        super().__init__(cfg)
        self._frame = 0
        self._width: int | None = None

    @classmethod
    def needs_encoder(cls, cfg: BotSortConfig) -> bool:
        return cfg.with_reid

    @classmethod
    def build_front_end(cls, cfg: BotSortConfig, encoder: AppearanceEncoder | None) -> FrontEnd:
        return FrontEnd(encoder=encoder if cfg.with_reid else None, embed_min_score=cfg.det_thresh, keep_image=True)

    def _native_config(self, cfg: BotSortConfig):
        from boxmot import BotSortConfig as NativeConfig
        return NativeConfig(track_high_thresh=cfg.det_thresh, track_low_thresh=cfg.track_low_thresh,
                            new_track_thresh=cfg.new_track_thresh, match_thresh=cfg.match_thresh,
                            max_lost_frames=int(cfg.frame_rate / 30.0 * cfg.track_buffer),
                            proximity_thresh=cfg.proximity_thresh, appearance_thresh=cfg.appearance_thresh,
                            use_cmc=cfg.cmc_method != "none", cmc_method=cfg.cmc_method if cfg.cmc_method != "none" else "sof",
                            use_embeddings=cfg.with_reid, det_thresh=cfg.track_low_thresh, min_hits=NO_OUTPUT_FILTER,
                            second_match_thresh=SECOND_MATCH_THRESH, unconfirmed_match_thresh=UNCONFIRMED_MATCH_THRESH)

    def step(self, dets: np.ndarray, ctx: FrameContext) -> np.ndarray:
        import torch
        from boxmot.structures import Boxes, Detections

        self._frame += 1
        dets, features = self._align_features(dets, ctx.embeddings)
        packed = Detections(geometry=Boxes(torch.from_numpy(np.ascontiguousarray(dets[:, :4], dtype=np.float32))),
                            scores=torch.from_numpy(dets[:, 4].astype(np.float32)),
                            class_ids=torch.from_numpy(dets[:, 5].astype(np.int64)),
                            sample_id=f"frame{self._frame}",
                            embeddings=None if features is None else torch.from_numpy(features))
        rows = self._tracker.update(packed, ctx.image).to_aabb_rows()
        rows = np.asarray(rows, dtype=np.float64)
        return rows[:, :7] if rows.size else np.zeros((0, TRACK_COLS))

    def _align_features(self, dets: np.ndarray, features: np.ndarray | None):
        """boxmot needs one fixed feature width, also on frames where no box was encoded.

        Before the first encoded box no track exists, so low-score boxes have nothing to match and can be dropped.
        """
        if features is None:
            return dets, None
        if features.shape[1] > 0:
            self._width = features.shape[1]
        if len(dets) == 0 or self._width is None:
            return dets[:0], None
        if features.shape[1] == 0:
            features = np.zeros((len(dets), self._width), dtype=np.float32)
        return dets, features
