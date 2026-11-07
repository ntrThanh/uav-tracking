"""The public entry point: Tracker(name, cfg).update(dets, frame) -> (M, 7).

Tracker splits a frame by class, runs one independent algorithm instance per class and gives every track a number
that is unique in the sequence. Create one Tracker per sequence.
"""
from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from .base import DET_COLS, TRACK_COLS, AppearanceEncoder, FrameContext, TrackerConfig, TrackingAlgorithm
from .registry import available_trackers, get_algorithm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data" / "tools"))
from visdrone import TRACK_CLASSES  # noqa: E402

CONFIG_DIR = ROOT / "src" / "configs" / "trackers"


def load_config(name: str, path: Path | None = None, **overrides) -> TrackerConfig:
    """Read the yaml of a tracker. det_thresh goes through with_det_thresh so dependent fields follow it."""
    values = yaml.safe_load(Path(path or CONFIG_DIR / f"{name}.yaml").read_text()) or {}
    det_thresh = overrides.pop("det_thresh", None)
    cfg = get_algorithm(name).config_cls.from_dict({**values, **overrides})
    return cfg if det_thresh is None else cfg.with_det_thresh(det_thresh)


def default_encoder(cfg: TrackerConfig) -> AppearanceEncoder:
    from .appearance import OsnetEncoder
    return OsnetEncoder(cfg.reid_weights)


def validate_detections(dets: np.ndarray) -> np.ndarray:
    dets = np.asarray(dets, dtype=np.float64)
    if dets.size == 0:
        return np.zeros((0, DET_COLS))
    if dets.ndim != 2 or dets.shape[1] != DET_COLS:
        raise ValueError(f"Detections must be (N, {DET_COLS}), got {dets.shape}")
    if not np.isfinite(dets).all():
        raise ValueError("Detections contain NaN or infinite values")
    return dets[(dets[:, 2] > dets[:, 0]) & (dets[:, 3] > dets[:, 1])]


class Tracker:
    def __init__(self, name: str, cfg: TrackerConfig | None = None, classes: tuple[int, ...] = tuple(TRACK_CLASSES),
                 encoder: AppearanceEncoder | None = None):
        algorithm = get_algorithm(name)
        self.name = name
        self.cfg = cfg if cfg is not None else load_config(name)
        if not isinstance(self.cfg, algorithm.config_cls):
            raise TypeError(f"{name} expects {algorithm.config_cls.__name__}, got {type(self.cfg).__name__}")
        if encoder is None and algorithm.needs_encoder(self.cfg):
            encoder = default_encoder(self.cfg)
        self.front_end = algorithm.build_front_end(self.cfg, encoder)
        self.classes = tuple(classes)
        self._engines = ({None: algorithm(self.cfg)} if algorithm.handles_classes
                         else {c: algorithm(self.cfg) for c in self.classes})
        self._ids: dict[tuple[int, int], int] = {}
        self.seconds = {"front_end": 0.0, "association": 0.0}

    @property
    def requires_image(self) -> bool:
        return self.front_end.requires_image

    def config_dict(self) -> dict:
        return asdict(self.cfg)

    def update(self, dets: np.ndarray, frame: np.ndarray | None = None) -> np.ndarray:
        dets = validate_detections(dets)
        dets = dets[np.isin(dets[:, 5].astype(int), self.classes)]

        t0 = time.perf_counter()
        ctx = self.front_end.analyze(dets, frame)
        t1 = time.perf_counter()
        out = [self._step(c, engine, dets, ctx) for c, engine in self._engines.items()]  # empty classes still age
        self.seconds["front_end"] += t1 - t0
        self.seconds["association"] += time.perf_counter() - t1
        return np.concatenate(out) if out else np.zeros((0, TRACK_COLS))

    def _step(self, cls: int | None, engine: TrackingAlgorithm, dets: np.ndarray, ctx: FrameContext) -> np.ndarray:
        mask = np.ones(len(dets), dtype=bool) if cls is None else dets[:, 5].astype(int) == cls
        tracks = engine.step(dets[mask], ctx.subset(mask)).copy()
        if cls is not None:
            tracks[:, 6] = cls
        tracks[:, 4] = [self._global_id(int(c), int(i)) for i, c in zip(tracks[:, 4], tracks[:, 6])]
        return tracks

    def _global_id(self, cls: int, local_id: int) -> int:
        return self._ids.setdefault((cls, local_id), len(self._ids) + 1)


__all__ = ["Tracker", "available_trackers", "load_config", "validate_detections"]
