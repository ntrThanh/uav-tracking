"""Reading detection caches and writing tracker results."""
from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CACHE_COLS = 8      # frame,-1,x,y,w,h,score,class


@dataclass(frozen=True)
class SequenceInfo:
    name: str
    length: int
    frame_rate: int
    image_dir: Path
    image_ext: str

    def image_path(self, frame: int) -> Path:
        return self.image_dir / f"{frame:07d}{self.image_ext}"


def read_seqinfo(sequence_dir: Path) -> SequenceInfo:
    parser = configparser.ConfigParser()
    if not parser.read(sequence_dir / "seqinfo.ini"):
        raise FileNotFoundError(f"Missing seqinfo.ini in {sequence_dir}")
    s = parser["Sequence"]
    return SequenceInfo(s["name"], int(s["seqLength"]), int(s["frameRate"]), sequence_dir / s["imDir"], s["imExt"])


def read_detections(path: Path) -> dict[int, np.ndarray]:
    """Detection cache -> {frame: (N, 6) x1, y1, x2, y2, score, class}."""
    table = np.loadtxt(path, delimiter=",", ndmin=2)
    if table.size == 0:
        return {}
    if table.shape[1] != CACHE_COLS:
        raise ValueError(f"{path}: expected {CACHE_COLS} columns, got {table.shape[1]}")
    frames = table[:, 0].astype(int)
    x, y, w, h = table[:, 2], table[:, 3], table[:, 4], table[:, 5]
    dets = np.c_[x, y, x + w, y + h, table[:, 6], table[:, 7]]
    return {int(f): dets[frames == f] for f in np.unique(frames)}


def format_tracks(frame: int, tracks: np.ndarray) -> list[str]:
    """MOTChallenge line with the class in column 8: frame,id,x,y,w,h,score,class,-1,-1."""
    lines = []
    for x1, y1, x2, y2, track_id, score, cls in tracks:
        lines.append(f"{frame},{int(track_id)},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},"
                     f"{score:.4f},{int(cls)},-1,-1\n")
    return lines
