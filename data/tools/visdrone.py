"""Shared constants and parsers for VisDrone2019-MOT.

Every converter in this folder imports from here so that all team members use
the same class ids, splits and box cleaning rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

SPLITS = ["train", "val", "test-dev"]

# VisDrone category ids. 0 (ignored region) and 11 (others) are never trained on.
CLASS_NAMES = {
    1: "pedestrian", 2: "people", 3: "bicycle", 4: "car", 5: "van",
    6: "truck", 7: "tricycle", 8: "awning-tricycle", 9: "bus", 10: "motor",
}
IGNORED_CLASS = 0
OTHERS_CLASS = 11
TRACK_CLASSES = [1, 4, 5, 6, 9]          # classes evaluated by VisDrone-MOT

TRAIN_STRIDE = 2                         # keep every 2nd train frame (odd frames)
FRAME_RATE = 30                          # VisDrone does not ship FPS; 30 is the common assumption


def log(msg: str, level: int = 0) -> None:
    """Human-readable log: [+] for main items, [-] for details."""
    print("    " * level + ("[+] " if level == 0 else "[-] ") + msg, flush=True)


@dataclass
class Box:
    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    score: int
    cls: int
    trunc: int
    occ: int


@dataclass
class Sequence:
    split: str
    name: str
    image_dir: Path
    n_frames: int
    width: int
    height: int
    objects: list[Box]          # classes 1-10, clipped to the image
    ignored: list[Box]          # class 0 regions, clipped to the image

    def image_path(self, frame: int) -> Path:
        return self.image_dir / f"{frame:07d}.jpg"


def split_dir(raw_root: Path, split: str) -> Path:
    return raw_root / f"VisDrone2019-MOT-{split}"


def clip_box(b: Box, width: int, height: int) -> Box | None:
    """Clip a box to the image; return None if nothing is left."""
    x1, y1 = max(0.0, b.x), max(0.0, b.y)
    x2, y2 = min(float(width), b.x + b.w), min(float(height), b.y + b.h)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return Box(b.frame, b.track_id, x1, y1, x2 - x1, y2 - y1, b.score, b.cls, b.trunc, b.occ)


def load_sequence(raw_root: Path, split: str, name: str) -> Sequence:
    base = split_dir(raw_root, split)
    image_dir = base / "sequences" / name
    frames = sorted(image_dir.glob("*.jpg"))
    with Image.open(frames[0]) as im:
        width, height = im.size

    objects, ignored = [], []
    for line in open(base / "annotations" / f"{name}.txt"):
        v = [int(t) for t in line.strip().split(",")[:10]]
        box = clip_box(Box(*v), width, height)
        if box is None:
            continue
        if box.cls == IGNORED_CLASS:
            ignored.append(box)
        elif box.cls in CLASS_NAMES:
            objects.append(box)
    return Sequence(split, name, image_dir, len(frames), width, height, objects, ignored)


def list_sequences(raw_root: Path, split: str) -> list[str]:
    return sorted(p.stem for p in (split_dir(raw_root, split) / "annotations").glob("*.txt"))


def selected_frames(seq: Sequence) -> list[int]:
    """Frames used for detector training/evaluation: stride 2 on train, all frames elsewhere."""
    frames = range(1, seq.n_frames + 1)
    if seq.split == "train":
        return [f for f in frames if (f - 1) % TRAIN_STRIDE == 0]
    return list(frames)
