"""Shared constants and parsers for VisDrone2019-MOT and the VisDrone2019-DET train set.

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
MOTDET_TRAIN_STRIDE = 6                  # motdet variant: fewer redundant MOT frames, all DET images (~1.5 h training)
FRAME_RATE = 30                          # VisDrone does not ship FPS; 30 is the common assumption

DET_TRAIN = "VisDrone2019-DET-train"     # extra training images (Task 1), see det_overlap.py
DET_EXCLUDED = "det_train_excluded.txt"  # in data/splits/: DET images removed to avoid leakage


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


def selected_frames(seq: Sequence, stride: int = TRAIN_STRIDE) -> list[int]:
    """Frames used for detector training/evaluation: every `stride`-th frame on train, all frames elsewhere."""
    frames = range(1, seq.n_frames + 1)
    if seq.split == "train":
        return [f for f in frames if (f - 1) % stride == 0]
    return list(frames)


@dataclass
class Item:
    """One detector image (a MOT frame or a DET image) with its cleaned labels."""
    name: str                   # flat unique file name used by the RF-DETR and YOLO layouts
    path: Path
    width: int
    height: int
    objects: list[Box]
    ignored: list[Box]
    source: str                 # "mot" or "det"
    seq: str | None = None
    frame: int | None = None


def mot_items(seq: Sequence, stride: int = TRAIN_STRIDE) -> list[Item]:
    by_frame: dict[int, list[Box]] = {}
    ign_frame: dict[int, list[Box]] = {}
    for b in seq.objects:
        by_frame.setdefault(b.frame, []).append(b)
    for b in seq.ignored:
        ign_frame.setdefault(b.frame, []).append(b)
    return [Item(f"{seq.name}_{f:07d}.jpg", seq.image_path(f), seq.width, seq.height, by_frame.get(f, []),
                 ign_frame.get(f, []), "mot", seq.name, f) for f in selected_frames(seq, stride)]


def video_id(name: str) -> str:
    """Video id shared by MOT sequences (uav0000086_00000_v) and DET images (0000086_00118_d_0000001)."""
    return name.split("_")[0].replace("uav", "")


def det_dir(raw_root: Path, split: str = "train") -> Path:
    return raw_root / f"VisDrone2019-DET-{split}"


def load_det_image(raw_root: Path, name: str, split: str = "train") -> Item:
    """DET labels: x,y,w,h,score,class,truncation,occlusion (no frame or track id)."""
    path = det_dir(raw_root, split) / "images" / f"{name}.jpg"
    with Image.open(path) as im:
        width, height = im.size
    objects, ignored = [], []
    for line in open(det_dir(raw_root, split) / "annotations" / f"{name}.txt"):
        x, y, w, h, s, c, t, o = [int(v) for v in line.strip().rstrip(",").split(",")[:8]]
        box = clip_box(Box(0, -1, x, y, w, h, s, c, t, o), width, height)
        if box is None:
            continue
        if box.cls == IGNORED_CLASS:
            ignored.append(box)
        elif box.cls in CLASS_NAMES:
            objects.append(box)
    return Item(f"det_{name}.jpg", path, width, height, objects, ignored, "det")


def list_det_images(raw_root: Path, split: str = "train") -> list[str]:
    return sorted(p.stem for p in (det_dir(raw_root, split) / "annotations").glob("*.txt"))
