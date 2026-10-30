"""Convert VisDrone2019-MOT into the shared formats used by the team.

Usage (from the project root):
    python data/tools/prepare.py all          # splits + coco + rfdetr + yolo + mot
    python data/tools/prepare.py coco rfdetr  # only some targets

Inputs:  data/raw/VisDrone2019-MOT-{train,val,test-dev}/{sequences,annotations}
Outputs: data/splits/ and data/processed/{coco,rfdetr,yolo,mot}/ (see data/README.md)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from visdrone import (CLASS_NAMES, FRAME_RATE, SPLITS, TRAIN_STRIDE, Sequence, list_sequences, load_sequence, log,
                      selected_frames)

DATA_DIR = Path(__file__).resolve().parents[1]
RAW = DATA_DIR / "raw"
SPLIT_DIR = DATA_DIR / "splits"
OUT = DATA_DIR / "processed"
RFDETR_SPLIT = {"train": "train", "val": "valid", "test-dev": "test"}
TARGETS = ["splits", "coco", "rfdetr", "yolo", "mot"]


def image_name(seq: Sequence, frame: int) -> str:
    """Flat, unique file name used by the RF-DETR and YOLO layouts."""
    return f"{seq.name}_{frame:07d}.jpg"


def relink(link: Path, target: Path) -> None:
    """Create a relative symlink (images are never copied)."""
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(target, link.parent))


def reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def load_split(split: str) -> list[Sequence]:
    seqs = [load_sequence(RAW, split, name) for name in list_sequences(RAW, split)]
    n_obj = sum(len(s.objects) for s in seqs)
    log(f"Loaded {split}: {len(seqs)} sequences, {sum(s.n_frames for s in seqs):,} frames, {n_obj:,} boxes", 1)
    return seqs


# ---------------------------------------------------------------- splits
def make_splits(data: dict[str, list[Sequence]]) -> None:
    log("Writing frame lists")
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for split, seqs in data.items():
        lines = [f"{s.name}/{f:07d}" for s in seqs for f in selected_frames(s)]
        (SPLIT_DIR / f"{split}.txt").write_text("\n".join(lines) + "\n")
        note = f" (stride {TRAIN_STRIDE})" if split == "train" else " (all frames)"
        log(f"splits/{split}.txt: {len(lines):,} frames{note}", 1)


# ---------------------------------------------------------------- coco
def build_coco(seqs: list[Sequence], file_name) -> dict:
    """COCO dict for the selected frames. `file_name(seq, frame)` decides the image path."""
    images, annotations, ignored = [], [], []
    by_frame: dict[tuple[str, int], list] = {}
    for s in seqs:
        for b in s.objects:
            by_frame.setdefault((s.name, b.frame), []).append(b)
    ign_frame: dict[tuple[str, int], list] = {}
    for s in seqs:
        for b in s.ignored:
            ign_frame.setdefault((s.name, b.frame), []).append(b)

    for s in seqs:
        for f in selected_frames(s):
            image_id = len(images) + 1
            images.append({"id": image_id, "file_name": file_name(s, f), "width": s.width, "height": s.height,
                           "seq": s.name, "frame": f})
            for b in by_frame.get((s.name, f), []):
                annotations.append({
                    "id": len(annotations) + 1, "image_id": image_id, "category_id": b.cls,
                    "bbox": [b.x, b.y, b.w, b.h], "area": b.w * b.h, "iscrowd": 0,
                    "track_id": b.track_id, "occlusion": b.occ, "truncation": b.trunc,
                })
            for b in ign_frame.get((s.name, f), []):
                ignored.append({"image_id": image_id, "bbox": [b.x, b.y, b.w, b.h]})

    categories = [{"id": k, "name": v, "supercategory": "visdrone"} for k, v in CLASS_NAMES.items()]
    return {"info": {"description": "VisDrone2019-MOT (team shared format)"}, "images": images,
            "annotations": annotations, "categories": categories, "ignored_regions": ignored}


def make_coco(data: dict[str, list[Sequence]]) -> None:
    log("Writing COCO json (file_name relative to data/raw)")
    out = OUT / "coco"
    out.mkdir(parents=True, exist_ok=True)
    for split, seqs in data.items():
        coco = build_coco(seqs, lambda s, f: str(s.image_path(f).relative_to(RAW)))
        path = out / f"{split}.json"
        json.dump(coco, open(path, "w"))
        log(f"coco/{split}.json: {len(coco['images']):,} images, {len(coco['annotations']):,} boxes, "
            f"{len(coco['ignored_regions']):,} ignored regions ({path.stat().st_size / 1e6:.0f} MB)", 1)


# ---------------------------------------------------------------- rf-detr
def make_rfdetr(data: dict[str, list[Sequence]]) -> None:
    log("Writing RF-DETR layout (train/valid/test, images are symlinks)")
    root = OUT / "rfdetr"
    for split, seqs in data.items():
        d = reset_dir(root / RFDETR_SPLIT[split])
        coco = build_coco(seqs, image_name)
        for s in seqs:
            for f in selected_frames(s):
                relink(d / image_name(s, f), s.image_path(f))
        json.dump(coco, open(d / "_annotations.coco.json", "w"))
        log(f"rfdetr/{RFDETR_SPLIT[split]}/: {len(coco['images']):,} images, {len(coco['annotations']):,} boxes", 1)


# ---------------------------------------------------------------- yolo
def make_yolo(data: dict[str, list[Sequence]]) -> None:
    log("Writing YOLO layout (class index = VisDrone id - 1, images are symlinks)")
    root = OUT / "yolo"
    for split, seqs in data.items():
        img_dir = reset_dir(root / split / "images")
        lbl_dir = reset_dir(root / split / "labels")
        n_img = n_box = 0
        for s in seqs:
            frames = set(selected_frames(s))
            lines: dict[int, list[str]] = {f: [] for f in frames}
            for b in s.objects:
                if b.frame in frames:
                    cx, cy = (b.x + b.w / 2) / s.width, (b.y + b.h / 2) / s.height
                    lines[b.frame].append(f"{b.cls - 1} {cx:.6f} {cy:.6f} {b.w / s.width:.6f} {b.h / s.height:.6f}")
            for f in sorted(frames):
                name = image_name(s, f)
                relink(img_dir / name, s.image_path(f))
                (lbl_dir / name.replace(".jpg", ".txt")).write_text("\n".join(lines[f]) + ("\n" if lines[f] else ""))
                n_img += 1
                n_box += len(lines[f])
        log(f"yolo/{split}/: {n_img:,} images, {n_box:,} boxes", 1)

    names = "\n".join(f"  {k - 1}: {v}" for k, v in CLASS_NAMES.items())
    (root / "visdrone.yaml").write_text(
        f"# Generated by data/tools/prepare.py\npath: {root}\ntrain: train/images\nval: val/images\n"
        f"test: test-dev/images\nnames:\n{names}\n")
    log("yolo/visdrone.yaml", 1)


# ---------------------------------------------------------------- mot
def make_mot(data: dict[str, list[Sequence]]) -> None:
    log("Writing MOTChallenge layout (all frames, for tracking evaluation)")
    root = OUT / "mot"
    for split, seqs in data.items():
        reset_dir(root / split)
        for s in seqs:
            d = root / split / s.name
            (d / "gt").mkdir(parents=True)
            relink(d / "img1", s.image_dir)
            gt = sorted(s.objects, key=lambda b: (b.frame, b.track_id))
            (d / "gt" / "gt.txt").write_text("".join(
                f"{b.frame},{b.track_id},{b.x:.2f},{b.y:.2f},{b.w:.2f},{b.h:.2f},1,{b.cls},{b.occ}\n" for b in gt))
            (d / "gt" / "ignored.txt").write_text("".join(
                f"{b.frame},-1,{b.x:.2f},{b.y:.2f},{b.w:.2f},{b.h:.2f},0,0,0\n" for b in s.ignored))
            (d / "seqinfo.ini").write_text(
                f"[Sequence]\nname={s.name}\nimDir=img1\nframeRate={FRAME_RATE}\nseqLength={s.n_frames}\n"
                f"imWidth={s.width}\nimHeight={s.height}\nimExt=.jpg\n")
        (root / f"{split}_seqmap.txt").write_text("name\n" + "".join(f"{s.name}\n" for s in seqs))
        log(f"mot/{split}/: {len(seqs)} sequences, {sum(len(s.objects) for s in seqs):,} boxes", 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="+", choices=TARGETS + ["all"])
    ap.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    args = ap.parse_args()
    targets = TARGETS if "all" in args.targets else args.targets

    t0 = time.time()
    log(f"Raw data: {RAW}")
    data = {split: load_split(split) for split in args.splits}
    steps = {"splits": make_splits, "coco": make_coco, "rfdetr": make_rfdetr, "yolo": make_yolo, "mot": make_mot}
    for t in targets:
        steps[t](data)
    log(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
