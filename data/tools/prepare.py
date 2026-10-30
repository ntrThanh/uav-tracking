"""Convert VisDrone into the shared formats used by the team.

Two training variants share the same val and test-dev (MOT):
    mot     train = VisDrone2019-MOT train, stride 2                 -> processed/{coco,rfdetr,yolo}/
    motdet  train = mot + VisDrone2019-DET train minus leaked images -> processed/{coco,rfdetr,yolo}_motdet/

Usage (from the project root):
    python data/tools/prepare.py all                       # every target, both variants (motdet if DET exists)
    python data/tools/prepare.py rfdetr --variants motdet  # only some targets / variants

Inputs:  data/raw/VisDrone2019-MOT-{train,val,test-dev}/, data/raw/VisDrone2019-DET-train/ (optional)
         data/splits/det_train_excluded.txt (from det_overlap.py, required for motdet)
Outputs: data/splits/ and data/processed/ (see data/README.md)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

from visdrone import (CLASS_NAMES, DET_EXCLUDED, DET_TRAIN, FRAME_RATE, SPLITS, TRAIN_STRIDE, Item, Sequence,
                      list_det_images, list_sequences, load_det_image, load_sequence, log, mot_items)

DATA_DIR = Path(__file__).resolve().parents[1]
RAW = DATA_DIR / "raw"
SPLIT_DIR = DATA_DIR / "splits"
OUT = DATA_DIR / "processed"
RFDETR_SPLIT = {"train": "train", "val": "valid", "test-dev": "test"}
TARGETS = ["splits", "coco", "rfdetr", "yolo", "mot"]
VARIANTS = ["mot", "motdet"]


def suffix(variant: str) -> str:
    return "" if variant == "mot" else f"_{variant}"


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


# ---------------------------------------------------------------- loading
def load_mot(splits: list[str]) -> dict[str, list[Sequence]]:
    seqs = {}
    for split in splits:
        seqs[split] = [load_sequence(RAW, split, name) for name in list_sequences(RAW, split)]
        n_obj = sum(len(s.objects) for s in seqs[split])
        log(f"MOT {split}: {len(seqs[split])} sequences, {sum(s.n_frames for s in seqs[split]):,} frames, "
            f"{n_obj:,} boxes", 1)
    return seqs


def load_det() -> list[Item]:
    excluded_file = SPLIT_DIR / DET_EXCLUDED
    if not excluded_file.exists():
        raise SystemExit(f"Missing {excluded_file}. Run: python data/tools/det_overlap.py")
    excluded = {line.split("\t")[0] for line in excluded_file.read_text().splitlines() if line}
    names = [n for n in list_det_images(RAW) if n not in excluded]
    items = [load_det_image(RAW, n) for n in names]
    log(f"DET train: {len(items):,} images kept ({len(excluded)} excluded), "
        f"{sum(len(i.objects) for i in items):,} boxes", 1)
    return items


def build_items(mot: dict[str, list[Sequence]], det: list[Item] | None, variant: str) -> dict[str, list[Item]]:
    items = {split: [it for s in seqs for it in mot_items(s)] for split, seqs in mot.items()}
    if variant == "motdet" and "train" in items:
        items["train"] = items["train"] + det
    return items


# ---------------------------------------------------------------- splits
def make_splits(items: dict[str, list[Item]], variant: str) -> None:
    log(f"[{variant}] Writing frame lists")
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for split, its in items.items():
        if variant != "mot" and split != "train":
            continue                       # val and test-dev are identical across variants
        lines = [f"{it.seq}/{it.frame:07d}" if it.source == "mot" else f"det/{it.path.stem}" for it in its]
        name = f"{split}{suffix(variant)}.txt"
        (SPLIT_DIR / name).write_text("\n".join(lines) + "\n")
        src = Counter(it.source for it in its)
        log(f"splits/{name}: {len(lines):,} images ({', '.join(f'{k} {v:,}' for k, v in sorted(src.items()))})", 1)


# ---------------------------------------------------------------- coco
def build_coco(items: list[Item], file_name) -> dict:
    """COCO dict. `file_name(item)` decides the image path stored in the json."""
    images, annotations, ignored = [], [], []
    for it in items:
        image_id = len(images) + 1
        images.append({"id": image_id, "file_name": file_name(it), "width": it.width, "height": it.height,
                       "source": it.source, "seq": it.seq, "frame": it.frame})
        for b in it.objects:
            annotations.append({
                "id": len(annotations) + 1, "image_id": image_id, "category_id": b.cls,
                "bbox": [b.x, b.y, b.w, b.h], "area": b.w * b.h, "iscrowd": 0,
                "track_id": b.track_id, "occlusion": b.occ, "truncation": b.trunc,
            })
        for b in it.ignored:
            ignored.append({"image_id": image_id, "bbox": [b.x, b.y, b.w, b.h]})
    categories = [{"id": k, "name": v, "supercategory": "visdrone"} for k, v in CLASS_NAMES.items()]
    return {"info": {"description": "VisDrone2019 (team shared format)"}, "images": images,
            "annotations": annotations, "categories": categories, "ignored_regions": ignored}


def make_coco(items: dict[str, list[Item]], variant: str) -> None:
    log(f"[{variant}] Writing COCO json (file_name relative to data/raw)")
    out = OUT / f"coco{suffix(variant)}"
    out.mkdir(parents=True, exist_ok=True)
    for split, its in items.items():
        coco = build_coco(its, lambda it: str(it.path.relative_to(RAW)))
        path = out / f"{split}.json"
        json.dump(coco, open(path, "w"))
        log(f"{path.relative_to(OUT)}: {len(coco['images']):,} images, {len(coco['annotations']):,} boxes, "
            f"{len(coco['ignored_regions']):,} ignored regions ({path.stat().st_size / 1e6:.0f} MB)", 1)


# ---------------------------------------------------------------- rf-detr
def make_rfdetr(items: dict[str, list[Item]], variant: str) -> None:
    log(f"[{variant}] Writing RF-DETR layout (train/valid/test, images are symlinks)")
    root = OUT / f"rfdetr{suffix(variant)}"
    for split, its in items.items():
        d = reset_dir(root / RFDETR_SPLIT[split])
        for it in its:
            relink(d / it.name, it.path)
        coco = build_coco(its, lambda it: it.name)
        json.dump(coco, open(d / "_annotations.coco.json", "w"))
        log(f"{root.name}/{RFDETR_SPLIT[split]}/: {len(coco['images']):,} images, "
            f"{len(coco['annotations']):,} boxes", 1)


# ---------------------------------------------------------------- yolo
def make_yolo(items: dict[str, list[Item]], variant: str) -> None:
    log(f"[{variant}] Writing YOLO layout (class index = VisDrone id - 1, images are symlinks)")
    root = OUT / f"yolo{suffix(variant)}"
    for split, its in items.items():
        img_dir = reset_dir(root / split / "images")
        lbl_dir = reset_dir(root / split / "labels")
        n_box = 0
        for it in its:
            lines = [f"{b.cls - 1} {(b.x + b.w / 2) / it.width:.6f} {(b.y + b.h / 2) / it.height:.6f} "
                     f"{b.w / it.width:.6f} {b.h / it.height:.6f}" for b in it.objects]
            relink(img_dir / it.name, it.path)
            (lbl_dir / it.name.replace(".jpg", ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
            n_box += len(lines)
        log(f"{root.name}/{split}/: {len(its):,} images, {n_box:,} boxes", 1)

    names = "\n".join(f"  {k - 1}: {v}" for k, v in CLASS_NAMES.items())
    (root / "visdrone.yaml").write_text(
        f"# Generated by data/tools/prepare.py (variant {variant})\npath: {root}\ntrain: train/images\n"
        f"val: val/images\ntest: test-dev/images\nnames:\n{names}\n")
    log(f"{root.name}/visdrone.yaml", 1)


# ---------------------------------------------------------------- mot (tracking ground truth, MOT only)
def make_mot(mot: dict[str, list[Sequence]]) -> None:
    log("Writing MOTChallenge layout (all frames, for tracking evaluation)")
    root = OUT / "mot"
    for split, seqs in mot.items():
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
    ap.add_argument("--variants", nargs="+", choices=VARIANTS,
                    help=f"default: mot, plus motdet when data/raw/{DET_TRAIN} exists")
    args = ap.parse_args()
    targets = TARGETS if "all" in args.targets else args.targets
    variants = args.variants or (VARIANTS if (RAW / DET_TRAIN).exists() else ["mot"])

    t0 = time.time()
    log(f"Raw data: {RAW}")
    log(f"Variants: {', '.join(variants)} (train stride {TRAIN_STRIDE} on MOT)")
    mot = load_mot(args.splits)
    det = load_det() if "motdet" in variants and "train" in args.splits else None
    steps = {"splits": make_splits, "coco": make_coco, "rfdetr": make_rfdetr, "yolo": make_yolo}
    for variant in variants:
        items = build_items(mot, det, variant)
        for t in targets:
            if t in steps:
                steps[t](items, variant)
    if "mot" in targets:
        make_mot(mot)
    log(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
