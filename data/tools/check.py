"""Sanity checks for the outputs of prepare.py, plus label overlays for visual inspection.

Usage (from the project root):
    python data/tools/check.py              # all checks, every available variant, 6 overlay images per split
    python data/tools/check.py --variants mot --draw 12
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

from prepare import RAW, RFDETR_SPLIT, VARIANTS, available_variants, build_items, load_det, load_mot, suffix
from visdrone import CLASS_NAMES, SPLITS, list_sequences, log, video_id

DATA_DIR = Path(__file__).resolve().parents[1]
OUT = DATA_DIR / "processed"
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
          "#17becf"]

failures = 0


def expect(ok: bool, msg: str) -> None:
    global failures
    log(("OK    " if ok else "FAIL  ") + msg, 1)
    failures += not ok


def check_variant(items: dict, variant: str) -> None:
    sfx = suffix(variant)
    for split, its in items.items():
        log(f"Checking {variant} / {split}")
        n_img, n_box = len(its), sum(len(it.objects) for it in its)

        own_list = split == "train" or variant == "det"
        split_file = DATA_DIR / "splits" / (f"{split}{sfx}.txt" if own_list else f"{split}.txt")
        lines = split_file.read_text().split()
        expect(len(lines) == n_img, f"{split_file.name} has {len(lines):,} images (expected {n_img:,})")

        coco = json.load(open(OUT / f"coco{sfx}" / f"{split}.json"))
        expect(len(coco["images"]) == n_img, f"coco images {len(coco['images']):,} == {n_img:,}")
        expect(len(coco["annotations"]) == n_box, f"coco boxes {len(coco['annotations']):,} == {n_box:,}")
        expect({a["category_id"] for a in coco["annotations"]} <= set(CLASS_NAMES), "coco category ids within 1-10")
        sizes = {i["id"]: (i["width"], i["height"]) for i in coco["images"]}
        inside = all(a["bbox"][0] >= 0 and a["bbox"][1] >= 0
                     and a["bbox"][0] + a["bbox"][2] <= sizes[a["image_id"]][0] + 1e-6
                     and a["bbox"][1] + a["bbox"][3] <= sizes[a["image_id"]][1] + 1e-6 for a in coco["annotations"])
        expect(inside, "all coco boxes inside their image")
        missing = [i["file_name"] for i in coco["images"] if not (RAW / i["file_name"]).exists()]
        expect(not missing, f"coco file_name resolves under data/raw ({len(missing)} missing)")

        rd = OUT / f"rfdetr{sfx}" / RFDETR_SPLIT[split]
        rf = json.load(open(rd / "_annotations.coco.json"))
        expect(len(rf["images"]) == n_img and len(rf["annotations"]) == n_box, f"{rd.parent.name} matches counts")
        broken = [i["file_name"] for i in rf["images"] if not (rd / i["file_name"]).exists()]
        expect(not broken, f"{rd.parent.name} symlinks resolve ({len(broken)} broken)")

        yd = OUT / f"yolo{sfx}" / split
        imgs, lbls = sorted((yd / "images").iterdir()), sorted((yd / "labels").iterdir())
        expect(len(imgs) == len(lbls) == n_img, f"yolo images {len(imgs):,} == labels {len(lbls):,} == {n_img:,}")
        rows = [r.split() for p in lbls for r in p.read_text().splitlines()]
        expect(len(rows) == n_box, f"yolo boxes {len(rows):,} == {n_box:,}")
        expect(all(0 <= int(r[0]) <= 9 and all(0 <= float(v) <= 1 for v in r[1:]) for r in rows),
               "yolo class index in 0-9 and coordinates in [0, 1]")

        if split == "train" and variant in ("motdet", "det"):
            held_out = {video_id(s) for sp in ("val", "test-dev") for s in list_sequences(RAW, sp)}
            leaked = [it.name for it in its if it.source == "det" and video_id(it.path.stem) in held_out]
            expect(not leaked, f"no DET image from a MOT val/test-dev video in train ({len(leaked)} found)")


def check_mot(mot: dict) -> None:
    log("Checking MOT tracking ground truth")
    for split, seqs in mot.items():
        md = OUT / "mot" / split
        n_gt = sum(len((md / s.name / "gt" / "gt.txt").read_text().splitlines()) for s in seqs)
        n_all = sum(len(s.objects) for s in seqs)
        expect(n_gt == n_all, f"mot/{split} gt boxes {n_gt:,} == {n_all:,} (all frames)")


def draw(variant: str, split: str, n: int) -> None:
    rd = OUT / f"rfdetr{suffix(variant)}" / RFDETR_SPLIT[split]
    rf = json.load(open(rd / "_annotations.coco.json"))
    by_img: dict[int, list] = {}
    for a in rf["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    out = OUT / "check"
    out.mkdir(parents=True, exist_ok=True)
    pool = rf["images"]
    if variant == "motdet" and split == "train":
        pool = [i for i in pool if i["source"] == "det"]       # MOT frames are already drawn by the mot variant
    for img in random.Random(0).sample(pool, n):
        im = Image.open(rd / img["file_name"]).convert("RGB")
        d = ImageDraw.Draw(im)
        for a in by_img.get(img["id"], []):
            x, y, w, h = a["bbox"]
            c = COLORS[a["category_id"] - 1]
            d.rectangle([x, y, x + w, y + h], outline=c, width=2)
            d.text((x, max(0, y - 11)), CLASS_NAMES[a["category_id"]], fill=c)
        im.save(out / f"{variant}_{split}_{img['file_name']}", quality=90)
    log(f"Saved {n} overlays to {out.relative_to(DATA_DIR.parent)}/{variant}_{split}_*.jpg", 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    ap.add_argument("--variants", nargs="+", choices=VARIANTS)
    ap.add_argument("--draw", type=int, default=6, help="overlay images per split (0 to skip)")
    args = ap.parse_args()
    variants = args.variants or available_variants()

    mot = load_mot(args.splits)
    det = load_det(args.splits) if {"motdet", "det"} & set(variants) else None
    for variant in variants:
        items = build_items(mot, det, variant)
        if variant == "motdet":
            items = {k: v for k, v in items.items() if k == "train"}   # val/test-dev are shared with mot
        check_variant(items, variant)
        if args.draw:
            for split in items:
                draw(variant, split, args.draw)
    check_mot(mot)
    log("All checks passed" if failures == 0 else f"{failures} check(s) failed")
    raise SystemExit(failures > 0)


if __name__ == "__main__":
    main()
