"""Sanity checks for the outputs of prepare.py, plus label overlays for visual inspection.

Usage (from the project root):
    python data/tools/check.py              # all checks, 6 overlay images per split
    python data/tools/check.py --draw 12
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

from visdrone import CLASS_NAMES, SPLITS, list_sequences, load_sequence, log, selected_frames

DATA_DIR = Path(__file__).resolve().parents[1]
RAW = DATA_DIR / "raw"
OUT = DATA_DIR / "processed"
RFDETR_SPLIT = {"train": "train", "val": "valid", "test-dev": "test"}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
          "#17becf"]

failures = 0


def expect(ok: bool, msg: str) -> None:
    global failures
    log(("OK    " if ok else "FAIL  ") + msg, 1)
    failures += not ok


def check_split(split: str) -> None:
    log(f"Checking {split}")
    seqs = [load_sequence(RAW, split, n) for n in list_sequences(RAW, split)]
    n_frames = sum(len(selected_frames(s)) for s in seqs)
    n_boxes = sum(sum(b.frame in set(selected_frames(s)) for b in s.objects) for s in seqs)
    n_all_boxes = sum(len(s.objects) for s in seqs)

    lines = (DATA_DIR / "splits" / f"{split}.txt").read_text().split()
    expect(len(lines) == n_frames, f"splits/{split}.txt has {len(lines):,} frames (expected {n_frames:,})")

    coco = json.load(open(OUT / "coco" / f"{split}.json"))
    expect(len(coco["images"]) == n_frames, f"coco images {len(coco['images']):,} == {n_frames:,}")
    expect(len(coco["annotations"]) == n_boxes, f"coco boxes {len(coco['annotations']):,} == {n_boxes:,}")
    expect({a["category_id"] for a in coco["annotations"]} <= set(CLASS_NAMES), "coco category ids within 1-10")
    sizes = {i["id"]: (i["width"], i["height"]) for i in coco["images"]}
    inside = all(a["bbox"][0] >= 0 and a["bbox"][1] >= 0 and a["bbox"][0] + a["bbox"][2] <= sizes[a["image_id"]][0]
                 + 1e-6 and a["bbox"][1] + a["bbox"][3] <= sizes[a["image_id"]][1] + 1e-6 for a in coco["annotations"])
    expect(inside, "all coco boxes inside their image")
    missing = [i["file_name"] for i in coco["images"] if not (RAW / i["file_name"]).exists()]
    expect(not missing, f"coco file_name resolves under data/raw ({len(missing)} missing)")

    rd = OUT / "rfdetr" / RFDETR_SPLIT[split]
    rf = json.load(open(rd / "_annotations.coco.json"))
    expect(len(rf["images"]) == n_frames and len(rf["annotations"]) == n_boxes,
           f"rfdetr/{RFDETR_SPLIT[split]} matches coco counts")
    broken = [i["file_name"] for i in rf["images"] if not (rd / i["file_name"]).exists()]
    expect(not broken, f"rfdetr symlinks resolve ({len(broken)} broken)")

    yd = OUT / "yolo" / split
    imgs, lbls = sorted((yd / "images").iterdir()), sorted((yd / "labels").iterdir())
    expect(len(imgs) == len(lbls) == n_frames, f"yolo images {len(imgs):,} == labels {len(lbls):,} == {n_frames:,}")
    rows = [r.split() for p in lbls for r in p.read_text().splitlines()]
    expect(len(rows) == n_boxes, f"yolo boxes {len(rows):,} == {n_boxes:,}")
    expect(all(0 <= int(r[0]) <= 9 and all(0 <= float(v) <= 1 for v in r[1:]) for r in rows),
           "yolo class index in 0-9 and coordinates in [0, 1]")

    md = OUT / "mot" / split
    n_gt = sum(len((md / s.name / "gt" / "gt.txt").read_text().splitlines()) for s in seqs)
    expect(n_gt == n_all_boxes, f"mot gt boxes {n_gt:,} == {n_all_boxes:,} (all frames)")


def draw(split: str, n: int) -> None:
    rd = OUT / "rfdetr" / RFDETR_SPLIT[split]
    rf = json.load(open(rd / "_annotations.coco.json"))
    by_img: dict[int, list] = {}
    for a in rf["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    out = OUT / "check"
    out.mkdir(parents=True, exist_ok=True)
    for img in random.Random(0).sample(rf["images"], n):
        im = Image.open(rd / img["file_name"]).convert("RGB")
        d = ImageDraw.Draw(im)
        for a in by_img.get(img["id"], []):
            x, y, w, h = a["bbox"]
            c = COLORS[a["category_id"] - 1]
            d.rectangle([x, y, x + w, y + h], outline=c, width=2)
            d.text((x, max(0, y - 11)), CLASS_NAMES[a["category_id"]], fill=c)
        im.save(out / f"{split}_{img['file_name']}", quality=90)
    log(f"Saved {n} overlays to {out.relative_to(DATA_DIR.parent)}/{split}_*.jpg", 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    ap.add_argument("--draw", type=int, default=6, help="overlay images per split (0 to skip)")
    args = ap.parse_args()
    for split in args.splits:
        check_split(split)
        if args.draw:
            draw(split, args.draw)
    log("All checks passed" if failures == 0 else f"{failures} check(s) failed")
    raise SystemExit(failures > 0)


if __name__ == "__main__":
    main()
