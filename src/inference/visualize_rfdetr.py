"""Draw RF-DETR predictions next to the ground truth for a handful of images.

Half of the images are frames from VisDrone2019-MOT test-dev (one frame per sequence), half are VisDrone2019-DET
test-dev images. Each output is a side-by-side JPG: ground truth on the left, prediction on the right.

Usage (from the project root):
    python -m src.inference.visualize_rfdetr --checkpoint runs/det/T3det_rfdetr_m_576_det/checkpoint_best_total.pth \
        --tag T3det_rfdetr_m_576_det --n 20
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data" / "tools"))
from visdrone import CLASS_NAMES  # noqa: E402

SOURCES = {"mot": ROOT / "data/processed/rfdetr/test", "det": ROOT / "data/processed/rfdetr_det/test"}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
          "#17becf"]
PANEL_W = 960


def log(msg: str, level: int = 0) -> None:
    print("    " * level + ("[+] " if level == 0 else "[-] ") + msg, flush=True)


def pick_images(n: int, seed: int) -> list[tuple[str, dict, list[dict]]]:
    """n/2 MOT test-dev frames (middle frame of random sequences) and n/2 random DET test-dev images."""
    rng = random.Random(seed)
    picks = []
    for src, n_src in (("mot", n // 2), ("det", n - n // 2)):
        coco = json.load(open(SOURCES[src] / "_annotations.coco.json"))
        anns: dict[int, list[dict]] = {}
        for a in coco["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)
        if src == "mot":
            by_seq: dict[str, list[dict]] = {}
            for im in coco["images"]:
                by_seq.setdefault(im["seq"], []).append(im)
            seqs = rng.sample(sorted(by_seq), min(n_src, len(by_seq)))
            images = [sorted(by_seq[s], key=lambda x: x["frame"])[len(by_seq[s]) // 2] for s in seqs]
        else:
            images = rng.sample(coco["images"], n_src)
        picks += [(src, im, anns.get(im["id"], [])) for im in images]
    return picks


def draw_panel(image: Image.Image, boxes: list[tuple], title: str, font, small) -> Image.Image:
    """boxes: (x1, y1, x2, y2, class_id, score or None)."""
    scale = PANEL_W / image.width
    panel = image.resize((PANEL_W, round(image.height * scale)))
    d = ImageDraw.Draw(panel)
    for x1, y1, x2, y2, c, s in boxes:
        color = COLORS[c - 1]
        d.rectangle([x1 * scale, y1 * scale, x2 * scale, y2 * scale], outline=color, width=2)
        if s is not None and (x2 - x1) * scale > 18:
            d.text((x1 * scale, max(0, y1 * scale - 10)), f"{s:.2f}", fill=color, font=small)
    d.rectangle([0, 0, PANEL_W, 26], fill=(0, 0, 0))
    d.text((6, 4), title, fill="white", font=font)
    return panel


def legend(width: int, font) -> Image.Image:
    bar = Image.new("RGB", (width, 26), "white")
    d = ImageDraw.Draw(bar)
    x = 6
    for c, name in CLASS_NAMES.items():
        d.rectangle([x, 8, x + 12, 20], fill=COLORS[c - 1])
        d.text((x + 16, 5), name, fill="black", font=font)
        x += 22 + int(d.textlength(name, font=font)) + 10
    return bar


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="RFDETRMedium")
    ap.add_argument("--resolution", type=int, default=576)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import rfdetr
    model = getattr(rfdetr, args.model)(pretrain_weights=str(args.checkpoint), num_classes=len(CLASS_NAMES),
                                        resolution=args.resolution)
    out = ROOT / "outputs" / "inference" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    font, small = ImageFont.load_default(size=16), ImageFont.load_default(size=10)
    log(f"Checkpoint {args.checkpoint}, threshold {args.threshold}, {args.n} images -> {out.relative_to(ROOT)}")

    for i, (src, im, anns) in enumerate(pick_images(args.n, args.seed), 1):
        path = SOURCES[src] / im["file_name"]
        image = Image.open(path).convert("RGB")
        det = model.predict(image, threshold=args.threshold)
        pred = [(*xyxy, int(c) + 1, float(s)) for xyxy, c, s in zip(det.xyxy, det.class_id, det.confidence)]
        gt = [(a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3],
               a["category_id"], None) for a in anns]
        name = f"{im['seq']} #{im['frame']}" if src == "mot" else Path(im["file_name"]).stem
        left = draw_panel(image, gt, f"GROUND TRUTH  {len(gt)} boxes   [{src.upper()} test-dev] {name}", font, small)
        right = draw_panel(image, pred, f"RF-DETR  {len(pred)} boxes (score >= {args.threshold})", font, small)
        sheet = Image.new("RGB", (2 * PANEL_W + 8, left.height + 26), "white")
        sheet.paste(left, (0, 0))
        sheet.paste(right, (PANEL_W + 8, 0))
        sheet.paste(legend(2 * PANEL_W + 8, small), (0, left.height))
        file = out / f"{i:02d}_{src}_{Path(im['file_name']).stem}.jpg"
        sheet.save(file, quality=88)
        log(f"{file.name}: GT {len(gt)}, predicted {len(pred)}", 1)
    log("Done")


if __name__ == "__main__":
    main()
