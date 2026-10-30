"""Overfitting check: evaluate one RF-DETR checkpoint on a train subset, val and a test-dev subset.

The same prediction and pycocotools code is used for every split, so a large gap between train and val mAP
means the model memorises the training scenes. Ignored regions are not filtered here (same for all splits),
so absolute numbers are slightly lower than the final evaluation.

Usage (from the project root, one checkpoint per GPU):
    CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.diagnose_overfit \
        --checkpoint runs/det/T3_rfdetr_m_576/checkpoint_best_total.pth --tag epoch1
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data" / "tools"))
from visdrone import CLASS_NAMES  # noqa: E402

DATA = ROOT / "data" / "processed" / "rfdetr"
OUT = ROOT / "results" / "diagnostics"
SPLITS = {"train": "train", "val": "valid", "test-dev": "test"}


def log(msg: str, level: int = 0) -> None:
    print("    " * level + ("[+] " if level == 0 else "[-] ") + msg, flush=True)


def subset(images: list[dict], every: int) -> list[dict]:
    """Every n-th image per sequence, so all sequences stay represented."""
    by_seq: dict[str, list[dict]] = {}
    for im in images:
        by_seq.setdefault(im["seq"], []).append(im)
    return [im for seq in sorted(by_seq) for im in sorted(by_seq[seq], key=lambda x: x["frame"])[::every]]


def predict(model, split_dir: Path, images: list[dict], batch: int, threshold: float) -> list[dict]:
    results = []
    for i in range(0, len(images), batch):
        chunk = images[i:i + batch]
        dets = model.predict([Image.open(split_dir / im["file_name"]).convert("RGB") for im in chunk],
                             threshold=threshold)
        for im, d in zip(chunk, dets if isinstance(dets, list) else [dets]):
            for (x1, y1, x2, y2), s, c in zip(d.xyxy, d.confidence, d.class_id):
                results.append({"image_id": im["id"], "category_id": int(c) + 1,
                                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)})
    return results


def coco_eval(ann_file: Path, results: list[dict], image_ids: list[int]) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ann_file))
        dt = gt.loadRes(results) if results else None
        if dt is None:
            return {}
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = image_ids
        ev.params.maxDets = [1, 100, 500]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats
    per_class = {}
    prec = ev.eval["precision"]  # [T, R, K, A, M]
    for k, cat in enumerate(ev.params.catIds):
        p = prec[:, :, k, 0, -1]
        per_class[CLASS_NAMES[cat]] = float(np.mean(p[p > -1])) if (p > -1).any() else float("nan")
    return {"mAP50_95": s[0], "mAP50": s[1], "mAP75": s[2], "AP_small": s[3], "AP_medium": s[4], "AP_large": s[5],
            "AR_500": s[8], "per_class": per_class}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--tag", required=True, help="name used in the output file, e.g. epoch1")
    ap.add_argument("--model", default="RFDETRMedium")
    ap.add_argument("--resolution", type=int, default=576)
    ap.add_argument("--train-every", type=int, default=10, help="every n-th train frame per sequence")
    ap.add_argument("--test-every", type=int, default=5, help="every n-th test-dev frame per sequence")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=0.01)
    args = ap.parse_args()

    import rfdetr
    model = getattr(rfdetr, args.model)(pretrain_weights=str(args.checkpoint), num_classes=len(CLASS_NAMES),
                                        resolution=args.resolution)
    log(f"Checkpoint {args.checkpoint} ({args.tag}), resolution {args.resolution}")

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"checkpoint": str(args.checkpoint), "tag": args.tag, "splits": {}}
    for split, folder in SPLITS.items():
        ann = DATA / folder / "_annotations.coco.json"
        images = json.load(open(ann))["images"]
        every = {"train": args.train_every, "val": 1, "test-dev": args.test_every}[split]
        images = subset(images, every)
        t0 = time.time()
        res = predict(model, DATA / folder, images, args.batch, args.threshold)
        m = coco_eval(ann, res, [im["id"] for im in images])
        m.update(n_images=len(images), n_dets=len(res), seconds=round(time.time() - t0))
        summary["splits"][split] = m
        log(f"{split:<8} {len(images):>5} images  mAP50-95 {m['mAP50_95']:.3f}  mAP50 {m['mAP50']:.3f}  "
            f"AP_small {m['AP_small']:.3f}  AR@500 {m['AR_500']:.3f}  ({m['seconds']}s)", 1)
    out = OUT / f"overfit_{args.tag}.json"
    json.dump(summary, open(out, "w"), indent=2)
    log(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
