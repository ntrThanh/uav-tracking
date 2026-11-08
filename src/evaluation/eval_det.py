"""Detection metrics (E0) of every detector, from its detection cache, with one pycocotools protocol for all.

Every model is scored by this script on the same files, so the numbers can be compared (rule 4 of the plan): mAP
from `yolo val`, torchvision or the RF-DETR log is not used.

Protocol:
  * GT: data/processed/coco/<split>.json (MOT val / test-dev, every frame, VisDrone classes 1-10).
  * Detections: results/dets/<det>/<split>/<seq>.txt, frame,-1,x,y,w,h,score,class (cache made at conf 0.05).
  * COCO bbox AP with maxDets = [1, 100, 500]; every AP / AR is read at maxDets = 500 (VisDrone frames are crowded).
  * Two variants: `ign` drops GT boxes and detections with more than half of their area inside the frame's ignored
    regions (the rule of the VisDrone toolkit and of eval_track.py, main variant); `raw` keeps everything.
  * P / R / F1 at the working point conf >= 0.5, IoU 0.5, micro over the 10 classes and over the 5 tracking classes.
    Matches are the greedy COCO matches; a detection's match never depends on lower-score detections, so keeping
    the matched detections with score >= 0.5 is the same as evaluating only those.

Output:
  results/tables/det_<split>.csv            one row per detector and variant
  results/tables/det_<split>_per_class.csv  AP50-95 and AP50 per class
  results/tables/det/<det>_<split>_<variant>.json

Usage (from the project root):
    python -m src.evaluation.eval_det --split val test-dev
    python -m src.evaluation.eval_det --split test-dev --det yolo11m oracle --variant ign
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data" / "tools"))
from visdrone import CLASS_NAMES, TRACK_CLASSES, log  # noqa: E402

IGNORE_OVERLAP = 0.5
WORK_CONF = 0.5
VARIANTS = ["ign", "raw"]
DETECTORS = ["yolo11m", "frcnn_r50", "rfdetr_m", "oracle"]
COLUMNS = ["det", "split", "variant", "mAP50_95", "mAP50", "mAP75", "AP_small", "AP_medium", "AP_large",
           "AR_1", "AR_100", "AR_500", "AR_small", "AR_medium", "AR_large", "mAP50_95_track5", "mAP50_track5",
           "P@0.5", "R@0.5", "F1@0.5", "P@0.5_track5", "R@0.5_track5", "F1@0.5_track5",
           "n_images", "n_gt", "n_dets", "seconds"]


# ----------------------------------------------------------------------------------------------- data
def ignored_keep(boxes: np.ndarray, regions: np.ndarray, width: int, height: int) -> np.ndarray:
    """boxes, regions: (N, 4) x, y, w, h. Mask of boxes NOT more than half inside the union of the regions."""
    keep = np.ones(len(boxes), dtype=bool)
    if not len(regions) or not len(boxes):
        return keep
    mask = np.zeros((height, width), dtype=bool)

    def pixels(x, y, w, h):
        return (int(np.clip(np.floor(x), 0, width)), int(np.clip(np.floor(y), 0, height)),
                int(np.clip(np.ceil(x + w), 0, width)), int(np.clip(np.ceil(y + h), 0, height)))

    for r in regions:
        x1, y1, x2, y2 = pixels(*r)
        mask[y1:y2, x1:x2] = True
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = pixels(*b)
        keep[i] = mask[y1:y2, x1:x2].sum() / max((x2 - x1) * (y2 - y1), 1) <= IGNORE_OVERLAP
    return keep


def load_gt(split: str) -> dict:
    return json.load(open(ROOT / "data" / "processed" / "coco" / f"{split}.json"))


def load_dets(det: str, split: str, images: list[dict]) -> list[dict]:
    """Cache files -> COCO result dicts. Every image of the GT must belong to a cached sequence."""
    folder = ROOT / "results" / "dets" / det / split
    image_id = {(im["seq"], im["frame"]): im["id"] for im in images}
    seqs = sorted({im["seq"] for im in images})
    out = []
    for seq in seqs:
        path = folder / f"{seq}.txt"
        if not path.is_file():
            raise SystemExit(f"{det}/{split}: missing {path}")
        if path.stat().st_size == 0:
            continue
        rows = np.loadtxt(path, delimiter=",", ndmin=2)
        for frame, _, x, y, w, h, score, cls in rows[:, :8]:
            iid = image_id.get((seq, int(frame)))
            if iid is None:
                raise SystemExit(f"{det}/{split}: {seq} frame {int(frame)} is not in the GT")
            out.append({"image_id": iid, "category_id": int(cls), "bbox": [float(x), float(y), float(w), float(h)],
                        "score": float(score)})
    return out


def apply_ignored(gt: dict, dets: list[dict]) -> tuple[dict, list[dict]]:
    regions: dict[int, list] = {}
    for r in gt.get("ignored_regions", []):
        regions.setdefault(r["image_id"], []).append(r["bbox"])
    size = {im["id"]: (im["width"], im["height"]) for im in gt["images"]}

    def filt(items):
        by_image: dict[int, list] = {}
        for a in items:
            by_image.setdefault(a["image_id"], []).append(a)
        kept = []
        for iid, group in by_image.items():
            reg = np.array(regions.get(iid, []), dtype=float).reshape(-1, 4)
            keep = ignored_keep(np.array([a["bbox"] for a in group], dtype=float).reshape(-1, 4), reg, *size[iid])
            kept.extend(a for a, k in zip(group, keep) if k)
        return kept

    return {**gt, "annotations": filt(gt["annotations"])}, filt(dets)


# ----------------------------------------------------------------------------------------------- metrics
def mean_valid(a: np.ndarray) -> float:
    v = a[a > -1]
    return round(float(v.mean()) * 100, 2) if v.size else float("nan")


def working_point(ev, cat_ids: list[int], scores: dict[int, float]) -> tuple[float, float, float]:
    """Micro P / R / F1 at conf >= WORK_CONF and IoU 0.5 over the given categories (area all, maxDets 500)."""
    tp = fp = n_gt = 0
    for e in ev.evalImgs:
        if e is None or e["category_id"] not in cat_ids or e["aRng"] != ev.params.areaRng[0] \
                or e["maxDet"] != ev.params.maxDets[-1]:
            continue
        gt_ignore = np.asarray(e["gtIgnore"], dtype=bool)
        n_gt += int((~gt_ignore).sum())
        matched = np.asarray(e["dtMatches"])[0] > 0           # IoU 0.5 row
        dt_ignore = np.asarray(e["dtIgnore"])[0].astype(bool)
        conf = np.array([scores[d] for d in e["dtIds"]])
        work = (conf >= WORK_CONF) & ~dt_ignore
        tp += int((work & matched).sum())
        fp += int((work & ~matched).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(n_gt, 1)
    f1 = 2 * p * r / max(p + r, 1e-12)
    return round(p * 100, 2), round(r * 100, 2), round(f1 * 100, 2)


def evaluate(det: str, split: str, variant: str) -> tuple[dict, list[dict]]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    t0 = time.time()
    gt_json = load_gt(split)
    dets = load_dets(det, split, gt_json["images"])
    if variant == "ign":
        gt_json, dets = apply_ignored(gt_json, dets)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = gt_json
        gt.createIndex()
        dt = gt.loadRes(dets)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.maxDets = [1, 100, 500]
        ev.evaluate()
        ev.accumulate()
    prec, rec = ev.eval["precision"], ev.eval["recall"]   # [T, R, K, A, M], [T, K, A, M]
    cats = list(ev.params.catIds)
    track_k = [cats.index(c) for c in TRACK_CLASSES]
    scores = {d["id"]: d["score"] for d in dt.dataset["annotations"]}
    p, r, f1 = working_point(ev, cats, scores)
    pt, rt, f1t = working_point(ev, TRACK_CLASSES, scores)
    row = {
        "det": det, "split": split, "variant": variant,
        "mAP50_95": mean_valid(prec[:, :, :, 0, 2]), "mAP50": mean_valid(prec[0, :, :, 0, 2]),
        "mAP75": mean_valid(prec[5, :, :, 0, 2]),
        "AP_small": mean_valid(prec[:, :, :, 1, 2]), "AP_medium": mean_valid(prec[:, :, :, 2, 2]),
        "AP_large": mean_valid(prec[:, :, :, 3, 2]),
        "AR_1": mean_valid(rec[:, :, 0, 0]), "AR_100": mean_valid(rec[:, :, 0, 1]),
        "AR_500": mean_valid(rec[:, :, 0, 2]),
        "AR_small": mean_valid(rec[:, :, 1, 2]), "AR_medium": mean_valid(rec[:, :, 2, 2]),
        "AR_large": mean_valid(rec[:, :, 3, 2]),
        "mAP50_95_track5": mean_valid(prec[:, :, track_k, 0, 2]), "mAP50_track5": mean_valid(prec[0, :, track_k, 0, 2]),
        "P@0.5": p, "R@0.5": r, "F1@0.5": f1, "P@0.5_track5": pt, "R@0.5_track5": rt, "F1@0.5_track5": f1t,
        "n_images": len(gt_json["images"]), "n_gt": len(gt_json["annotations"]), "n_dets": len(dets),
        "seconds": round(time.time() - t0),
    }
    per_class = [{"det": det, "split": split, "variant": variant, "class_id": int(c), "class": CLASS_NAMES[int(c)],
                  "AP50_95": mean_valid(prec[:, :, k, 0, 2]), "AP50": mean_valid(prec[0, :, k, 0, 2]),
                  "AR_500": mean_valid(rec[:, k, 0, 2]), "track_class": c in TRACK_CLASSES}
                 for k, c in enumerate(cats)]
    return row, per_class


def _job(args):
    return evaluate(*args)


# ----------------------------------------------------------------------------------------------- output
def merge(path: Path, rows: list[dict], key: list[str]) -> None:
    """Replace rows with the same key, keep the others, sort."""
    old = list(csv.DictReader(open(path))) if path.is_file() else []
    new_keys = {tuple(str(r[k]) for k in key) for r in rows}
    out = [r for r in old if tuple(r[k] for k in key) not in new_keys] + rows
    out.sort(key=lambda r: tuple(str(r[k]) for k in key))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", nargs="+", default=["val", "test-dev"], choices=["val", "test-dev"])
    ap.add_argument("--det", nargs="+", default=DETECTORS)
    ap.add_argument("--variant", nargs="+", default=VARIANTS, choices=VARIANTS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tables", type=Path, default=ROOT / "results" / "tables")
    args = ap.parse_args()

    jobs = [(d, s, v) for s in args.split for d in args.det for v in args.variant]
    log(f"{len(jobs)} evaluations on {args.workers} workers")
    results: dict[str, list] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for (det, split, variant), (row, per_class) in zip(jobs, pool.map(_job, jobs)):
            log(f"{det:10s} {split:8s} {variant}  mAP50-95 {row['mAP50_95']:6.2f}  mAP50 {row['mAP50']:6.2f}  "
                f"AP_s {row['AP_small']:6.2f}  AR@500 {row['AR_500']:6.2f}  P/R/F1@0.5 {row['P@0.5']:.1f}/"
                f"{row['R@0.5']:.1f}/{row['F1@0.5']:.1f}  ({row['seconds']}s)", 1)
            results.setdefault(split, []).append((row, per_class))
            out = args.tables / "det" / f"{det}_{split}_{variant}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            json.dump({"metrics": row, "per_class": per_class,
                       "protocol": "COCO bbox, maxDets [1,100,500], AP/AR at 500, values in %, "
                                   f"variant {variant}, P/R/F1 at conf>={WORK_CONF} IoU 0.5"},
                      open(out, "w"), indent=2, default=lambda o: o.item())
    for split, items in results.items():
        merge(args.tables / f"det_{split}.csv", [r for r, _ in items], ["det", "variant"])
        merge(args.tables / f"det_{split}_per_class.csv", [c for _, pc in items for c in pc],
              ["det", "variant", "class_id"])
        log(f"Saved results/tables/det_{split}.csv and det_{split}_per_class.csv")


if __name__ == "__main__":
    main()
