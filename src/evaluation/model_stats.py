"""Parameters and GFLOPs of one detector, on real VisDrone frames, through the same predict() used for the caches.

FLOPs are counted with torch.utils.flop_counter.FlopCounterMode, which sees every op dispatched during predict(),
so the resize each model does (YOLO letterbox 640, Faster R-CNN 800x1333, RF-DETR 576x576) is the real one. One
multiply-add counts as 2 FLOPs (the convention of Ultralytics' GFLOPs). Because the cost depends on the frame size,
it is measured on one frame of every resolution present in the split and averaged with the frame counts as weights.

Output: results/tables/model_stats/<det>.json; merge into results/tables/model_stats.csv with --merge.

Usage (from the project root; RF-DETR in the rfdetr environment):
    python -m src.evaluation.model_stats --det yolo11m
    python -m src.evaluation.model_stats --det rfdetr_m
    python -m src.evaluation.model_stats --merge
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data" / "tools"))
from visdrone import log  # noqa: E402

from src.detectors import available_detectors, build_detector  # noqa: E402
from src.trackers.io import read_seqinfo  # noqa: E402

OUT = ROOT / "results" / "tables" / "model_stats"


def torch_module(detector):
    """The nn.Module that holds the weights of each wrapper."""
    import torch
    for path in ("model.model", "model.model.model", "model"):
        obj = detector
        try:
            for name in path.split("."):
                obj = getattr(obj, name)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.Module):
            return obj
    raise TypeError(f"No nn.Module found in {type(detector).__name__}")


def frames_by_resolution(split: str) -> dict[tuple[int, int], tuple[Path, int]]:
    """{(width, height): (first image of that size, frames of the split with that size)}."""
    out: dict[tuple[int, int], tuple[Path, int]] = {}
    for d in sorted(p for p in (ROOT / "data" / "processed" / "mot" / split).iterdir() if p.is_dir()):
        info = read_seqinfo(d)
        image = info.image_path(1)
        h, w = cv2.imread(str(image)).shape[:2]
        first, n = out.get((w, h), (image, 0))
        out[(w, h)] = (first, n + info.length)
    return out


def measure(det: str, split: str, device: str) -> dict:
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    detector = build_detector(det, device=device)
    module = torch_module(detector)
    params = sum(p.numel() for p in module.parameters())
    module.requires_grad_(False)    # FlopCounterMode's module hooks fail on parameters that require grad
    per_res = []
    for (w, h), (image_path, frames) in sorted(frames_by_resolution(split).items()):
        image = cv2.imread(str(image_path))
        detector.predict(image)                         # build lazy state outside the count
        counter = FlopCounterMode(display=False)
        with counter, torch.inference_mode():
            detector.predict(image)
        per_res.append({"width": w, "height": h, "frames": frames, "image": str(image_path.relative_to(ROOT)),
                        "GFLOPs": counter.get_total_flops() / 1e9})
        log(f"{w}x{h}: {per_res[-1]['GFLOPs']:.1f} GFLOPs ({frames} frames)", 1)
    total = sum(r["frames"] for r in per_res)
    return {"det": det, "split": split, "params_M": round(params / 1e6, 2),
            "GFLOPs_mean": round(sum(r["GFLOPs"] * r["frames"] for r in per_res) / total, 1),
            "GFLOPs_1920x1080": next((round(r["GFLOPs"], 1) for r in per_res if (r["width"], r["height"]) ==
                                      (1920, 1080)), None),
            "flop_convention": "2 FLOPs per multiply-add, torch FlopCounterMode over predict()",
            "settings": detector.settings(), "per_resolution": per_res}


def merge() -> None:
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
    cols = ["det", "split", "params_M", "GFLOPs_mean", "GFLOPs_1920x1080"]
    with open(OUT.parent / "model_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log(f"Saved results/tables/model_stats.csv ({len(rows)} detectors)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--det", choices=available_detectors())
    ap.add_argument("--split", default="test-dev", choices=["val", "test-dev"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--merge", action="store_true", help="only write model_stats.csv from the json files")
    args = ap.parse_args()
    if args.det:
        log(f"{args.det}: parameters and GFLOPs on {args.split}")
        stats = measure(args.det, args.split, args.device)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{args.det}.json").write_text(json.dumps(stats, indent=2))
        log(f"{args.det}: {stats['params_M']} M parameters, {stats['GFLOPs_mean']} GFLOPs (mean over {args.split})")
    if args.merge or not args.det:
        merge()


if __name__ == "__main__":
    main()
