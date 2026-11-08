"""Cache the detections of one detector on every frame of the MOT val / test-dev sequences.

Writes results/dets/<det>/<split>/<seq>.txt, one line per box: frame,-1,x,y,w,h,score,class (original image
coordinates, VisDrone class 1-10, all 10 classes kept), plus timing.json with the settings and the latency.

Latency is measured the same way for every model: batch 1, the first --warmup frames of the split are run once and
discarded, torch.cuda.synchronize() around each call, image decoding excluded. Pre- and post-processing of the model
(resize, NMS) are included.

--det oracle writes the ground truth as detections (score 1); it needs no model and no GPU.

Usage (from the project root):
    python -m src.inference.cache_dets --det yolo11m --split val test-dev
    python -m src.inference.cache_dets --det rfdetr_m --split val --device cuda:1
    python -m src.inference.cache_dets --det oracle --split val test-dev
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data" / "tools"))
from visdrone import log  # noqa: E402

from src.detectors import available_detectors, build_detector  # noqa: E402
from src.trackers.io import SequenceInfo, read_seqinfo  # noqa: E402

ORACLE = "oracle"


def cache_lines(frame: int, dets: np.ndarray) -> list[str]:
    return [f"{frame},-1,{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{s:.5f},{int(c)}\n"
            for x1, y1, x2, y2, s, c in dets]


def write_atomic(path: Path, lines: list[str]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(lines))
    os.replace(temporary, path)


def oracle_sequence(sequence_dir: Path) -> list[str]:
    """gt.txt is frame,id,x,y,w,h,1,class,occlusion with classes 1-10 (made by data/tools/prepare.py)."""
    gt = np.loadtxt(sequence_dir / "gt" / "gt.txt", delimiter=",", ndmin=2)
    gt = gt[np.lexsort((gt[:, 1], gt[:, 0]))]
    return [f"{int(f)},-1,{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,{int(c)}\n" for f, _, x, y, w, h, _, c, *_ in gt]


class Timer:
    def __init__(self, device: str):
        import torch
        self.cuda = device.startswith("cuda")
        self.torch = torch

    def __call__(self, fn, *args):
        if self.cuda:
            self.torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*args)
        if self.cuda:
            self.torch.cuda.synchronize()
        return out, time.perf_counter() - t0


def warm_up(detector, sequences: list[SequenceInfo], n: int) -> None:
    frames = [(s, f) for s in sequences for f in range(1, s.length + 1)][:n]
    for info, frame in frames:
        detector.predict(cv2.imread(str(info.image_path(frame))))


def model_sequence(detector, timer: Timer, info: SequenceInfo) -> tuple[list[str], list[float], int]:
    lines, seconds = [], []
    for frame in range(1, info.length + 1):
        image = cv2.imread(str(info.image_path(frame)))
        if image is None:
            raise FileNotFoundError(info.image_path(frame))
        dets, dt = timer(detector.predict, image)
        seconds.append(dt)
        lines += cache_lines(frame, dets)
    return lines, seconds, len(lines)


def environment(device: str) -> dict:
    import torch
    env = {"python": platform.python_version(), "torch": torch.__version__, "host": platform.node()}
    if device.startswith("cuda"):
        env["gpu"] = torch.cuda.get_device_name(torch.device(device))
    return env


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--det", required=True, choices=available_detectors() + [ORACLE])
    ap.add_argument("--split", nargs="+", required=True, choices=["val", "test-dev"])
    ap.add_argument("--weights", help="default: the checkpoint in weights/turning (see src/detectors)")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--mot-root", type=Path, default=ROOT / "data" / "processed" / "mot")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "dets")
    ap.add_argument("--seqs", nargs="+", help="only these sequences")
    args = ap.parse_args()

    detector = None
    if args.det != ORACLE:
        detector = build_detector(args.det, weights=args.weights, device=args.device, conf=args.conf)
        timer = Timer(args.device)

    for split in args.split:
        dirs = sorted(p for p in (args.mot_root / split).iterdir() if p.is_dir())
        if args.seqs:
            dirs = [p for p in dirs if p.name in args.seqs]
        out_dir = args.out / args.det / split
        out_dir.mkdir(parents=True, exist_ok=True)
        infos = [read_seqinfo(d) for d in dirs]
        log(f"{args.det} on {split}: {len(dirs)} sequences, {sum(i.length for i in infos)} frames -> {out_dir}")

        if detector is None:
            for d, info in zip(dirs, infos):
                lines = oracle_sequence(d)
                write_atomic(out_dir / f"{info.name}.txt", lines)
                log(f"{info.name}: {len(lines)} boxes", 1)
            meta = {"det": ORACLE, "split": split, "source": "data/processed/mot/<split>/<seq>/gt/gt.txt", "score": 1}
            (out_dir / "timing.json").write_text(json.dumps(meta, indent=2))
            continue

        warm_up(detector, infos, args.warmup)
        per_sequence, all_seconds = {}, []
        for info in infos:
            lines, seconds, n = model_sequence(detector, timer, info)
            write_atomic(out_dir / f"{info.name}.txt", lines)
            all_seconds += seconds
            per_sequence[info.name] = {"frames": info.length, "boxes": n, "ms_per_frame": 1000 * float(np.mean(seconds))}
            log(f"{info.name}: {info.length} frames, {n} boxes, {per_sequence[info.name]['ms_per_frame']:.1f} ms/frame", 1)

        ms = 1000 * np.asarray(all_seconds)
        meta = {"det": args.det, "split": split, **detector.describe(), "batch": 1, "warmup_frames": args.warmup,
                "frames": len(ms), "ms_per_frame": float(ms.mean()), "ms_per_frame_median": float(np.median(ms)),
                "fps": float(1000 / ms.mean()), "environment": environment(args.device), "sequences": per_sequence}
        (out_dir / "timing.json").write_text(json.dumps(meta, indent=2))
        log(f"Done {split}: {meta['ms_per_frame']:.1f} ms/frame ({meta['fps']:.1f} FPS)")


if __name__ == "__main__":
    main()
