"""Run one tracker on a detection cache.

Reads results/dets/<det>/<split>/<seq>.txt and writes results/tracks/<det>/<tracker>/<split>/<seq>.txt, one file per
sequence in MOTChallenge format with the class in column 8, plus run_meta.json (parameters and timing).

Usage (from the project root):
    python -m src.inference.run_tracker --det frcnn_r50 --split val --tracker bytetrack
    python -m src.inference.run_tracker --det frcnn_r50 --split val --tracker botsort --det-thresh 0.4 --tag thr04
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data" / "tools"))
from visdrone import log  # noqa: E402

from src.trackers import Tracker, available_trackers, load_config  # noqa: E402
from src.trackers.io import SequenceInfo, format_tracks, read_detections, read_seqinfo  # noqa: E402


def run_sequence(tracker: Tracker, info: SequenceInfo, detections: dict, out_path: Path) -> dict:
    """Every frame is processed, also those without detections, so that tracks age correctly."""
    empty = np.zeros((0, 6))
    temporary = out_path.with_suffix(".tmp")
    t0 = time.perf_counter()
    n_rows = 0
    with open(temporary, "w") as f:
        for frame in range(1, info.length + 1):
            image = cv2.imread(str(info.image_path(frame))) if tracker.requires_image else None
            if tracker.requires_image and image is None:
                raise FileNotFoundError(info.image_path(frame))
            lines = format_tracks(frame, tracker.update(detections.get(frame, empty), image))
            f.writelines(lines)
            n_rows += len(lines)
    os.replace(temporary, out_path)
    timing = {"frames": info.length, "rows": n_rows, "wall_seconds": time.perf_counter() - t0}
    return {**timing, **{f"{k}_seconds": v for k, v in tracker.seconds.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--det", required=True, help="detector id, e.g. yolo11m, frcnn_r50, rfdetr_m, oracle")
    ap.add_argument("--split", required=True, choices=["val", "test-dev"])
    ap.add_argument("--tracker", required=True, choices=available_trackers())
    ap.add_argument("--config", type=Path, help="default: src/configs/trackers/<tracker>.yaml")
    ap.add_argument("--det-thresh", type=float, help="override the high detection threshold (tuning stage)")
    ap.add_argument("--tag", help="suffix for the output folder, e.g. thr04")
    ap.add_argument("--dets", type=Path, help="default: results/dets/<det>/<split>")
    ap.add_argument("--mot-root", type=Path, default=ROOT / "data" / "processed" / "mot")
    ap.add_argument("--out", type=Path, help="default: results/tracks/<det>/<tracker>[_tag]/<split>")
    ap.add_argument("--seqs", nargs="+", help="only these sequences")
    args = ap.parse_args()

    dets_dir = args.dets or ROOT / "results" / "dets" / args.det / args.split
    folder = args.tracker + (f"_{args.tag}" if args.tag else "")
    out_dir = args.out or ROOT / "results" / "tracks" / args.det / folder / args.split
    sequences = sorted(p for p in (args.mot_root / args.split).iterdir() if p.is_dir())
    if args.seqs:
        sequences = [p for p in sequences if p.name in args.seqs]
    if not sequences:
        raise SystemExit(f"No sequences found in {args.mot_root / args.split}")
    missing = [p.name for p in sequences if not (dets_dir / f"{p.name}.txt").is_file()]
    if missing:
        raise SystemExit(f"Detection cache incomplete in {dets_dir}: missing {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"{args.tracker} on {args.det}/{args.split}: {len(sequences)} sequences -> {out_dir}")
    overrides = {} if args.det_thresh is None else {"det_thresh": args.det_thresh}
    encoder = None
    meta = {"tracker": args.tracker, "det": args.det, "split": args.split, "sequences": {}}
    for sequence in sequences:
        info = read_seqinfo(sequence)
        cfg = load_config(args.tracker, args.config, frame_rate=info.frame_rate, **overrides)
        tracker = Tracker(args.tracker, cfg, encoder=encoder)
        encoder = tracker.front_end.encoder     # load the ReID network once, not per sequence
        result = run_sequence(tracker, info, read_detections(dets_dir / f"{sequence.name}.txt"), out_dir / f"{info.name}.txt")
        meta["sequences"][info.name] = result
        meta["config"] = tracker.config_dict()
        log(f"{info.name}: {result['frames']} frames, {result['rows']} rows, "
            f"{result['frames'] / result['wall_seconds']:.1f} FPS", 1)

    runs = meta["sequences"].values()
    frames = sum(r["frames"] for r in runs)
    meta["fps_tracker"] = frames / max(sum(r["association_seconds"] for r in runs), 1e-9)
    meta["fps_tracker_with_front_end"] = frames / max(
        sum(r["association_seconds"] + r["front_end_seconds"] for r in runs), 1e-9)
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"Done: {meta['fps_tracker']:.1f} FPS association, {meta['fps_tracker_with_front_end']:.1f} FPS with ReID/CMC")


if __name__ == "__main__":
    main()
