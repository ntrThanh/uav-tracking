"""Speed benchmark under one protocol: one GPU used by nothing else, the same frames, repeated runs.

The FPS columns written during caching and tracking are not comparable: they were measured while several jobs shared
the GPUs and the CPU. This script measures again, one job at a time:

  det    latency of detector.predict() per frame (image decoding excluded, cuda.synchronize around each call,
         --warmup frames run first and discarded), on the frames of --seqs, --repeats times.
  track  time of every tracker run of the test-dev matrix of one detector (default and tuned thresholds) on the
         cached detections of the same frames: association, and front end (ReID crops + network, camera motion).
  merge  median over repeats -> results/tables/fps_bench.csv, and replaces the FPS columns of
         matrix_testdev_mean.csv for the runs that were benchmarked. FPS_e2e = frames / (detector + front end +
         association seconds), as in eval_track.py.

The default sequences are the sparsest, a medium and the densest test-dev sequence (GT boxes per frame 9, 43, 110).

Usage (from the project root, one GPU visible):
    CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.bench_fps det --det yolo11m
    CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.bench_fps track --det yolo11m
    python -m src.evaluation.bench_fps merge
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data" / "tools"))
from visdrone import log  # noqa: E402

from src.trackers.io import read_detections, read_seqinfo  # noqa: E402

SPLIT = "test-dev"
SEQS = ["uav0000370_00001_v", "uav0000073_00600_v", "uav0000073_04464_v"]
OUT = ROOT / "results" / "tables" / "fps_bench"
MATRIX = ROOT / "results" / "tables" / "matrix_testdev_mean.csv"
TAG = re.compile(r"^(?P<tracker>[a-z]+)(?:_thr(?P<thr>\d+))?$")


def sequences(names: list[str]):
    return [read_seqinfo(ROOT / "data" / "processed" / "mot" / SPLIT / n) for n in names]


def environment() -> dict:
    import torch
    env = {"host": platform.node(), "python": platform.python_version(), "torch": torch.__version__,
           "cpu": platform.processor() or platform.machine()}
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
    return env


# ----------------------------------------------------------------------------------------------- detector
def bench_det(det: str, names: list[str], repeats: int, warmup: int) -> dict:
    import torch
    from src.detectors import build_detector

    detector = build_detector(det, device="cuda:0")
    infos = sequences(names)
    images = [cv2.imread(str(i.image_path(f))) for i in infos for f in range(1, i.length + 1)]
    log(f"{det}: {len(images)} frames loaded, warm-up {warmup}")
    for image in images[:warmup]:
        detector.predict(image)
    runs = []
    for r in range(repeats):
        seconds = []
        for image in images:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            detector.predict(image)
            torch.cuda.synchronize()
            seconds.append(time.perf_counter() - t0)
        ms = 1000 * np.asarray(seconds)
        runs.append({"ms_per_frame": float(ms.mean()), "ms_median": float(np.median(ms))})
        log(f"repeat {r + 1}: {runs[-1]['ms_per_frame']:.2f} ms/frame ({1000 / runs[-1]['ms_per_frame']:.1f} FPS)", 1)
    ms = float(np.median([r["ms_per_frame"] for r in runs]))
    return {"det": det, "split": SPLIT, "seqs": names, "frames": len(images), "warmup": warmup, "repeats": runs,
            "ms_per_frame": ms, "fps": 1000 / ms, "settings": detector.settings(), "environment": environment()}


# ----------------------------------------------------------------------------------------------- tracker
def matrix_runs(det: str) -> list[str]:
    """Tracker folders of this detector in the test-dev matrix (default and tuned)."""
    return sorted(r["tracker"] for r in csv.DictReader(open(MATRIX)) if r["det"] == det)


def bench_track(det: str, names: list[str], repeats: int, runs: list[str] | None) -> dict:
    from src.evaluation.select_thresh import tag_value
    from src.inference.run_tracker import run_sequence
    from src.trackers import Tracker, load_config

    infos = sequences(names)
    dets = {i.name: read_detections(ROOT / "results" / "dets" / det / SPLIT / f"{i.name}.txt") for i in infos}
    frames = sum(i.length for i in infos)
    out: dict[str, dict] = {}
    encoder = None
    with tempfile.TemporaryDirectory() as tmp:
        for run in runs or matrix_runs(det):
            m = TAG.match(run)
            if not m:
                raise SystemExit(f"Cannot read the tracker and threshold of run {run!r}")
            name = m["tracker"]
            overrides = {} if m["thr"] is None else {"det_thresh": tag_value(m["thr"])}
            reps = []
            for r in range(repeats + 1):                 # repeat 0 warms up the ReID network and caches
                assoc = front = 0.0
                for info in infos:
                    cfg = load_config(name, frame_rate=info.frame_rate, **overrides)
                    tracker = Tracker(name, cfg, encoder=encoder)
                    encoder = tracker.front_end.encoder or encoder
                    res = run_sequence(tracker, info, dets[info.name], Path(tmp) / f"{info.name}.txt")
                    assoc += res["association_seconds"]
                    front += res["front_end_seconds"]
                if r:
                    reps.append({"association_seconds": assoc, "front_end_seconds": front})
            a = float(np.median([x["association_seconds"] for x in reps]))
            f = float(np.median([x["front_end_seconds"] for x in reps]))
            out[run] = {"tracker": name, **overrides, "repeats": reps, "association_seconds": a,
                        "front_end_seconds": f, "FPS_tracker": frames / max(a, 1e-9),
                        "FPS_tracker_front_end": frames / max(a + f, 1e-9)}
            log(f"{det} {run:16s} {out[run]['FPS_tracker']:8.1f} FPS association, "
                f"{out[run]['FPS_tracker_front_end']:7.1f} FPS with front end", 1)
    return {"det": det, "split": SPLIT, "seqs": names, "frames": frames, "runs": out, "environment": environment()}


# ----------------------------------------------------------------------------------------------- merge
def merge() -> None:
    rows = []
    for path in sorted(OUT.glob("track_*.json")):
        t = json.loads(path.read_text())
        det_path = OUT / f"det_{t['det']}.json"
        det = json.loads(det_path.read_text()) if det_path.is_file() else None
        if det and det["seqs"] != t["seqs"]:
            raise SystemExit(f"{t['det']}: detector and tracker benchmarks use different sequences")
        for run, r in t["runs"].items():
            det_s = det["ms_per_frame"] / 1000 * t["frames"] if det else None
            rows.append({"det": t["det"], "tracker": run, "split": SPLIT, "frames": t["frames"],
                         "FPS_det": round(det["fps"], 2) if det else "",
                         "FPS_tracker": round(r["FPS_tracker"], 2),
                         "FPS_tracker_front_end": round(r["FPS_tracker_front_end"], 2),
                         "FPS_e2e": round(t["frames"] / (det_s + r["association_seconds"] + r["front_end_seconds"]), 2)
                         if det else ""})
    for path in sorted(OUT.glob("det_*.json")):          # oracle-free detectors without a tracker benchmark
        d = json.loads(path.read_text())
        if not any(r["det"] == d["det"] for r in rows):
            rows.append({"det": d["det"], "tracker": "", "split": SPLIT, "frames": d["frames"],
                         "FPS_det": round(d["fps"], 2), "FPS_tracker": "", "FPS_tracker_front_end": "", "FPS_e2e": ""})
    if not rows:
        raise SystemExit(f"No benchmark files in {OUT}")
    with open(OUT.parent / "fps_bench.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    log(f"Saved results/tables/fps_bench.csv ({len(rows)} rows)")

    bench = {(r["det"], r["tracker"]): r for r in rows if r["tracker"]}
    matrix = list(csv.DictReader(open(MATRIX)))
    updated = 0
    for row in matrix:
        b = bench.get((row["det"], row["tracker"]))
        if b:
            for col in ("FPS_det", "FPS_tracker", "FPS_tracker_front_end", "FPS_e2e"):
                row[col] = b[col]
            updated += 1
    with open(MATRIX, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(matrix[0]))
        w.writeheader()
        w.writerows(matrix)
    log(f"Updated the FPS columns of {updated} rows in {MATRIX.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=["det", "track", "merge"])
    ap.add_argument("--det")
    ap.add_argument("--seqs", nargs="+", default=SEQS)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--runs", nargs="+", help="tracker folders, default: every run of this detector in the matrix")
    args = ap.parse_args()
    if args.part == "merge":
        return merge()
    if not args.det:
        raise SystemExit("--det is required")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.part == "det":
        result = bench_det(args.det, args.seqs, args.repeats, args.warmup)
    else:
        result = bench_track(args.det, args.seqs, args.repeats, args.runs)
    (OUT / f"{args.part}_{args.det}.json").write_text(json.dumps(result, indent=2))
    log(f"Saved {(OUT / f'{args.part}_{args.det}.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
