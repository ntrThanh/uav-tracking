"""Evaluate tracking results with TrackEval (HOTA, CLEAR, Identity), per class, then averaged over the 5 classes.

Protocol (VisDrone-MOT):
  * Only the 5 evaluated classes [1, 4, 5, 6, 9]; each class is evaluated on its own, so a car track can never match
    a pedestrian, and the class score is the TrackEval COMBINED_SEQ result over all sequences of the split.
  * Ignored regions: in every frame, GT boxes and tracker boxes with more than half of their area inside the union of
    that frame's ignored regions are removed (same rule as dropObjectsInIgr in the official VisDrone toolkit).
  * Mean over classes: percentages are averaged (macro mean), counts (IDSW, FP, FN, Frag, MT, ML) are summed.

Input: every run folder results/tracks/<det>/<tracker>[_tag]/<split>/ with run_meta.json (made by run_tracker.py).
Output:
  results/tables/track/<split>/<det>__<tracker>.csv   5 class rows + the mean row of one run
  results/tables/matrix_<split>_mean.csv              one row per run (merged with the rows already there)
  results/tables/matrix_<split>_per_class.csv         one row per run and class

Usage (from the project root):
    python -m src.evaluation.eval_track --split val                         # every run on val
    python -m src.evaluation.eval_track --split val --det oracle            # only some detectors / trackers
    python -m src.evaluation.eval_track --split val --det yolo11m --tracker bytetrack botsort
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data" / "tools"))
from visdrone import CLASS_NAMES, TRACK_CLASSES, log  # noqa: E402

IGNORE_OVERLAP = 0.5
PERCENT = ["HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA",
           "MOTA", "MOTP", "IDF1", "IDP", "IDR", "MT%", "ML%"]
COUNTS = ["IDSW", "Frag", "MT", "ML", "FP", "FN", "GT_Dets", "GT_IDs"]
SPEED = ["FPS_det", "FPS_tracker", "FPS_tracker_front_end", "FPS_e2e"]
KEY = ["det", "tracker", "split"]


@dataclass
class Run:
    det: str
    tracker: str            # folder name, e.g. bytetrack or bytetrack_thr04
    folder: Path

    @property
    def label(self) -> str:
        return f"{self.det}__{self.tracker}"


# ----------------------------------------------------------------------------------------------- ignored regions
class IgnoredMask:
    """Union of the ignored regions of one frame as a pixel mask, built for one frame at a time."""

    def __init__(self, regions: np.ndarray, width: int, height: int):
        self.width, self.height = width, height
        frames = regions[:, 0].astype(int) if len(regions) else np.zeros(0, dtype=int)
        self.regions = {int(f): regions[frames == f, 2:6] for f in np.unique(frames)}

    def _pixels(self, x, y, w, h):
        x1, y1 = int(np.clip(np.floor(x), 0, self.width)), int(np.clip(np.floor(y), 0, self.height))
        x2, y2 = int(np.clip(np.ceil(x + w), 0, self.width)), int(np.clip(np.ceil(y + h), 0, self.height))
        return x1, y1, x2, y2

    def _mask(self, frame: int) -> np.ndarray:
        mask = np.zeros((self.height, self.width), dtype=bool)
        for x, y, w, h in self.regions[frame]:
            x1, y1, x2, y2 = self._pixels(x, y, w, h)
            mask[y1:y2, x1:x2] = True
        return mask

    def keep(self, rows: np.ndarray) -> np.ndarray:
        """rows: frame, id, x, y, w, h, ... Returns the mask of rows that are NOT mostly inside an ignored region."""
        keep = np.ones(len(rows), dtype=bool)
        frames = rows[:, 0].astype(int) if len(rows) else np.zeros(0, dtype=int)
        for frame in set(np.unique(frames).tolist()) & set(self.regions):
            mask = self._mask(frame)
            for i in np.flatnonzero(frames == frame):
                x1, y1, x2, y2 = self._pixels(*rows[i, 2:6])
                area = max((x2 - x1) * (y2 - y1), 1)
                keep[i] = mask[y1:y2, x1:x2].sum() / area <= IGNORE_OVERLAP
        return keep


# ----------------------------------------------------------------------------------------------- data preparation
def load_table(path: Path, cols: int) -> np.ndarray:
    if not path.is_file() or path.stat().st_size == 0:
        return np.zeros((0, cols))
    return np.loadtxt(path, delimiter=",", ndmin=2)


def write_mot(path: Path, rows: np.ndarray) -> None:
    """TrackEval MotChallenge2DBox layout with every row marked as class 1 (pedestrian) and conf 1."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for frame, tid, x, y, w, h in rows[:, :6]:
            f.write(f"{int(frame)},{int(tid)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1\n")


def read_seqinfo(sequence_dir: Path) -> dict:
    values = dict(line.strip().split("=", 1) for line in open(sequence_dir / "seqinfo.ini") if "=" in line)
    return {"length": int(values["seqLength"]), "width": int(values["imWidth"]), "height": int(values["imHeight"])}


def prepare(split: str, mot_root: Path, runs: list[Run], work: Path) -> dict[str, int]:
    """Writes work/gt/<cls>/<seq>.txt and work/trackers/<cls>/<label>/data/<seq>.txt."""
    sequences = sorted(p for p in (mot_root / split).iterdir() if p.is_dir())
    lengths = {}
    for seq in sequences:
        info = read_seqinfo(seq)
        lengths[seq.name] = info["length"]
        mask = IgnoredMask(load_table(seq / "gt" / "ignored.txt", 9), info["width"], info["height"])
        gt = load_table(seq / "gt" / "gt.txt", 9)
        gt = gt[mask.keep(gt)]
        for cls in TRACK_CLASSES:
            write_mot(work / "gt" / CLASS_NAMES[cls] / f"{seq.name}.txt", gt[gt[:, 7] == cls])
        for run in runs:
            path = run.folder / f"{seq.name}.txt"
            if not path.is_file():
                raise SystemExit(f"{run.label}: missing {path}")
            res = load_table(path, 10)
            res = res[mask.keep(res)]
            for cls in TRACK_CLASSES:
                write_mot(work / "trackers" / CLASS_NAMES[cls] / run.label / "data" / f"{seq.name}.txt",
                          res[res[:, 7] == cls])
    return lengths


# ----------------------------------------------------------------------------------------------- TrackEval
def _numpy_aliases() -> None:
    """Older TrackEval releases still use np.float / np.int / np.bool, removed in NumPy 1.24."""
    for name, value in {"float": float, "int": int, "bool": bool}.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def run_trackeval(work: Path, cls_name: str, labels: list[str], lengths: dict[str, int], cores: int) -> dict:
    _numpy_aliases()
    import trackeval

    eval_cfg = trackeval.Evaluator.get_default_eval_config()
    eval_cfg.update({"USE_PARALLEL": cores > 1, "NUM_PARALLEL_CORES": cores, "PRINT_RESULTS": False,
                     "PRINT_CONFIG": False, "TIME_PROGRESS": False, "OUTPUT_SUMMARY": False,
                     "OUTPUT_DETAILED": False, "PLOT_CURVES": False, "DISPLAY_LESS_PROGRESS": True,
                     "LOG_ON_ERROR": str(work / "error_log.txt")})
    data_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    data_cfg.update({"GT_FOLDER": str(work / "gt" / cls_name), "TRACKERS_FOLDER": str(work / "trackers" / cls_name),
                     "OUTPUT_FOLDER": str(work / "output" / cls_name), "TRACKERS_TO_EVAL": labels,
                     "CLASSES_TO_EVAL": ["pedestrian"], "SKIP_SPLIT_FOL": True, "DO_PREPROC": False,
                     "SEQ_INFO": lengths, "GT_LOC_FORMAT": "{gt_folder}/{seq}.txt", "PRINT_CONFIG": False})
    evaluator = trackeval.Evaluator(eval_cfg)
    quiet = {"PRINT_CONFIG": False}
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(quiet), trackeval.metrics.Identity(quiet)]
    results, messages = evaluator.evaluate([trackeval.datasets.MotChallenge2DBox(data_cfg)], metrics)
    results = results["MotChallenge2DBox"]
    failed = [label for label in labels if messages["MotChallenge2DBox"].get(label) != "Success"]
    if failed:
        raise RuntimeError(f"TrackEval failed for {failed}: {[messages['MotChallenge2DBox'][f] for f in failed]}")
    return {label: results[label]["COMBINED_SEQ"]["pedestrian"] for label in labels}


def class_row(res: dict) -> dict:
    hota, clear, ident = res["HOTA"], res["CLEAR"], res["Identity"]
    gt_ids = clear["MT"] + clear["PT"] + clear["ML"]
    row = {k: 100 * float(np.mean(hota[k])) for k in ["HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA"]}
    row.update({"MOTA": 100 * clear["MOTA"], "MOTP": 100 * clear["MOTP"], "IDF1": 100 * ident["IDF1"],
                "IDP": 100 * ident["IDP"], "IDR": 100 * ident["IDR"],
                "MT%": 100 * clear["MT"] / max(gt_ids, 1), "ML%": 100 * clear["ML"] / max(gt_ids, 1),
                "IDSW": clear["IDSW"], "Frag": clear["Frag"], "MT": clear["MT"], "ML": clear["ML"],
                "FP": clear["CLR_FP"], "FN": clear["CLR_FN"], "GT_Dets": clear["CLR_TP"] + clear["CLR_FN"],
                "GT_IDs": gt_ids})
    row.update({k: int(row[k]) for k in COUNTS})
    return row


def mean_row(rows: list[dict]) -> dict:
    valid = [r for r in rows if r["GT_Dets"] > 0]      # a class absent from the split has no score
    out = {k: float(np.mean([r[k] for r in valid])) for k in PERCENT}
    out.update({k: int(sum(r[k] for r in rows)) for k in COUNTS})
    out["classes"] = len(valid)
    return out


def speed(run: Run, split: str) -> dict:
    meta = json.loads((run.folder / "run_meta.json").read_text())
    seqs = meta["sequences"].values()
    frames = sum(s["frames"] for s in seqs)
    assoc = sum(s["association_seconds"] for s in seqs)
    front = sum(s["front_end_seconds"] for s in seqs)
    out = {"FPS_tracker": frames / max(assoc, 1e-9), "FPS_tracker_front_end": frames / max(assoc + front, 1e-9),
           "FPS_det": None, "FPS_e2e": None}
    timing = ROOT / "results" / "dets" / run.det / split / "timing.json"
    det_ms = json.loads(timing.read_text()).get("ms_per_frame") if timing.is_file() else None
    if det_ms:
        out["FPS_det"] = 1000 / det_ms
        out["FPS_e2e"] = frames / (frames * det_ms / 1000 + assoc + front)
    return out


# ----------------------------------------------------------------------------------------------- tables
def fmt(value):
    return f"{value:.2f}" if isinstance(value, float) else value


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows({k: fmt(v) for k, v in r.items()} for r in rows)


def merge_csv(path: Path, rows: list[dict], columns: list[str], key: list[str]) -> None:
    """Replace the rows of these runs, keep the others, sort by detector then tracker."""
    old = list(csv.DictReader(open(path))) if path.is_file() else []
    new_keys = {tuple(str(r[k]) for k in key) for r in rows}
    kept = [r for r in old if tuple(r[k] for k in key) not in new_keys]
    merged = sorted(kept + [{k: fmt(v) for k, v in r.items()} for r in rows], key=lambda r: tuple(str(r[k]) for k in key))
    write_csv(path, merged, columns)


def find_runs(split: str, dets: list[str] | None, trackers: list[str] | None) -> list[Run]:
    runs = []
    for meta in sorted((ROOT / "results" / "tracks").glob(f"*/*/{split}/run_meta.json")):
        folder = meta.parent
        run = Run(folder.parents[1].name, folder.parent.name, folder)
        if (dets is None or run.det in dets) and (trackers is None or run.tracker in trackers):
            runs.append(run)
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["val", "test-dev"])
    ap.add_argument("--det", nargs="+", help="detector folders, default: all")
    ap.add_argument("--tracker", nargs="+", help="tracker folders (with tag), default: all")
    ap.add_argument("--mot-root", type=Path, default=ROOT / "data" / "processed" / "mot")
    ap.add_argument("--tables", type=Path, default=ROOT / "results" / "tables")
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--keep-work", action="store_true", help="keep results/trackeval/<split> for inspection")
    args = ap.parse_args()

    runs = find_runs(args.split, args.det, args.tracker)
    if not runs:
        raise SystemExit(f"No finished runs (run_meta.json) under results/tracks/*/*/{args.split}")
    log(f"Evaluating {len(runs)} runs on {args.split}: {', '.join(r.label for r in runs)}")

    work = ROOT / "results" / "trackeval" / args.split
    shutil.rmtree(work, ignore_errors=True)
    lengths = prepare(args.split, args.mot_root, runs, work)

    per_class: dict[str, list[dict]] = {r.label: [] for r in runs}
    for cls in TRACK_CLASSES:
        name = CLASS_NAMES[cls]
        results = run_trackeval(work, name, [r.label for r in runs], lengths, args.cores)
        for run in runs:
            per_class[run.label].append({"det": run.det, "tracker": run.tracker, "split": args.split,
                                         "class": name, **class_row(results[run.label])})
        log(f"{name} done", 1)

    split_name = args.split.replace("-", "")
    class_columns = KEY + ["class"] + PERCENT + COUNTS
    mean_columns = KEY + PERCENT + COUNTS + ["classes"] + SPEED
    means = []
    for run in runs:
        mean = {"det": run.det, "tracker": run.tracker, "split": args.split, **mean_row(per_class[run.label]),
                **speed(run, args.split)}
        means.append(mean)
        write_csv(args.tables / "track" / args.split / f"{run.label}.csv",
                  per_class[run.label] + [{**mean, "class": "mean"}], class_columns)
        log(f"{run.label}: HOTA {mean['HOTA']:.2f}  MOTA {mean['MOTA']:.2f}  IDF1 {mean['IDF1']:.2f}  "
            f"IDSW {mean['IDSW']}", 1)
    merge_csv(args.tables / f"matrix_{split_name}_mean.csv", means, mean_columns, KEY)
    merge_csv(args.tables / f"matrix_{split_name}_per_class.csv",
              [r for run in runs for r in per_class[run.label]], class_columns, KEY + ["class"])
    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    log(f"Tables in {args.tables}")


if __name__ == "__main__":
    main()
