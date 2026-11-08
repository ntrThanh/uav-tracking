"""Stage 2: pick the detection threshold of every detector x tracker pair by HOTA on val.

Reads results/tables/matrix_val_mean.csv, where the tuning runs are the tracker folders <tracker>_thrXX made by
    python -m src.inference.run_tracker --det <det> --split val --tracker <tracker> --det-thresh 0.XX --tag thrXX
and writes results/tables/tuned_val.csv: for each pair the best threshold, its row, and the HOTA of every threshold.

--grid limits the choice to some thresholds, e.g. --grid 0.3 0.4 0.5 0.6 for the grid of the plan.

Usage: python -m src.evaluation.select_thresh [--grid 0.3 0.4 0.5 0.6] [--out results/tables/tuned_val.csv]
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAG = re.compile(r"^(?P<tracker>.+)_thr(?P<thr>\d+)$")


def tag_value(digits: str) -> float:
    """thr03 -> 0.3, thr045 -> 0.45: the digits after the leading zero are the decimals."""
    return round(int(digits) / 10 ** (len(digits) - 1), 4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", type=Path, default=ROOT / "results" / "tables" / "matrix_val_mean.csv")
    ap.add_argument("--grid", type=float, nargs="+", help="allowed thresholds, default: every tuned value")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "tables" / "tuned_val.csv")
    args = ap.parse_args()

    groups: dict[tuple[str, str], list[tuple[float, dict]]] = {}
    for row in csv.DictReader(open(args.matrix)):
        m = TAG.match(row["tracker"])
        if not m:
            continue
        thr = tag_value(m["thr"])
        if args.grid and not any(abs(thr - g) < 1e-9 for g in args.grid):
            continue
        groups.setdefault((row["det"], m["tracker"]), []).append((thr, row))
    if not groups:
        raise SystemExit(f"No tuning runs (<tracker>_thrXX) in {args.matrix}")

    out = []
    for (det, tracker), runs in sorted(groups.items()):
        runs.sort()
        best_thr, best = max(runs, key=lambda r: float(r[1]["HOTA"]))
        sweep = " ".join(f"{t:g}:{float(r['HOTA']):.2f}" for t, r in runs)
        out.append({**best, "tracker": tracker, "det_thresh": f"{best_thr:g}", "run": best["tracker"],
                    "HOTA_by_thresh": sweep})
    columns = ["det", "tracker", "det_thresh", "run"] + [c for c in out[0] if c not in
                                                          {"det", "tracker", "det_thresh", "run", "HOTA_by_thresh"}]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns + ["HOTA_by_thresh"])
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(f"{r['det']:10s} {r['tracker']:10s} thr {r['det_thresh']}  HOTA {r['HOTA']}  ({r['HOTA_by_thresh']})")


if __name__ == "__main__":
    main()
