"""Figures of the detection and tracking results, from the CSV tables only (no model, no GPU).

Inputs (results/tables/): matrix_testdev_mean.csv, matrix_testdev_per_class.csv, matrix_val_mean.csv,
tuned_val.csv, det_test-dev.csv, model_stats.csv, fps_bench.csv (optional).
Output: results/figures/*.png (and the small tables behind each figure as *.csv next to them).

Colors: one fixed categorical slot per detector (blue, orange, aqua; validated all-pairs), the tracker is the marker
shape; magnitude grids use the single-hue blue ramp. Oracle is neutral gray.

Usage (from the project root): python -m src.evaluation.make_figures
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import matplotlib.legend

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
OUT = ROOT / "results" / "figures"

DETS = ["yolo11m", "frcnn_r50", "rfdetr_m"]
DET_LABEL = {"yolo11m": "YOLO11m", "frcnn_r50": "Faster R-CNN", "rfdetr_m": "RF-DETR", "oracle": "Oracle (GT)"}
DET_COLOR = {"yolo11m": "#2a78d6", "frcnn_r50": "#eb6834", "rfdetr_m": "#1baf7a", "oracle": "#8a8986"}
TRACKERS = ["sort", "deepsort", "bytetrack", "ocsort", "botsort"]
TRK_LABEL = {"sort": "SORT", "deepsort": "DeepSORT", "bytetrack": "ByteTrack", "ocsort": "OC-SORT",
             "botsort": "BoT-SORT"}
TRK_MARKER = {"sort": "o", "deepsort": "s", "bytetrack": "^", "ocsort": "D", "botsort": "P"}
CLASSES = ["pedestrian", "car", "van", "truck", "bus"]
BLUE = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2, "savefig.dpi": 200,
})


def read(name: str) -> list[dict]:
    path = TABLES / name
    return list(csv.DictReader(open(path))) if path.is_file() else []


def f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def base(tracker: str) -> str:
    return tracker.split("_thr")[0]


def tuned_runs() -> dict[tuple[str, str], str]:
    return {(r["det"], r["tracker"]): r["run"] for r in read("tuned_val.csv")}


def pick(rows, det, run, split="test-dev"):
    return next((r for r in rows if r["det"] == det and r["tracker"] == run and r["split"] == split), None)


def save(fig, name: str, table: list[dict] | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    legends = fig.findobj(matplotlib.legend.Legend)     # legends outside the axes stay inside the image
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", bbox_extra_artists=legends)
    plt.close(fig)
    if table:
        with open(OUT / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
    print(f"[+] results/figures/{name}.png")


def tracker_legend(ax, **kw):
    handles = [plt.Line2D([], [], marker=TRK_MARKER[t], linestyle="", color=INK2, markersize=7, label=TRK_LABEL[t])
               for t in TRACKERS]
    return ax.legend(handles=handles, title="Tracker", **kw)


def det_legend(ax, dets=DETS, **kw):
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=DET_COLOR[d], markersize=8, label=DET_LABEL[d])
               for d in dets]
    return ax.legend(handles=handles, title="Detector", **kw)


# ----------------------------------------------------------------------------------------------- figures
def heatmaps(matrix, tuned):
    for metric in ["HOTA", "IDF1", "MOTA"]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), constrained_layout=True)
        table = []
        grids = []
        for variant in ["default", "tuned"]:
            grid = np.full((len(DETS), len(TRACKERS)), np.nan)
            for i, d in enumerate(DETS):
                for j, t in enumerate(TRACKERS):
                    run = t if variant == "default" else tuned.get((d, t), t)
                    r = pick(matrix, d, run)
                    grid[i, j] = f(r[metric]) if r else np.nan
                    table.append({"variant": variant, "det": d, "tracker": t, "run": run, metric: grid[i, j]})
            grids.append(grid)
        lo, hi = np.nanmin(grids), np.nanmax(grids)
        for ax, grid, variant in zip(axes, grids, ["Default thresholds", "Thresholds tuned on val"]):
            im = ax.imshow(grid, cmap=BLUE, vmin=lo, vmax=hi, aspect="auto")
            ax.grid(False)
            ax.set_xticks(range(len(TRACKERS)), [TRK_LABEL[t] for t in TRACKERS])
            ax.set_yticks(range(len(DETS)), [DET_LABEL[d] for d in DETS])
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    v = grid[i, j]
                    light = (v - lo) / max(hi - lo, 1e-9) > 0.55
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=10,
                            color="white" if light else INK, fontweight="bold" if v == np.nanmax(grid[i]) else None)
            ax.set_title(variant, fontsize=11)
        fig.colorbar(im, ax=axes, shrink=0.85, label=metric)
        fig.suptitle(f"{metric} on VisDrone2019-MOT test-dev (mean over 5 classes)", x=0.01, ha="left",
                     fontweight="bold")
        save(fig, f"heatmap_{metric.lower()}_testdev", table)


def scatter(ax, points, x, y):
    for p in points:
        ax.scatter(f(p[x]), f(p[y]), s=70, marker=TRK_MARKER[p["base"]], color=DET_COLOR[p["det"]],
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)


def official_points(matrix, tuned):
    pts = []
    for d in DETS:
        for t in TRACKERS:
            r = pick(matrix, d, tuned.get((d, t), t))
            if r:
                pts.append({**r, "base": t})
    return pts


def tradeoff(matrix, tuned):
    pts = [p for p in official_points(matrix, tuned) if np.isfinite(f(p["FPS_e2e"]))]
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    scatter(ax, pts, "FPS_e2e", "HOTA")
    ax.set_xscale("log")
    ax.set_xlabel("End-to-end FPS (detector + ReID/CMC + association), log scale")
    ax.set_ylabel("HOTA")
    ax.set_title("Speed vs accuracy, tuned thresholds, test-dev")
    best = max(pts, key=lambda p: f(p["HOTA"]))
    fast = max(pts, key=lambda p: f(p["FPS_e2e"]))
    for p, dx, ha in [(best, 8, "left"), (fast, -8, "right")]:
        ax.annotate(f"{DET_LABEL[p['det']]} + {TRK_LABEL[p['base']]}", (f(p["FPS_e2e"]), f(p["HOTA"])),
                    xytext=(dx, -12), textcoords="offset points", fontsize=9, color=INK2, ha=ha)
    leg = det_legend(ax, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.add_artist(leg)
    tracker_legend(ax, loc="upper left", bbox_to_anchor=(1.01, 0.62))
    save(fig, "tradeoff_fps_hota", [{k: p[k] for k in ["det", "tracker", "HOTA", "FPS_e2e"]} for p in pts])


def map_vs_hota(matrix, tuned, det_rows):
    m = {r["det"]: f(r["mAP50_95_track5"]) for r in det_rows if r["variant"] == "ign"}
    pts = [p for p in official_points(matrix, tuned) if p["det"] in m]
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for p in pts:
        p["mAP"] = m[p["det"]]
    scatter(ax, pts, "mAP", "HOTA")
    ax.set_xlabel("Detector mAP50-95 on the 5 tracking classes (test-dev, ignored regions removed)")
    ax.set_ylabel("HOTA")
    ax.set_title("Does a better detector give better tracking?")
    leg = det_legend(ax, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.add_artist(leg)
    tracker_legend(ax, loc="upper left", bbox_to_anchor=(1.01, 0.62))
    save(fig, "map_vs_hota", [{"det": p["det"], "tracker": p["tracker"], "mAP50_95_track5": p["mAP"],
                               "HOTA": p["HOTA"]} for p in pts])


def deta_assa(matrix, tuned):
    pts = official_points(matrix, tuned)       # oracle (DetA ~90-97) is in oracle_gap.png, it would squash this view
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    scatter(ax, pts, "DetA", "AssA")
    ax.set_xlabel("DetA (detection quality)")
    ax.set_ylabel("AssA (association quality)")
    ax.set_title("HOTA = sqrt(DetA x AssA), tuned thresholds, test-dev")
    leg = det_legend(ax, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.add_artist(leg)
    tracker_legend(ax, loc="upper left", bbox_to_anchor=(1.01, 0.62))
    save(fig, "deta_vs_assa", [{k: p[k] for k in ["det", "tracker", "DetA", "AssA", "HOTA"]} for p in pts])


def per_class(per_cls, matrix, tuned):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.26
    table = []
    for k, d in enumerate(DETS):
        cands = [pick(matrix, d, tuned.get((d, t), t)) for t in TRACKERS]
        best = max((c for c in cands if c), key=lambda r: f(r["HOTA"]))
        vals = []
        for c in CLASSES:
            r = next((x for x in per_cls if x["det"] == d and x["tracker"] == best["tracker"]
                      and x["split"] == "test-dev" and x["class"] == c), None)
            vals.append(f(r["HOTA"]) if r else np.nan)
            table.append({"det": d, "run": best["tracker"], "class": c, "HOTA": vals[-1]})
        x = np.arange(len(CLASSES)) + (k - 1) * (width + 0.02)
        ax.bar(x, vals, width, color=DET_COLOR[d], label=f"{DET_LABEL[d]} + {TRK_LABEL[base(best['tracker'])]}",
               edgecolor=SURFACE, linewidth=2)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.8, f"{v:.0f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_ylabel("HOTA")
    ax.set_title("HOTA per class, best tracker of each detector (tuned, test-dev)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(axis="x", visible=False)
    save(fig, "per_class_hota", table)


def threshold_sweep(val):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4), sharey=True, constrained_layout=True)
    table = []
    for ax, t in zip(axes, TRACKERS):
        for d in DETS:
            pts = []
            for r in val:
                if r["det"] == d and base(r["tracker"]) == t and "_thr" in r["tracker"]:
                    digits = r["tracker"].split("_thr")[1]
                    thr = int(digits) / 10 ** (len(digits) - 1)
                    pts.append((thr, f(r["HOTA"])))
                    table.append({"det": d, "tracker": t, "det_thresh": thr, "HOTA": f(r["HOTA"])})
            pts.sort()
            if pts:
                ax.plot(*zip(*pts), marker="o", markersize=4, color=DET_COLOR[d], label=DET_LABEL[d])
        ax.axvspan(0.8, 0.92, color=GRID, alpha=0.5, lw=0)
        ax.set_title(TRK_LABEL[t], fontsize=11)
        ax.set_xlabel("high det threshold")
    axes[0].set_ylabel("HOTA (val)")
    axes[-1].legend(loc="lower left", fontsize=8)
    fig.suptitle("Threshold sweep on val (shaded: extended grid, Faster R-CNN only)", x=0.01, ha="left",
                 fontweight="bold")
    save(fig, "threshold_sweep_val", table)


def oracle_gap(matrix, tuned):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.2
    series = ["oracle"] + DETS
    table = []
    for k, d in enumerate(series):
        vals = []
        for t in TRACKERS:
            r = pick(matrix, d, t if d == "oracle" else tuned.get((d, t), t))
            vals.append(f(r["HOTA"]) if r else np.nan)
            table.append({"det": d, "tracker": t, "HOTA": vals[-1]})
        x = np.arange(len(TRACKERS)) + (k - 1.5) * (width + 0.015)
        ax.bar(x, vals, width, color=DET_COLOR[d], label=DET_LABEL[d], edgecolor=SURFACE, linewidth=2)
    ax.set_xticks(range(len(TRACKERS)), [TRK_LABEL[t] for t in TRACKERS])
    ax.set_ylabel("HOTA")
    ax.set_title("Real detectors vs ground-truth detections (test-dev, tuned)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(axis="x", visible=False)
    save(fig, "oracle_gap", table)


def detection_table(det_rows, stats):
    s = {r["det"]: r for r in stats}
    rows = [r for r in det_rows if r["variant"] == "ign" and r["det"] in DETS]
    if not rows:
        return
    metrics = ["mAP50_95", "mAP50", "AP_small", "AP_medium", "AP_large", "AR_500", "F1@0.5"]
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.26
    for k, d in enumerate(DETS):
        r = next(x for x in rows if x["det"] == d)
        vals = [f(r[m]) for m in metrics]
        x = np.arange(len(metrics)) + (k - 1) * (width + 0.02)
        lab = DET_LABEL[d]
        if d in s:
            lab += f" ({s[d]['params_M']} M, {s[d]['GFLOPs_mean']} GFLOPs)"
        ax.bar(x, vals, width, color=DET_COLOR[d], label=lab, edgecolor=SURFACE, linewidth=2)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.6, f"{v:.1f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(range(len(metrics)), metrics)
    ax.set_ylabel("%")
    ax.set_title("Detection on MOT test-dev, one pycocotools protocol (ignored regions removed)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(axis="x", visible=False)
    save(fig, "detection_testdev")


def main() -> None:
    matrix = read("matrix_testdev_mean.csv")
    tuned = tuned_runs()
    det_rows = read("det_test-dev.csv")
    heatmaps(matrix, tuned)
    tradeoff(matrix, tuned)
    map_vs_hota(matrix, tuned, det_rows)
    deta_assa(matrix, tuned)
    per_class(read("matrix_testdev_per_class.csv"), matrix, tuned)
    threshold_sweep(read("matrix_val_mean.csv"))
    oracle_gap(matrix, tuned)
    detection_table(det_rows, read("model_stats.csv"))


if __name__ == "__main__":
    main()
