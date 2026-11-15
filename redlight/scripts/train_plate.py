#!/usr/bin/env python
"""Fine-tune a YOLO plate detector; all hyper-parameters come from a JSON config.

Example:
    python scripts/train_plate.py --config config/plate_train.json
Writes best weights to ``weights/<name>.pt`` and metrics to ``weights/<name>_metrics.json``.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import resolve_path  # noqa: E402
from src.common.logging_utils import setup_logging  # noqa: E402
from src.detect.vehicle import resolve_device  # noqa: E402

log = logging.getLogger("train_plate")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config/plate_train.json")
    args = p.parse_args()
    setup_logging("INFO")
    cfg = json.loads(resolve_path(args.config).read_text())

    from ultralytics import YOLO

    device = resolve_device(cfg.get("device", "auto"))
    data = resolve_path(cfg["data"])
    base = cfg["base_weights"]
    base_path = resolve_path(base)
    model = YOLO(str(base_path) if base_path.exists() else base)
    train_kwargs = dict(cfg.get("train", {}))
    project = resolve_path(cfg.get("project", "runs/plate"))
    name = cfg.get("name", "plate")
    log.info("Training %s on %s (device=%s) -> %s/%s", base, data, device, project, name)
    model.train(data=str(data), device=device, project=str(project), name=name, exist_ok=True, **train_kwargs)

    run_dir = project / name
    best = run_dir / "weights" / "best.pt"
    out_weights = resolve_path(cfg.get("output_weights", f"weights/{name}.pt"))
    out_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out_weights)
    log.info("Best weights copied to %s", out_weights)

    metrics = {}
    best_model = YOLO(str(best))
    for split in ("val", "test"):
        try:
            r = best_model.val(data=str(data), split=split, device=device, imgsz=train_kwargs.get("imgsz", 640),
                               project=str(project), name=f"{name}_{split}", exist_ok=True, verbose=False, plots=False)
        except Exception as exc:  # e.g. split missing
            log.warning("Validation on %s skipped: %s", split, exc)
            continue
        metrics[split] = {
            "precision": round(float(r.box.mp), 4), "recall": round(float(r.box.mr), 4),
            "mAP50": round(float(r.box.map50), 4), "mAP50_95": round(float(r.box.map), 4),
        }
        log.info("%s: %s", split, metrics[split])
    out_metrics = out_weights.with_name(out_weights.stem + "_metrics.json")
    out_metrics.write_text(json.dumps({"base_weights": base, "data": str(data), "train": train_kwargs, "metrics": metrics}, indent=2))
    log.info("Metrics written to %s", out_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
