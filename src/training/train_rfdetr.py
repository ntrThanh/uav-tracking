"""Train RF-DETR on the shared VisDrone2019-MOT data from a YAML config.

Usage (from the project root):
    python -m src.training.train_rfdetr --config src/configs/detectors/rfdetr_m_576.yaml
    python -m src.training.train_rfdetr --config ... --dry-run          # validate setup, no training
    python -m src.training.train_rfdetr --config ... --resume runs/det/<run>/last.ckpt
    python -m src.training.train_rfdetr --config ... --epochs 1 --run-name smoke_test
    python -m src.training.train_rfdetr --config ... --set batch_size=16 grad_accum_steps=1

Multi-GPU: when CUDA_VISIBLE_DEVICES lists several GPUs (e.g. "0,1"), training runs with DDP on all of them.
The effective batch is then batch_size x grad_accum_steps x number of GPUs.

The run directory runs/det/<run_name>/ receives the RF-DETR outputs plus run_config.yaml and env.txt.
One row per finished run is appended to results/runs_log.csv.
"""
from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data" / "tools"))
from visdrone import CLASS_NAMES  # noqa: E402

MODELS = {
    "nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",
    "base": "RFDETRBase", "large": "RFDETRLarge",
}
SPLIT_DIRS = ["train", "valid", "test"]
RUNS_LOG = ROOT / "results" / "runs_log.csv"
RUNS_LOG_FIELDS = ["run_id", "date", "user", "model", "resolution", "data", "epochs", "batch", "lr", "gpu",
                   "duration_h", "best_checkpoint", "output_dir", "status", "notes"]


def log(msg: str, level: int = 0) -> None:
    """Human-readable log: [+] for main items, [-] for details."""
    print("    " * level + ("[+] " if level == 0 else "[-] ") + msg, flush=True)


def load_config(path: Path, args: argparse.Namespace) -> dict:
    cfg = yaml.safe_load(path.read_text())
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.run_name:
        cfg["run_name"] = args.run_name
    if args.resume:
        cfg["train"]["resume"] = str(Path(args.resume).resolve())
    for item in args.set or []:
        key, _, value = item.partition("=")
        cfg["train"][key] = yaml.safe_load(value)
    n_gpus = len([g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()])
    if n_gpus > 1:
        cfg["train"].update(devices=n_gpus, strategy="ddp")
    return cfg


def is_main_process() -> bool:
    return os.environ.get("LOCAL_RANK", "0") == "0"


def check_dataset(dataset_dir: Path) -> None:
    log(f"Dataset: {dataset_dir.relative_to(ROOT)}")
    for split in SPLIT_DIRS:
        ann = dataset_dir / split / "_annotations.coco.json"
        if not ann.exists():
            raise SystemExit(f"Missing {ann}. Run: python data/tools/prepare.py rfdetr")
        coco = json.load(open(ann))
        ids = sorted(c["id"] for c in coco["categories"])
        if ids != sorted(CLASS_NAMES):
            raise SystemExit(f"{ann}: unexpected category ids {ids}")
        log(f"{split:<6} {len(coco['images']):>6,} images  {len(coco['annotations']):>8,} boxes", 1)


def gpu_info() -> str:
    import torch
    if not torch.cuda.is_available():
        return "cpu"
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return f"{len(names)}x {names[0]}"


def write_env(run_dir: Path, cfg: dict) -> None:
    import torch
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip() or "n/a"
    (run_dir / "env.txt").write_text(
        f"date: {datetime.now().isoformat(timespec='seconds')}\nhost: {platform.node()}\nuser: {getpass.getuser()}\n"
        f"python: {platform.python_version()}\ntorch: {torch.__version__}\nrfdetr: {version('rfdetr')}\n"
        f"gpu: {gpu_info()}\nCUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}\n"
        f"git_commit: {commit}\n\n{freeze}")


def append_runs_log(row: dict) -> None:
    RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not RUNS_LOG.exists()
    with open(RUNS_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUNS_LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--device", default="cuda", help="cuda, cuda:N or cpu (prefer CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--epochs", type=int, help="override train.epochs")
    ap.add_argument("--run-name", help="override run_name")
    ap.add_argument("--resume", help="checkpoint to resume from")
    ap.add_argument("--set", nargs="+", metavar="KEY=VALUE", help="override train.* keys, e.g. batch_size=16")
    ap.add_argument("--dry-run", action="store_true", help="check data, model and GPU, then exit")
    args = ap.parse_args()

    cfg = load_config(args.config, args)
    run_name, train_cfg = cfg["run_name"], dict(cfg["train"])
    dataset_dir = (ROOT / cfg["dataset_dir"]).resolve()
    run_dir = (ROOT / cfg["output_root"] / run_name).resolve()

    log(f"Run {run_name}")
    log(f"config: {args.config}", 1)
    log(f"model: RF-DETR {cfg['model']}, resolution {cfg['resolution']}", 1)
    log(f"output: {run_dir.relative_to(ROOT)}", 1)
    for k, v in train_cfg.items():
        log(f"{k}: {v}", 2)

    if cfg["resolution"] % 32:
        raise SystemExit(f"resolution {cfg['resolution']} is not a multiple of 32")
    check_dataset(dataset_dir)

    import rfdetr
    import torch
    log(f"Environment: torch {torch.__version__}, rfdetr {version('rfdetr')}, GPU {gpu_info()}")
    model = getattr(rfdetr, MODELS[cfg["model"]])()
    log("Pretrained COCO weights loaded", 1)

    if args.dry_run:
        from PIL import Image
        img = dataset_dir / "valid" / json.load(open(dataset_dir / "valid" / "_annotations.coco.json"))["images"][0][
            "file_name"]
        dets = model.predict(Image.open(img), threshold=0.3)
        log(f"Zero-shot forward pass on {img.name}: {len(dets)} detections (COCO classes)", 1)
        log("Dry run OK, nothing trained")
        return

    if is_main_process():
        write_env(run_dir, cfg)
    start = time.time()
    status = "finished"
    try:
        model.train(
            dataset_dir=str(dataset_dir),
            output_dir=str(run_dir),
            resolution=cfg["resolution"],
            device=args.device,
            class_names=[CLASS_NAMES[k] for k in sorted(CLASS_NAMES)],
            **train_cfg,
        )
        # Lightning swallows Ctrl+C and returns normally; checkpoint_best_total.pth is only written on a full finish.
        if not (run_dir / "checkpoint_best_total.pth").exists():
            status = "interrupted"
    except (KeyboardInterrupt, SystemExit):     # Lightning exits with SystemExit after handling Ctrl+C
        status = "interrupted"
        raise
    except Exception:
        status = "failed"
        raise
    finally:
        if is_main_process():
            hours = (time.time() - start) / 3600
            best = run_dir / "checkpoint_best_total.pth"
            append_runs_log({
                "run_id": run_name, "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "user": getpass.getuser(),
                "model": f"rfdetr_{cfg['model']}", "resolution": cfg["resolution"], "data": cfg["dataset_dir"],
                "epochs": train_cfg["epochs"], "batch": f"{train_cfg['batch_size']}x{train_cfg.get('grad_accum_steps', 1)}x{train_cfg.get('devices', 1)}gpu",
                "lr": train_cfg["lr"], "gpu": gpu_info(), "duration_h": f"{hours:.2f}",
                "best_checkpoint": str(best.relative_to(ROOT)) if best.exists() else "",
                "output_dir": str(run_dir.relative_to(ROOT)), "status": status, "notes": "",
            })
            log(f"Training {status} after {hours:.2f} h, logged to {RUNS_LOG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
