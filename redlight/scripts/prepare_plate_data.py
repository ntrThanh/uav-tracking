#!/usr/bin/env python
"""Merge one or more YOLO-format plate datasets into a single single-class dataset.

Every ``images/`` directory with a sibling ``labels/`` directory under each source root
is scanned. All classes are mapped to class 0 ("plate"), malformed boxes are dropped,
duplicate images (by content hash) are kept once, and images whose label file is empty
are kept only as a small fraction of negatives. Directories whose path contains
``test`` go to the test split untouched; the rest is split into train/val by hash.

Example:
    python scripts/prepare_plate_data.py --sources data/plates/tanhphp/extracted data/plates/topkek69/extracted \
        --out data/plates/merged
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import resolve_path  # noqa: E402
from src.common.logging_utils import setup_logging  # noqa: E402

log = logging.getLogger("prepare_plate_data")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", nargs="+", required=True, help="dataset root directories to scan")
    p.add_argument("--out", required=True, help="output dataset directory")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--keep-empty-frac", type=float, default=0.1, help="fraction of empty-label images kept as negatives")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--link", action="store_true", help="hard-link images instead of copying")
    return p.parse_args()


def find_pairs(root: Path) -> list[tuple[Path, Path]]:
    """(images_dir, labels_dir) pairs under ``root``."""
    pairs = []
    for images in sorted(root.rglob("images")):
        labels = images.parent / "labels"
        if images.is_dir() and labels.is_dir():
            pairs.append((images, labels))
    return pairs


def clean_label(text: str) -> list[str]:
    """Keep well-formed YOLO rows, mapping every class id to 0."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            vals = [float(v) for v in parts[1:5]]
        except ValueError:
            continue
        if not all(0.0 <= v <= 1.0 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
            continue
        rows.append("0 " + " ".join(f"{v:.6f}" for v in vals))
    return rows


def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    args = parse_args()
    setup_logging("INFO")
    rng = random.Random(args.seed)
    out = resolve_path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val", "test"):
        (out / split / "images").mkdir(parents=True)
        (out / split / "labels").mkdir(parents=True)

    seen: set[str] = set()
    stats: Counter = Counter()
    per_source: dict[str, Counter] = {}
    for src in args.sources:
        src_root = resolve_path(src)
        src_stats = per_source.setdefault(str(src_root), Counter())
        pairs = find_pairs(src_root)
        if not pairs:
            log.warning("No images/labels pairs under %s", src_root)
        for images_dir, labels_dir in pairs:
            is_test = "test" in images_dir.parent.name.lower() or "test" in str(images_dir.relative_to(src_root)).lower()
            for img in sorted(images_dir.iterdir()):
                if img.suffix.lower() not in IMAGE_EXTS:
                    continue
                label = labels_dir / (img.stem + ".txt")
                if not label.exists():
                    src_stats["no_label_file"] += 1
                    continue
                rows = clean_label(label.read_text(errors="ignore"))
                if not rows:
                    src_stats["empty"] += 1
                    if rng.random() >= args.keep_empty_frac:
                        continue
                    src_stats["empty_kept"] += 1
                digest = file_hash(img)
                if digest in seen:
                    src_stats["duplicate"] += 1
                    continue
                seen.add(digest)
                if is_test:
                    split = "test"
                else:
                    split = "val" if int(digest[:8], 16) / 0xFFFFFFFF < args.val_frac else "train"
                name = f"{digest[:12]}{img.suffix.lower()}"
                dst_img = out / split / "images" / name
                if args.link:
                    try:
                        dst_img.hardlink_to(img)
                    except OSError:
                        shutil.copy2(img, dst_img)
                else:
                    shutil.copy2(img, dst_img)
                (out / split / "labels" / f"{digest[:12]}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
                src_stats[split] += 1
                stats[split] += 1
                stats[f"boxes_{split}"] += len(rows)

    yaml_text = (
        f"path: {out}\ntrain: train/images\nval: val/images\n"
        + ("test: test/images\n" if stats["test"] else "")
        + "nc: 1\nnames: ['plate']\n"
    )
    (out / "data.yaml").write_text(yaml_text)
    summary = {"splits": dict(stats), "per_source": {k: dict(v) for k, v in per_source.items()}}
    (out / "prepare_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Merged dataset at %s", out)
    for k, v in per_source.items():
        log.info("source %s: %s", k, dict(v))
    log.info("splits: %s", dict(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
