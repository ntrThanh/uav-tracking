"""Find VisDrone2019-DET train images that would leak MOT val/test-dev scenes into training.

Two rules, both written to data/splits/det_train_excluded.txt (name<TAB>reason):
  1. same video id as a MOT val or test-dev sequence (DET 0000137_... comes from MOT uav0000137_...)
  2. near-duplicate of a MOT val/test-dev frame by 64-bit difference hash (Hamming distance <= --max-dist)

Rule 2 is a safety net only. A visual check of the pairs at distance 1-8 showed different scenes with a similar
layout (mostly night streets), so the default --max-dist 0 removes only identical hashes; all real leaks found so
far share a video id and are caught by rule 1.

Usage (from the project root):
    python data/tools/det_overlap.py
    python data/tools/det_overlap.py --max-dist 4 --frame-step 3   # stricter, more false positives
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from visdrone import DET_EXCLUDED, DET_TRAIN, list_det_images, list_sequences, log, split_dir, video_id

DATA_DIR = Path(__file__).resolve().parents[1]
RAW = DATA_DIR / "raw"
HELD_OUT = ["val", "test-dev"]


def dhash(path: Path) -> int:
    """64-bit difference hash: robust to resizing and mild compression."""
    with Image.open(path) as im:
        im.draft("L", (im.width // 8, im.height // 8))
        px = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def hashes(paths: list[Path]) -> np.ndarray:
    with ProcessPoolExecutor(min(32, os.cpu_count() or 4)) as ex:
        return np.array(list(ex.map(dhash, paths, chunksize=64)), dtype=np.uint64)


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distance between two uint64 hash arrays -> (len(a), len(b))."""
    x = np.bitwise_xor(a[:, None], b[None, :])
    return np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1).sum(-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-dist", type=int, default=0, help="Hamming distance counted as near-duplicate (of 64)")
    ap.add_argument("--frame-step", type=int, default=5, help="compare against every n-th MOT frame")
    args = ap.parse_args()

    det = list_det_images(RAW)
    log(f"DET train: {len(det):,} images")
    excluded: dict[str, str] = {}

    log("Rule 1: shared video id with MOT val/test-dev")
    for split in HELD_OUT:
        vids = {video_id(s) for s in list_sequences(RAW, split)}
        hit = [n for n in det if video_id(n) in vids]
        for n in hit:
            excluded.setdefault(n, f"video_id:{split}")
        log(f"{split}: {len(hit)} images from videos {sorted({video_id(n) for n in hit})}", 1)

    log(f"Rule 2: near-duplicates (dHash distance <= {args.max_dist}), MOT frames every {args.frame_step}")
    mot_paths, mot_names = [], []
    for split in HELD_OUT:
        for seq in list_sequences(RAW, split):
            frames = sorted((split_dir(RAW, split) / "sequences" / seq).glob("*.jpg"))[::args.frame_step]
            mot_paths += frames
            mot_names += [f"{split}/{seq}/{p.name}" for p in frames]
    h_det = hashes([RAW / DET_TRAIN / "images" / f"{n}.jpg" for n in det])
    h_mot = hashes(mot_paths)
    log(f"hashed {len(h_det):,} DET images and {len(h_mot):,} MOT frames", 1)

    best = np.full(len(det), 64)
    arg = np.zeros(len(det), dtype=int)
    for i in range(0, len(det), 512):
        d = hamming(h_det[i:i + 512], h_mot)
        best[i:i + 512], arg[i:i + 512] = d.min(1), d.argmin(1)
    for k in range(0, 17, 4):
        log(f"DET images with nearest MOT frame at distance <= {k:>2}: {(best <= k).sum()}", 1)
    new = 0
    for i in np.where(best <= args.max_dist)[0]:
        if det[i] not in excluded:
            excluded[det[i]] = f"near_duplicate:{mot_names[arg[i]]}:d={best[i]}"
            new += 1
    log(f"near-duplicates not already excluded by rule 1: {new}", 1)
    for i in np.argsort(best)[:5]:
        log(f"closest: {det[i]} ~ {mot_names[arg[i]]} (d={best[i]})", 2)

    out = DATA_DIR / "splits" / DET_EXCLUDED
    out.write_text("".join(f"{n}\t{r}\n" for n, r in sorted(excluded.items())))
    log(f"Excluded {len(excluded)} of {len(det):,} DET images, kept {len(det) - len(excluded):,} "
        f"-> {out.relative_to(DATA_DIR.parent)}")


if __name__ == "__main__":
    main()
