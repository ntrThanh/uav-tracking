"""Box overlap and optimal assignment."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

Matches = list[tuple[int, int]]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of (N, 4) and (M, 4) xyxy boxes."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def linear_assignment(cost: np.ndarray, thresh: float) -> tuple[Matches, list[int], list[int]]:
    """Minimum-cost matching that rejects pairs above thresh.

    Returns (matches, unmatched rows, unmatched columns). Costs above thresh are clipped just over it first, so one
    impossible pair cannot pull the rest of the solution towards it.
    """
    rows, cols = cost.shape
    if rows == 0 or cols == 0:
        return [], list(range(rows)), list(range(cols))
    r, c = linear_sum_assignment(np.minimum(cost, thresh + 1e-5))
    matches = [(int(i), int(j)) for i, j in zip(r, c) if cost[i, j] <= thresh]
    used_rows = {i for i, _ in matches}
    used_cols = {j for _, j in matches}
    return (matches, [i for i in range(rows) if i not in used_rows], [j for j in range(cols) if j not in used_cols])


def fuse_score(cost: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """ByteTrack: weight the IoU similarity by the detection confidence."""
    if cost.size == 0:
        return cost
    return 1.0 - (1.0 - cost) * scores[None, :]


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance of L2-normalised rows, clipped to [0, 2]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    return np.clip(1.0 - a @ b.T, 0.0, 2.0)
