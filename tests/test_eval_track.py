"""The parts of eval_track.py that are ours, not TrackEval's: ignored-region filtering and the class mean."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.eval_track import COUNTS, PERCENT, IgnoredMask, mean_row


def rows(*boxes):
    """frame, id, x, y, w, h"""
    return np.array([[f, i, x, y, w, h] for f, i, x, y, w, h in boxes], dtype=float)


def test_box_mostly_inside_an_ignored_region_is_dropped():
    mask = IgnoredMask(rows((1, -1, 0, 0, 100, 100)), width=200, height=200)
    keep = mask.keep(rows((1, 1, 10, 10, 20, 20),       # fully inside
                          (1, 2, 90, 0, 20, 10),        # half inside: kept, the rule is strictly more than half
                          (1, 3, 80, 0, 40, 10),        # half inside as well
                          (1, 4, 95, 0, 20, 10),        # a quarter inside
                          (1, 5, 150, 150, 20, 20)))    # outside
    assert keep.tolist() == [False, True, True, True, True]


def test_regions_only_apply_to_their_own_frame():
    mask = IgnoredMask(rows((2, -1, 0, 0, 100, 100)), width=200, height=200)
    assert mask.keep(rows((1, 1, 10, 10, 20, 20), (2, 1, 10, 10, 20, 20))).tolist() == [True, False]


def test_union_of_two_regions_counts_once():
    """Two overlapping regions that together cover 60% of the box drop it; each alone covers 40%."""
    mask = IgnoredMask(rows((1, -1, 0, 0, 40, 10), (1, -1, 20, 0, 40, 10)), width=200, height=200)
    assert mask.keep(rows((1, 1, 0, 0, 100, 10))).tolist() == [False]
    single = IgnoredMask(rows((1, -1, 0, 0, 40, 10)), width=200, height=200)
    assert single.keep(rows((1, 1, 0, 0, 100, 10))).tolist() == [True]


def test_no_regions_and_no_rows():
    assert IgnoredMask(np.zeros((0, 9)), 100, 100).keep(rows((1, 1, 0, 0, 5, 5))).tolist() == [True]
    assert IgnoredMask(rows((1, -1, 0, 0, 5, 5)), 100, 100).keep(np.zeros((0, 10))).shape == (0,)


def class_result(value, gt_dets=10):
    return {**{k: value for k in PERCENT}, **{k: 1 for k in COUNTS}, "GT_Dets": gt_dets}


def test_mean_averages_percentages_and_sums_counts():
    out = mean_row([class_result(10.0), class_result(30.0)])
    assert out["HOTA"] == pytest.approx(20.0) and out["IDSW"] == 2 and out["classes"] == 2


def test_class_without_ground_truth_is_left_out_of_the_mean():
    out = mean_row([class_result(10.0), class_result(0.0, gt_dets=0)])
    assert out["HOTA"] == pytest.approx(10.0) and out["classes"] == 1


def test_threshold_tags():
    from src.evaluation.select_thresh import tag_value
    assert tag_value("03") == 0.3 and tag_value("08") == 0.8 and tag_value("045") == 0.45
