import numpy as np

from src.common.geometry import denormalize, normalize, reference_point


def test_reference_point_is_bottom_centre():
    assert reference_point([100, 50, 200, 150]) == (150.0, 150.0)
    assert reference_point(np.array([0.0, 0.0, 10.0, 4.0], dtype=np.float32)) == (5.0, 4.0)


def test_normalize_roundtrip():
    p = (320.0, 180.0)
    assert denormalize(normalize(p, 640, 360), 640, 360) == p
