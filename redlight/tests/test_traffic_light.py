"""Unit tests for TrafficLightClassifier."""
import numpy as np
import pytest

from src.common.config import LightROI
from src.detect.traffic_light import TrafficLightClassifier


def create_solid_bgr(color_bgr: tuple[int, int, int], size: tuple[int, int] = (60, 30)) -> np.ndarray:
    """Create a test image filled with a specific color."""
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color_bgr
    return img


def create_traffic_light_sim(active: str) -> np.ndarray:
    """Create a vertical 3-bulb traffic light (60x20) with black housing and one active bulb."""
    img = np.zeros((60, 20, 3), dtype=np.uint8)  # black housing
    # Top bulb (y: 5-15, x: 5-15) -> Red (BGR: 0, 0, 255)
    # Mid bulb (y: 25-35, x: 5-15) -> Yellow (BGR: 0, 255, 255)
    # Bot bulb (y: 45-55, x: 5-15) -> Green (BGR: 0, 255, 0)
    if active == "red":
        img[5:15, 5:15] = (0, 0, 255)
    elif active == "yellow":
        img[25:35, 5:15] = (0, 240, 255)
    elif active == "green":
        img[45:55, 5:15] = (0, 255, 0)
    return img


def test_traffic_light_red():
    clf = TrafficLightClassifier(history_len=1)
    crop = create_traffic_light_sim("red")
    res = clf.classify_crop(crop)
    assert res.state == "red"
    assert res.confidence > 0.0


def test_traffic_light_yellow():
    clf = TrafficLightClassifier(history_len=1)
    crop = create_traffic_light_sim("yellow")
    res = clf.classify_crop(crop)
    assert res.state == "yellow"
    assert res.confidence > 0.0


def test_traffic_light_green():
    clf = TrafficLightClassifier(history_len=1)
    crop = create_traffic_light_sim("green")
    res = clf.classify_crop(crop)
    assert res.state == "green"
    assert res.confidence > 0.0


def test_traffic_light_off():
    clf = TrafficLightClassifier(history_len=1)
    crop = create_traffic_light_sim("off")
    res = clf.classify_crop(crop)
    assert res.state == "unknown"


def test_traffic_light_from_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Place a red bulb at [10:70, 40:60]
    frame[10:70, 40:60] = create_traffic_light_sim("red")
    roi = LightROI(x=0.4, y=0.1, w=0.2, h=0.6)
    clf = TrafficLightClassifier(history_len=1)
    res = clf.classify_frame(frame, roi)
    assert res.state == "red"
