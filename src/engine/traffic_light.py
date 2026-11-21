"""Traffic light state classifier from cropped region of interest (ROI).

Analyzes the cropped traffic light image using HSV color segmentation,
intensity peak detection, and optional spatial positioning (vertical/horizontal bulb priors).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class TrafficLightResult:
    """Detection result for a traffic light ROI."""

    state: str  # "red" | "yellow" | "green" | "unknown"
    confidence: float  # 0.0 to 1.0
    scores: dict[str, float]  # raw scores for each state
    dominant_color: str
    lit_ratio: float  # ratio of bright pixels in crop


class TrafficLightClassifier:
    """Classifies traffic light state from a cropped image or frame + LightROI."""

    def __init__(
        self,
        history_len: int = 5,
        min_val: int = 120,
        min_sat: int = 60,
        min_pixel_ratio: float = 0.015,
    ) -> None:
        self.history_len = max(1, history_len)
        self.min_val = min_val
        self.min_sat = min_sat
        self.min_pixel_ratio = min_pixel_ratio
        self._history: deque[str] = deque(maxlen=self.history_len)

    def reset(self) -> None:
        """Clear temporal smoothing history."""
        self._history.clear()

    def classify_crop(self, crop: np.ndarray) -> TrafficLightResult:
        """Classify a BGR crop of the traffic light."""
        if crop is None or crop.size == 0:
            return TrafficLightResult(
                state="unknown",
                confidence=0.0,
                scores={"red": 0.0, "yellow": 0.0, "green": 0.0},
                dominant_color="unknown",
                lit_ratio=0.0,
            )

        h, w = crop.shape[:2]
        total_pixels = float(h * w)

        # Convert to HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # Masks for Red (two hue bands in OpenCV)
        mask_r1 = cv2.inRange(hsv, np.array([0, self.min_sat, self.min_val]), np.array([12, 255, 255]))
        mask_r2 = cv2.inRange(hsv, np.array([162, self.min_sat, self.min_val]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask_r1, mask_r2)

        # Mask for Yellow / Amber
        mask_yellow = cv2.inRange(hsv, np.array([14, self.min_sat, int(self.min_val * 1.1)]), np.array([36, 255, 255]))

        # Mask for Green
        mask_green = cv2.inRange(hsv, np.array([38, int(self.min_sat * 0.9), self.min_val]), np.array([92, 255, 255]))

        # Calculate raw counts and intensity weights
        count_red = float(np.count_nonzero(mask_red))
        count_yellow = float(np.count_nonzero(mask_yellow))
        count_green = float(np.count_nonzero(mask_green))

        # Spatial priors: check vertical or horizontal 3-bulb layout
        is_vertical = h >= int(w * 1.25)
        is_horizontal = w >= int(h * 1.25)

        spatial_weight_red = 1.0
        spatial_weight_yellow = 1.0
        spatial_weight_green = 1.0

        if is_vertical and h >= 6:
            third = h // 3
            # In vertical lights: Red is top, Yellow middle, Green bottom
            r_top = np.count_nonzero(mask_red[:third, :])
            y_mid = np.count_nonzero(mask_yellow[third : 2 * third, :])
            g_bot = np.count_nonzero(mask_green[2 * third :, :])

            if count_red > 0 and r_top / count_red > 0.45:
                spatial_weight_red = 1.3
            if count_yellow > 0 and y_mid / count_yellow > 0.45:
                spatial_weight_yellow = 1.3
            if count_green > 0 and g_bot / count_green > 0.45:
                spatial_weight_green = 1.3

        elif is_horizontal and w >= 6:
            third = w // 3
            # In horizontal lights: Red is left, Yellow middle, Green right
            r_left = np.count_nonzero(mask_red[:, :third])
            y_mid = np.count_nonzero(mask_yellow[:, third : 2 * third])
            g_right = np.count_nonzero(mask_green[:, 2 * third :])

            if count_red > 0 and r_left / count_red > 0.45:
                spatial_weight_red = 1.3
            if count_yellow > 0 and y_mid / count_yellow > 0.45:
                spatial_weight_yellow = 1.3
            if count_green > 0 and g_right / count_green > 0.45:
                spatial_weight_green = 1.3

        score_red = (count_red / total_pixels) * spatial_weight_red
        score_yellow = (count_yellow / total_pixels) * spatial_weight_yellow
        score_green = (count_green / total_pixels) * spatial_weight_green

        scores = {"red": float(score_red), "yellow": float(score_yellow), "green": float(score_green)}

        best_color = max(scores, key=lambda k: scores[k])
        best_score = scores[best_color]
        lit_ratio = float((count_red + count_yellow + count_green) / total_pixels)

        # Decision threshold
        if best_score < self.min_pixel_ratio or (count_red == 0 and count_yellow == 0 and count_green == 0):
            instant_state = "unknown"
            confidence = 0.0
        else:
            instant_state = best_color
            confidence = min(1.0, float(best_score / (self.min_pixel_ratio * 4.0)))

        # Temporal smoothing
        self._history.append(instant_state)
        smoothed_state = self._vote_state(self._history)

        return TrafficLightResult(
            state=smoothed_state,
            confidence=round(confidence, 3),
            scores={k: round(v, 4) for k, v in scores.items()},
            dominant_color=instant_state,
            lit_ratio=round(lit_ratio, 4),
        )

    @staticmethod
    def _vote_state(history: deque[str]) -> str:
        """Majority voting with preference for valid colors over unknown."""
        if not history:
            return "unknown"
        valid = [s for s in history if s != "unknown"]
        if not valid:
            return "unknown"
        # Count frequencies
        counts: dict[str, int] = {}
        for s in valid:
            counts[s] = counts.get(s, 0) + 1
        most_common = max(counts, key=lambda k: counts[k])
        # If the most common state represents >= 40% of the history window, use it
        if counts[most_common] >= max(1, len(history) // 3):
            return most_common
        return "unknown"
