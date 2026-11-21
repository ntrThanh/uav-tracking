"""Bounded, online consensus for repeated OCR reads of one tracked plate."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class PlateSelection:
    text: str
    confidence: float
    frame_id: int


class PlateConsensus:
    """Keep a provisional first read; require repeated evidence to change it."""

    WINDOW = 12
    MIN_SWITCH_READS = 3
    SWITCH_MARGIN = 1.2

    def __init__(self) -> None:
        self._reads: deque[PlateSelection] = deque(maxlen=self.WINDOW)
        self.selection: PlateSelection | None = None

    def observe(self, text: str, confidence: float, frame_id: int) -> PlateSelection | None:
        if not text or not 0 < confidence <= 1:
            return self.selection
        observation = PlateSelection(text, confidence, frame_id)
        # Multiple plate regions in one vehicle frame must count as one vote.
        if self._reads and self._reads[-1].frame_id == frame_id:
            if confidence <= self._reads[-1].confidence:
                return self.selection
            self._reads.pop()
        self._reads.append(observation)

        scores: dict[str, float] = {}
        counts: dict[str, int] = {}
        best: dict[str, PlateSelection] = {}
        for read in self._reads:
            scores[read.text] = scores.get(read.text, 0.0) + read.confidence
            counts[read.text] = counts.get(read.text, 0) + 1
            if read.text not in best or read.confidence > best[read.text].confidence:
                best[read.text] = read

        current = self.selection
        if current is None:
            self.selection = observation
        elif current.frame_id == frame_id and confidence > current.confidence:
            self.selection = observation
        elif text == current.text:
            if confidence > current.confidence:
                self.selection = observation
        elif (counts[text] >= self.MIN_SWITCH_READS and
              scores[text] > scores.get(current.text, 0.0) * self.SWITCH_MARGIN):
            self.selection = best[text]
        return self.selection
