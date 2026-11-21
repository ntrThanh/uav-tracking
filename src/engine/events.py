"""Pluggable local event sinks."""
from __future__ import annotations

import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Protocol

import cv2

from .schema import Event, Frame, TrackingResult


class EventSink(Protocol):
    def write(self, event: Event, frame: Frame) -> None: ...
    def close(self) -> None: ...


class JsonlSink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: Event, frame: Frame) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

    def close(self) -> None:
        pass


class CallbackSink:
    def __init__(self, callback: Callable[[Event, Frame], None]) -> None:
        self.callback = callback

    def write(self, event: Event, frame: Frame) -> None:
        self.callback(event, frame)

    def close(self) -> None:
        pass


class EvidenceSink:
    """Store an original frame per event; paths derive only from generated event IDs."""
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, event: Event, frame: Frame) -> None:
        path = self.root / f"{event.event_id}.jpg"
        if not cv2.imwrite(str(path), frame.image):
            raise OSError(f"could not save evidence {path}")

    def close(self) -> None:
        pass


class TrackJsonlSink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, result: TrackingResult) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as out:
            for track in result.tracks:
                out.write(json.dumps({"source_id": result.frame.source_id,
                                      "frame_id": result.frame.frame_id,
                                      "timestamp": result.frame.timestamp,
                                      "track": asdict(track)}, ensure_ascii=False) + "\n")

    def close(self) -> None:
        pass
