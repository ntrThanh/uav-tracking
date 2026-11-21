"""Image, video file, USB and RTSP frame sources with bounded buffering."""
from __future__ import annotations

import time
import threading
from pathlib import Path
from typing import Iterator

import cv2

from .schema import Frame, FrameMetadata


class FrameSource:
    def __init__(self, uri: str, source_id: str, kind: str = "video", stride: int = 1,
                 buffer_size: int = 1, reconnect: bool = True) -> None:
        if kind not in {"image", "video", "usb", "rtsp"} or stride < 1 or buffer_size < 1:
            raise ValueError("invalid source kind, stride or buffer size")
        self.uri, self.source_id, self.kind = uri, source_id, kind
        self.stride, self.buffer_size, self.reconnect = stride, buffer_size, reconnect
        self.dropped = 0
        self._capture = None
        self._closed = False
        self._stop = threading.Event()

    def _open(self) -> None:
        value = int(self.uri) if self.kind == "usb" else self.uri
        if self.kind == "rtsp" and hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            try:
                cap = cv2.VideoCapture(value, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                                                               cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
            except (TypeError, cv2.error):
                cap = cv2.VideoCapture(value)
        else:
            cap = cv2.VideoCapture(value)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        if not cap.isOpened():
            cap.release()
            raise OSError(f"cannot open {self.kind} source")
        self._capture = cap

    def __iter__(self) -> Iterator[Frame]:
        if self._closed:
            raise RuntimeError("source is closed")
        if self.kind == "image":
            image = cv2.imread(self.uri)
            if image is None:
                raise FileNotFoundError(self.uri)
            h, w = image.shape[:2]
            yield Frame(image, FrameMetadata(self.source_id, 0, time.time(), w, h))
            return
        if self._capture is None:
            self._open()
        assert self._capture is not None
        cap = self._capture
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        if fps <= 0:
            fps = 25.0
        index, retries = 0, 0
        while not self._stop.is_set():
            ok, image = cap.read()
            if not ok or image is None:
                if self.kind != "rtsp" or not self.reconnect:
                    break
                cap.release()
                self._capture = None
                if self._stop.wait(min(2 ** retries, 30)):
                    break
                retries += 1
                try:
                    self._open()
                except OSError:
                    continue
                cap = self._capture
                assert cap is not None
                continue
            retries = 0
            if index % self.stride:
                self.dropped += 1
                index += 1
                continue
            h, w = image.shape[:2]
            timestamp = index / fps if self.kind == "video" else time.time()
            yield Frame(image, FrameMetadata(self.source_id, index, timestamp, w, h))
            index += 1

    def close(self) -> None:
        self._closed = True
        self._stop.set()
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
