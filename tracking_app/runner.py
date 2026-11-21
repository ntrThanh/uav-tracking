"""One active inference session, independent from UI and storage adapters."""
from __future__ import annotations

import json
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2

from tracking_app.engine.sources import FrameSource

from .config import Settings
from .evidence import draw_trajectory
from .pipeline import InferencePipeline, create_pipeline
from .storage import SessionStore


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


TIMING_STAGES = ("vehicle_detection", "tracking", "plate_detection", "ocr", "other")


class SessionRunner:
    def __init__(self, store: SessionStore,
                 pipeline_factory: Callable[[Settings], InferencePipeline] = create_pipeline) -> None:
        self.store = store
        self.pipeline_factory = pipeline_factory
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.source: FrameSource | None = None
        self.session_id: str | None = None
        self.status = "idle"
        self.error: str | None = None
        self.frame_count = 0
        self.plate_count = 0
        self.processing_fps = 0.0
        self.last_inference_ms = 0.0
        self.inference_total_s = 0.0
        self._frame_times: deque[float] = deque(maxlen=30)
        self._stage_stats = {name: {"last_ms": 0.0, "last_calls": 0, "total_ms": 0.0, "calls": 0}
                             for name in TIMING_STAGES}
        self.preview: bytes | None = None

    def snapshot(self) -> dict:
        with self.lock:
            return {"session_id": self.session_id, "status": self.status, "error": self.error,
                    "frames": self.frame_count, "plates": self.plate_count,
                    "processing_fps": self.processing_fps,
                    "last_inference_ms": self.last_inference_ms,
                    "avg_inference_ms": self.inference_total_s * 1000 / self.frame_count if self.frame_count else 0.0,
                    "inference_total_s": self.inference_total_s,
                    "stage_timings": {
                        name: {"last_ms": stats["last_ms"], "last_calls": stats["last_calls"],
                               "avg_ms": stats["total_ms"] / stats["calls"] if stats["calls"] else 0.0,
                               "calls": stats["calls"]}
                        for name, stats in self._stage_stats.items()}}

    def start(self, kind: str, uri: str, settings: Settings, label: str) -> str:
        if kind not in ("video", "rtsp", "usb"):
            raise ValueError("Source must be a video, RTSP stream, or USB camera")
        if kind == "rtsp" and not uri.lower().startswith("rtsp://"):
            raise ValueError("RTSP URL must start with rtsp://")
        if kind == "usb" and (not uri.isdigit() or int(uri) > 32):
            raise ValueError("USB camera index must be between 0 and 32")
        if kind == "video" and not Path(uri).is_file():
            raise ValueError("Video file not found")
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise RuntimeError("An AI session is already running; stop it before starting another")
            session_id = uuid.uuid4().hex[:16]
            self.store.start(session_id, kind, label, now(), json.dumps(asdict(settings)))
            self.stop_event.clear()
            self.session_id, self.status, self.error = session_id, "starting", None
            self.frame_count = self.plate_count = 0
            self.processing_fps = self.last_inference_ms = self.inference_total_s = 0.0
            self._frame_times.clear()
            self._stage_stats = {name: {"last_ms": 0.0, "last_calls": 0, "total_ms": 0.0, "calls": 0}
                                 for name in TIMING_STAGES}
            self.preview = None
            self.thread = threading.Thread(target=self._run, args=(session_id, kind, uri, settings), daemon=True)
            self.thread.start()
            return session_id

    def stop(self) -> None:
        with self.lock:
            self.stop_event.set()
            source = self.source
            if self.status in ("starting", "running"):
                self.status = "stopping"
        if source:
            source.close()

    def _run(self, session_id: str, kind: str, uri: str, settings: Settings) -> None:
        source = None
        try:
            pipeline = self.pipeline_factory(settings)
            trajectories: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=200))
            known_plates: dict[int, tuple[str, float]] = {}
            best_vehicle_scores: dict[int, float] = {}
            best_unread_plate_scores: dict[int, float] = {}
            plate_image_ids: set[int] = set()
            last_seen: dict[int, tuple] = {}
            source = FrameSource(uri, session_id, kind, stride=settings.stride, reconnect=kind == "rtsp")
            with self.lock:
                self.source = source
                self.status = "running"
            self.store.running(session_id)
            try:
                with pipeline, source:
                    for frame in source:
                        if self.stop_event.is_set():
                            break
                        inference_started = time.perf_counter()
                        tracking, reads = pipeline.process(frame)
                        inference_ms = (time.perf_counter() - inference_started) * 1000
                        stage_timings = getattr(pipeline, "last_timings", ())
                        tracks = tracking.tracks
                        self.store.add_points(session_id, frame.meta.frame_id, frame.meta.timestamp, tracks)
                        current_ids = {track.track_id for track in tracks}
                        for departed_id in last_seen.keys() - current_ids:
                            self._save_trajectory(session_id, departed_id, last_seen.pop(departed_id))
                        if tracks:
                            last_frame = frame.image.copy()
                            for track in tracks:
                                last_seen[track.track_id] = (last_frame, frame.meta.frame_id,
                                                              frame.meta.timestamp, track.confidence)
                        for track in tracks:
                            x1, _, x2, y2 = track.bbox
                            trajectories[track.track_id].append((int((x1+x2)/2), int(y2)))
                            vehicle_crop = self._crop(frame.image, track.bbox)
                            if vehicle_crop is not None:
                                quality = track.confidence * vehicle_crop.shape[0] * vehicle_crop.shape[1]
                                if (track.track_id not in best_vehicle_scores or
                                        quality > best_vehicle_scores[track.track_id] * 1.2):
                                    self.store.save_evidence(session_id, track.track_id, "vehicle", vehicle_crop,
                                                             frame.meta.frame_id, frame.meta.timestamp, track.confidence)
                                    best_vehicle_scores[track.track_id] = quality
                        for read in reads:
                            previous_confidence = known_plates.get(read.track_id, ("", 0))[1]
                            improved_read = bool(read.text) and read.confidence > previous_confidence
                            if improved_read:
                                known_plates[read.track_id] = (read.text, read.confidence)
                                self.store.add_plate(session_id, read.track_id, read.text,
                                                     read.confidence, frame.meta.frame_id)
                            plate_crop = self._crop(frame.image, read.bbox) if read.bbox else None
                            if plate_crop is not None:
                                unread_quality = read.detection_confidence * plate_crop.shape[0] * plate_crop.shape[1]
                                save_plate = (improved_read or (read.text and read.track_id not in plate_image_ids)
                                              or (not read.text and read.track_id not in known_plates and
                                                  (read.track_id not in best_unread_plate_scores or
                                                   unread_quality > best_unread_plate_scores[read.track_id] * 1.2)))
                                if save_plate:
                                    self.store.save_evidence(session_id, read.track_id, "plate", plate_crop,
                                                             frame.meta.frame_id, frame.meta.timestamp,
                                                             read.confidence if read.text else read.detection_confidence)
                                    plate_image_ids.add(read.track_id)
                                    if not read.text:
                                        best_unread_plate_scores[read.track_id] = unread_quality
                        preview = self._render(frame.image, tracks, trajectories, known_plates, settings.preview_width)
                        with self.lock:
                            completed_at = time.perf_counter()
                            self._frame_times.append(completed_at)
                            if len(self._frame_times) >= 2:
                                elapsed = self._frame_times[-1] - self._frame_times[0]
                                self.processing_fps = (len(self._frame_times) - 1) / elapsed if elapsed > 0 else 0.0
                            self.last_inference_ms = inference_ms
                            self.inference_total_s += inference_ms / 1000
                            for stats in self._stage_stats.values():
                                stats["last_ms"] = 0.0
                                stats["last_calls"] = 0
                            recorded_ms = 0.0
                            for name, duration_ms in stage_timings:
                                stats = self._stage_stats.setdefault(name, {"last_ms": 0.0, "last_calls": 0,
                                                                            "total_ms": 0.0, "calls": 0})
                                stats["last_ms"] += duration_ms
                                stats["last_calls"] += 1
                                stats["total_ms"] += duration_ms
                                stats["calls"] += 1
                                recorded_ms += duration_ms
                            other_ms = max(0.0, inference_ms - recorded_ms)
                            other = self._stage_stats["other"]
                            other["last_ms"] = other_ms
                            other["last_calls"] = 1
                            other["total_ms"] += other_ms
                            other["calls"] += 1
                            self.preview = preview
                            self.frame_count += 1
                            self.plate_count = len(known_plates)
            finally:
                try:
                    for track_id, snapshot in last_seen.items():
                        self._save_trajectory(session_id, track_id, snapshot)
                finally:
                    with self.lock:
                        self.source = None
            status = "stopped" if self.stop_event.is_set() else "completed"
            self.store.finish(session_id, status, now())
            with self.lock:
                self.status = status
        except Exception as exc:
            message = str(exc)
            self.store.finish(session_id, "failed", now(), message)
            with self.lock:
                self.error, self.status = message, "failed"
        finally:
            if source:
                source.close()

    @staticmethod
    def _crop(image, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        return image[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None

    def _save_trajectory(self, session_id: str, track_id: int, snapshot: tuple) -> None:
        image, frame_id, video_time, confidence = snapshot
        points = self.store.points(session_id, track_id)
        if points:
            overlaid = draw_trajectory(image, points, track_id)
            self.store.save_evidence(session_id, track_id, "trajectory", overlaid,
                                     frame_id, video_time, confidence)

    @staticmethod
    def _render(image, tracks, trajectories, plates, width: int) -> bytes:
        canvas = image.copy()
        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (34, 197, 94), 2)
            label = f"#{track.track_id} {plates.get(track.track_id, ('', 0))[0]}".strip()
            cv2.putText(canvas, label, (x1, max(20, y1-7)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (34, 197, 94), 2)
            path = list(trajectories[track.track_id])
            for a, b in zip(path, path[1:]):
                cv2.line(canvas, a, b, (255, 180, 60), 2)
        if canvas.shape[1] > width:
            height = round(canvas.shape[0] * width / canvas.shape[1])
            canvas = cv2.resize(canvas, (width, height))
        ok, buffer = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if not ok:
            raise RuntimeError("Could not encode the preview frame")
        return buffer.tobytes()
