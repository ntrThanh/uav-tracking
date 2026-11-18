"""Local camera manager: one capture/analytics worker per camera, shared MJPEG preview."""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit, urlunsplit

import cv2

from src.common.config import CameraConfig, PipelineConfig, PROJECT_ROOT
from src.detect.traffic_light import TrafficLightClassifier
from src.operations.engines import DETECTORS, TRACKERS, create_detector, create_tracker
from src.operations.rules import RedLightRule
from src.operations.plate import PlateRecognizer, PlateResult

log = logging.getLogger(__name__)
CAMERA_ID = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
ALLOWED_VIDEO_ROOT = (PROJECT_ROOT / "data").resolve()


def public_source(source: str) -> str:
    """Never return credentials in list/status responses."""
    parts = urlsplit(source)
    if parts.scheme in {"rtsp", "rtsps"}:
        host = parts.hostname or ""
        if parts.port:
            host += f":{parts.port}"
        return urlunsplit((parts.scheme, host, parts.path, "", ""))
    return Path(source).name


def open_source(profile: "CameraProfile") -> cv2.VideoCapture:
    """Open a source with bounded FFmpeg waits where OpenCV supports them."""
    if profile.source_type == "rtsp" and hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        try:
            return cv2.VideoCapture(profile.source, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
            ])
        except (TypeError, cv2.error):
            pass
    return cv2.VideoCapture(profile.source)


@dataclass(frozen=True)
class CameraProfile:
    camera_id: str
    name: str
    source_type: str
    source: str
    detector: str = "yolo11n"
    tracker: str = "bytetrack"
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any], camera_id: str) -> "CameraProfile":
        if not CAMERA_ID.fullmatch(camera_id) or camera_id == "pipeline":
            raise ValueError("Mã camera chỉ gồm chữ, số, _ hoặc - (tối đa 32 ký tự)")
        source_type = str(data.get("source_type", ""))
        source = str(data.get("source", "")).strip()
        if source_type == "rtsp":
            u = urlsplit(source)
            if u.scheme not in {"rtsp", "rtsps"} or not u.hostname:
                raise ValueError("Nguồn camera phải là URL rtsp:// hoặc rtsps:// hợp lệ")
        elif source_type == "video":
            path = Path(source)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            path = path.resolve()
            if not path.is_relative_to(ALLOWED_VIDEO_ROOT) or not path.is_file():
                raise ValueError("Video phải là tệp có sẵn trong redlight/data/")
            source = str(path)
        else:
            raise ValueError("Loại nguồn phải là rtsp hoặc video")
        detector = str(data.get("detector", "yolo11n"))
        tracker = str(data.get("tracker", "bytetrack"))
        if detector not in DETECTORS or tracker not in TRACKERS:
            raise ValueError("Engine chưa được đăng ký")
        name = str(data.get("name", camera_id)).strip()
        if not 1 <= len(name) <= 80:
            raise ValueError("Tên camera phải từ 1 đến 80 ký tự")
        return cls(camera_id, name, source_type, source, detector, tracker, bool(data.get("enabled", True)))

    def public(self) -> dict[str, Any]:
        return {**asdict(self), "source": public_source(self.source), "has_credentials": "@" in urlsplit(self.source).netloc}


class CameraStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        self._lock = threading.Lock()

    def path(self, camera_id: str) -> Path:
        if not CAMERA_ID.fullmatch(camera_id):
            raise ValueError("Mã camera không hợp lệ")
        return self.root / f"{camera_id}.json"

    def list(self) -> list[CameraProfile]:
        profiles = []
        for path in sorted(self.root.glob("*.json")):
            try:
                profiles.append(CameraProfile.from_dict(json.loads(path.read_text(encoding="utf-8")), path.stem))
            except (ValueError, OSError) as exc:
                log.warning("Skipping invalid camera profile %s: %s", path, exc)
        return profiles

    def get(self, camera_id: str) -> CameraProfile | None:
        path = self.path(camera_id)
        if not path.exists():
            return None
        return CameraProfile.from_dict(json.loads(path.read_text(encoding="utf-8")), camera_id)

    def save(self, profile: CameraProfile) -> None:
        path = self.path(profile.camera_id)
        with self._lock:
            temp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
            temp.write_text(json.dumps(asdict(profile), ensure_ascii=False, indent=2), encoding="utf-8")
            temp.chmod(0o600)
            temp.replace(path)

    def delete(self, camera_id: str) -> None:
        with self._lock:
            self.path(camera_id).unlink(missing_ok=True)


class EventStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        self._lock = threading.Lock()

    def record(self, camera_id: str, frame: Any, track: Any, light_state: str,
               geometry: CameraConfig | None = None, plate: PlateResult | None = None) -> dict[str, Any]:
        event_id = uuid.uuid4().hex
        folder = self.root / camera_id
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        folder.chmod(0o700)
        image = folder / f"{event_id}.jpg"
        x1, y1, x2, y2 = [int(v) for v in track.xyxy]
        annotated = frame.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (32, 65, 235), 3)
        cv2.putText(annotated, f"RED LIGHT  ID {track.track_id}", (max(0, x1), max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (32, 65, 235), 2)
        if not cv2.imwrite(str(image), annotated):
            raise OSError("Không lưu được ảnh bằng chứng")
        image.chmod(0o600)
        height, width = frame.shape[:2]
        if geometry and geometry.zones:
            points = [point for zone in geometry.zones for point in zone.polygon]
            rx0 = max(0, int(min(p[0] for p in points) * width))
            ry0 = max(0, int(min(p[1] for p in points) * height))
            rx1 = min(width, int(max(p[0] for p in points) * width))
            ry1 = min(height, int(max(p[1] for p in points) * height))
        else:
            pad = max(40, int(max(x2 - x1, y2 - y1) * 0.7))
            rx0, ry0, rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad), min(width, x2 + pad), min(height, y2 + pad)
        region = folder / f"{event_id}_region.jpg"
        if rx1 > rx0 and ry1 > ry0:
            cv2.imwrite(str(region), annotated[ry0:ry1, rx0:rx1])
            if region.exists():
                region.chmod(0o600)
        plate_image = folder / f"{event_id}_plate.jpg"
        if plate is not None and plate.crop is not None:
            cv2.imwrite(str(plate_image), plate.crop)
            if plate_image.exists():
                plate_image.chmod(0o600)
        event = {
            "event_id": event_id, "camera_id": camera_id, "type": "red_light",
            "occurred_at": datetime.now(timezone.utc).isoformat(), "track_id": track.track_id,
            "vehicle_class": track.class_name, "light_state": light_state,
            "review_status": "pending", "reviewed_at": None,
            "bbox": [x1, y1, x2, y2], "plate_text": plate.text if plate else None,
            "plate_status": plate.status if plate else "not_processed", "image": f"{camera_id}/{event_id}.jpg",
            "plate_confidence": plate.confidence if plate else None,
            "plate_image": f"{camera_id}/{event_id}_plate.jpg" if plate_image.exists() else None,
            "region_image": f"{camera_id}/{event_id}_region.jpg" if region.exists() else None,
        }
        with self._lock:
            temp = folder / f"{event_id}.tmp"
            temp.write_text(json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.chmod(0o600)
            temp.replace(folder / f"{event_id}.json")
        return event

    def list(self, camera_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if camera_id is not None and not CAMERA_ID.fullmatch(camera_id):
            raise ValueError("Mã camera không hợp lệ")
        paths = (self.root / camera_id).glob("*.json") if camera_id else self.root.glob("*/*.json")
        paths = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
        return [json.loads(path.read_text(encoding="utf-8")) for path in paths[:limit]]

    def image_path(self, camera_id: str, event_id: str) -> Path:
        if not CAMERA_ID.fullmatch(camera_id) or not re.fullmatch(r"[a-f0-9]{32}", event_id):
            raise ValueError("Mã sự kiện không hợp lệ")
        return self.root / camera_id / f"{event_id}.jpg"

    def region_path(self, camera_id: str, event_id: str) -> Path:
        return self.image_path(camera_id, event_id).with_name(f"{event_id}_region.jpg")

    def plate_path(self, camera_id: str, event_id: str) -> Path:
        return self.image_path(camera_id, event_id).with_name(f"{event_id}_plate.jpg")

    def review(self, camera_id: str, event_id: str, status: str) -> dict[str, Any]:
        if status not in {"confirmed", "dismissed"}:
            raise ValueError("Trạng thái phải là confirmed hoặc dismissed")
        path = self.image_path(camera_id, event_id).with_suffix(".json")
        with self._lock:
            if not path.is_file():
                raise FileNotFoundError("Không tìm thấy sự kiện")
            event = json.loads(path.read_text(encoding="utf-8"))
            event["review_status"] = status
            event["reviewed_at"] = datetime.now(timezone.utc).isoformat()
            temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
            temporary.write_text(json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.chmod(0o600)
            temporary.replace(path)
        return event


class CameraWorker:
    def __init__(self, profile: CameraProfile, geometry_path: Path, pipeline: PipelineConfig, events: EventStore):
        self.profile, self.geometry_path, self.pipeline, self.events = profile, geometry_path, pipeline, events
        self.status = "starting"
        self.error: str | None = None
        self.frames = 0
        self.started_at = time.time()
        self.latest_jpeg: bytes | None = None
        self.latest_event: dict[str, Any] | None = None
        self._plate_engine: PlateRecognizer | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"camera-{profile.camera_id}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        if not self._thread.is_alive():
            self.status = "stopped"

    def snapshot(self) -> bytes | None:
        with self._lock:
            return self.latest_jpeg

    def info(self) -> dict[str, Any]:
        return {"status": self.status, "error": self.error, "frames": self.frames,
                "fps": round(self.frames / max(time.time() - self.started_at, 0.01), 1),
                "latest_event": self.latest_event}

    def _run(self) -> None:
        cap = None
        try:
            if not self.geometry_path.exists():
                raise ValueError("Hãy cấu hình vùng, vạch dừng và đèn trước khi chạy")
            geometry = CameraConfig.load(self.geometry_path)
            if not geometry.stop_line or not geometry.light_roi:
                raise ValueError("Camera cần có vạch dừng và vùng đèn")
            rule = RedLightRule(geometry)
            detector = create_detector(self.profile.detector, self.pipeline.detector)
            tracker = create_tracker(self.profile.tracker, self.pipeline.tracker)
            light = TrafficLightClassifier()
            def connect() -> Any:
                delay = 1.0
                while not self._stop.is_set():
                    source = open_source(self.profile)
                    if source.isOpened():
                        self.error = None
                        self.status = "running"
                        return source
                    source.release()
                    if self.profile.source_type == "video":
                        raise OSError("Không đọc được video")
                    self.status = "reconnecting"
                    self.error = "Mất kết nối; đang thử lại"
                    self._stop.wait(delay)
                    delay = min(delay * 2, 30.0)
                raise InterruptedError("Đã dừng camera")

            cap = connect()
            self.status = "running"
            source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            if source_fps <= 0 or source_fps > 120:
                source_fps = 25.0
            frame_interval = 1.0 / source_fps
            last_frame_at = 0.0
            while not self._stop.is_set():
                if self.profile.source_type == "video" and last_frame_at:
                    self._stop.wait(max(0.0, frame_interval - (time.monotonic() - last_frame_at)))
                last_frame_at = time.monotonic()
                ok, frame = cap.read()
                if not ok:
                    if self.profile.source_type == "video":
                        if self.frames == 0:
                            raise OSError("Video không có frame")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        tracker = create_tracker(self.profile.tracker, self.pipeline.tracker)
                        rule = RedLightRule(geometry)
                        continue
                    cap.release()
                    cap = connect()
                    tracker = create_tracker(self.profile.tracker, self.pipeline.tracker)
                    rule = RedLightRule(geometry)
                    light.reset()
                    continue
                self.frames += 1
                h, w = frame.shape[:2]
                x0, y0, x1, y1 = geometry.light_roi.to_pixels(w, h)
                light_state = light.classify_crop(frame[y0:y1, x0:x1]).state
                detections = detector.detect(frame)
                tracks = tracker.update(detections, self.frames, time.time())
                preview = frame.copy()
                for track in tracks:
                    a, b, c, d = [int(v) for v in track.xyxy]
                    cv2.rectangle(preview, (a, b), (c, d), (32, 186, 173), 2)
                    cv2.putText(preview, f"{track.class_name} #{track.track_id}", (a, max(20, b - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    p = (track.ref_point[0] / w, track.ref_point[1] / h)
                    if rule.update(track.track_id, p, light_state):
                        try:
                            if self._plate_engine is None:
                                self._plate_engine = PlateRecognizer()
                            plate = self._plate_engine.recognize(frame, (a, b, c, d))
                        except Exception:
                            log.exception("Plate recognition failed for camera %s", self.profile.camera_id)
                            plate = PlateResult("engine_error")
                        self.latest_event = self.events.record(self.profile.camera_id, frame, track, light_state,
                                                               geometry, plate)
                p0, p1 = geometry.stop_line.to_pixels(w, h)
                cv2.line(preview, tuple(map(int, p0)), tuple(map(int, p1)), (32, 65, 235), 3)
                cv2.putText(preview, f"LIGHT: {light_state.upper()}", (16, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                target_w = min(w, 960)
                if target_w < w:
                    preview = cv2.resize(preview, (target_w, int(h * target_w / w)))
                good, jpeg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 78])
                if good:
                    with self._lock:
                        self.latest_jpeg = jpeg.tobytes()
        except Exception as exc:
            self.error = str(exc)
            self.status = "error"
            log.exception("Camera %s failed", self.profile.camera_id)
        finally:
            if cap is not None:
                cap.release()
            if self._stop.is_set():
                self.status = "stopped"


class MonitorManager:
    def __init__(self, profiles: CameraStore, geometry_root: Path, events: EventStore,
                 pipeline_path: Path = PROJECT_ROOT / "config/pipeline.json"):
        self.profiles, self.geometry_root, self.events, self.pipeline_path = profiles, geometry_root, events, pipeline_path
        self._workers: dict[str, CameraWorker] = {}
        self._lock = threading.Lock()

    def start(self, camera_id: str) -> dict[str, Any]:
        profile = self.profiles.get(camera_id)
        if not profile:
            raise ValueError("Camera không tồn tại")
        if not profile.enabled:
            raise ValueError("Camera đang tắt")
        with self._lock:
            old = self._workers.get(camera_id)
            if old and old.status in {"starting", "running", "reconnecting"}:
                return old.info()
            worker = CameraWorker(profile, self.geometry_root / f"{camera_id}.json",
                                  PipelineConfig.load(self.pipeline_path), self.events)
            self._workers[camera_id] = worker
            worker.start()
            return worker.info()

    def stop(self, camera_id: str) -> dict[str, Any]:
        worker = self._workers.get(camera_id)
        if worker:
            worker.stop()
            return worker.info()
        return {"status": "stopped", "error": None, "frames": 0, "fps": 0, "latest_event": None}

    def stop_all(self) -> None:
        for camera_id in list(self._workers):
            self.stop(camera_id)

    def status(self, camera_id: str) -> dict[str, Any]:
        worker = self._workers.get(camera_id)
        return worker.info() if worker else {"status": "stopped", "error": None, "frames": 0,
                                              "fps": 0, "latest_event": None}

    def preview(self, camera_id: str) -> Iterator[bytes]:
        worker = self._workers.get(camera_id)
        if worker is None or worker.status != "running":
            return
        last: bytes | None = None
        while worker.status == "running":
            image = worker.snapshot()
            if image and image is not last:
                last = image
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + image + b"\r\n"
            time.sleep(0.12)
