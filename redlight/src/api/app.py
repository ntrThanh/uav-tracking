"""FastAPI backend.

Stage 2: camera-geometry editor (stop line, zones, light ROI) with reference-frame
extraction and JSON save/load. Stage 5 adds violation browsing.

Run from the project root:  python scripts/run_web.py --port 8000
"""
from __future__ import annotations

import base64
import logging
import re
import os
import secrets
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.common.config import PROJECT_ROOT, CameraConfig, LightROI, resolve_path
from src.detect.traffic_light import TrafficLightClassifier
from src.ingest.video import save_reference_frame
from src.operations.monitor import CameraProfile, CameraStore, EventStore, MonitorManager, open_source
from src.operations.engines import DETECTORS, TRACKERS

log = logging.getLogger(__name__)

CAMERA_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
RESERVED_IDS = {"pipeline"}  # config/pipeline.json is not a camera


def create_app(config_dir: Path | None = None, web_dir: Path | None = None) -> FastAPI:
    """Build the application; directories are injectable for tests."""
    config_dir = Path(config_dir or PROJECT_ROOT / "config")
    web_dir = Path(web_dir or PROJECT_ROOT / "web")
    config_dir.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="Red-Light Violation Detection", version="0.2.0")
    profiles = CameraStore(config_dir / "cameras")
    events = EventStore(config_dir.parent / "outputs" / "events" if config_dir == PROJECT_ROOT / "config"
                        else config_dir / "events")
    monitor = MonitorManager(profiles, config_dir, events)
    app.router.add_event_handler("shutdown", monitor.stop_all)
    admin_token = os.environ.get("REDLIGHT_ADMIN_TOKEN", "")

    class AdminAuthMiddleware:
        def __init__(self, inner):
            self.inner = inner

        async def __call__(self, scope, receive, send):
            if admin_token and scope["type"] == "http" and scope.get("path", "").startswith("/api/") \
                    and scope.get("path") != "/api/auth/login":
                from http.cookies import SimpleCookie
                from fastapi.responses import JSONResponse
                headers = {k.lower(): v for k, v in scope.get("headers", [])}
                cookie = SimpleCookie()
                cookie.load(headers.get(b"cookie", b"").decode("latin1"))
                session = cookie.get("redlight_session")
                bearer = headers.get(b"authorization", b"").decode("latin1").removeprefix("Bearer ")
                if not (secrets.compare_digest(session.value if session else "", admin_token)
                        or secrets.compare_digest(bearer, admin_token)):
                    await JSONResponse({"detail": "Cần đăng nhập quản trị"}, status_code=401)(scope, receive, send)
                    return
            await self.inner(scope, receive, send)

    app.add_middleware(AdminAuthMiddleware)

    @app.post("/api/auth/login")
    def login(body: dict[str, Any] = Body(...)):
        from fastapi.responses import JSONResponse
        if not admin_token:
            return {"authenticated": True, "mode": "local"}
        if not secrets.compare_digest(str(body.get("token", "")), admin_token):
            raise HTTPException(401, "Mã quản trị không đúng")
        response = JSONResponse({"authenticated": True})
        response.set_cookie("redlight_session", admin_token, httponly=True, samesite="strict",
                            secure=os.environ.get("REDLIGHT_HTTPS", "") == "1", max_age=86400)
        return response

    @app.post("/api/auth/logout")
    def logout():
        from fastapi.responses import JSONResponse
        response = JSONResponse({"authenticated": False})
        response.delete_cookie("redlight_session")
        return response

    @app.get("/api/engines")
    def engines() -> dict[str, Any]:
        return {"detectors": list(DETECTORS), "trackers": list(TRACKERS)}

    @app.post("/api/videos/upload")
    async def upload_video(file: UploadFile = File(...)) -> dict[str, str]:
        if Path(file.filename or "").suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv"}:
            raise HTTPException(422, "Chỉ nhận video mp4, mov, avi hoặc mkv")
        folder = PROJECT_ROOT / "data" / "uploads"
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        folder.chmod(0o700)
        import uuid
        path = folder / f"{uuid.uuid4().hex}{Path(file.filename).suffix.lower()}"
        size = 0
        try:
            with path.open("wb") as target:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > 500 * 1024 * 1024:
                        raise HTTPException(413, "Video vượt giới hạn 500 MB")
                    target.write(chunk)
            path.chmod(0o600)
            cap = cv2.VideoCapture(str(path))
            ok, _ = cap.read()
            cap.release()
            if not ok:
                raise HTTPException(422, "Không đọc được video")
            return {"path": str(path.relative_to(PROJECT_ROOT)), "name": file.filename or path.name}
        except Exception:
            path.unlink(missing_ok=True)
            raise

    @app.get("/api/managed-cameras")
    def managed_cameras() -> list[dict[str, Any]]:
        return [{**p.public(), **monitor.status(p.camera_id)} for p in profiles.list()]

    @app.get("/api/managed-cameras/{camera_id}")
    def managed_camera(camera_id: str) -> dict[str, Any]:
        try:
            profile = profiles.get(camera_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not profile:
            raise HTTPException(404, "Không tìm thấy camera")
        return {**profile.public(), **monitor.status(camera_id)}

    @app.put("/api/managed-cameras/{camera_id}")
    def save_managed_camera(camera_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            old = profiles.get(camera_id)
            data = dict(body)
            if old and not data.get("source"):
                data["source"] = old.source
            profile = CameraProfile.from_dict(data, camera_id)
            if monitor.status(camera_id)["status"] in {"running", "starting", "reconnecting"}:
                raise HTTPException(409, "Hãy dừng camera trước khi sửa nguồn hoặc engine")
            profiles.save(profile)
            return profile.public()
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.delete("/api/managed-cameras/{camera_id}")
    def delete_managed_camera(camera_id: str) -> dict[str, bool]:
        try:
            monitor.stop(camera_id)
            profiles.delete(camera_id)
            camera_path(camera_id).unlink(missing_ok=True)
            reference_path(camera_id).unlink(missing_ok=True)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"deleted": True}

    @app.post("/api/managed-cameras/{camera_id}/start")
    def start_camera(camera_id: str) -> dict[str, Any]:
        try:
            return monitor.start(camera_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/managed-cameras/{camera_id}/stop")
    def stop_camera(camera_id: str) -> dict[str, Any]:
        return monitor.stop(camera_id)

    @app.get("/api/managed-cameras/{camera_id}/status")
    def camera_status(camera_id: str) -> dict[str, Any]:
        return monitor.status(camera_id)

    @app.post("/api/managed-cameras/{camera_id}/reference")
    def capture_managed_reference(camera_id: str) -> dict[str, Any]:
        try:
            profile = profiles.get(camera_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if profile is None:
            raise HTTPException(404, "Không tìm thấy camera")
        cap = open_source(profile)
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok or frame is None:
            raise HTTPException(502, "Không lấy được ảnh từ nguồn camera")
        if not cv2.imwrite(str(reference_path(camera_id)), frame):
            raise HTTPException(500, "Không lưu được ảnh tham chiếu")
        return reference_info(camera_id)

    @app.get("/api/managed-cameras/{camera_id}/preview.mjpg")
    def camera_preview(camera_id: str):
        if monitor.status(camera_id)["status"] != "running":
            raise HTTPException(409, "Camera chưa chạy")
        return StreamingResponse(monitor.preview(camera_id), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store"})

    @app.get("/api/events")
    def list_events(camera_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        try:
            return events.list(camera_id, min(max(limit, 1), 200))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.patch("/api/events/{camera_id}/{event_id}/review")
    def review_event(camera_id: str, event_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            return events.review(camera_id, event_id, str(body.get("status", "")))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/events/{camera_id}/{event_id}/image.jpg")
    def event_image(camera_id: str, event_id: str) -> FileResponse:
        try:
            path = events.image_path(camera_id, event_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not path.exists():
            raise HTTPException(404, "Không tìm thấy ảnh")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/events/{camera_id}/{event_id}/region.jpg")
    def event_region(camera_id: str, event_id: str) -> FileResponse:
        try:
            path = events.region_path(camera_id, event_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not path.exists():
            raise HTTPException(404, "Không tìm thấy ảnh vùng")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/events/{camera_id}/{event_id}/plate.jpg")
    def event_plate(camera_id: str, event_id: str) -> FileResponse:
        try:
            path = events.plate_path(camera_id, event_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not path.exists():
            raise HTTPException(404, "Không tìm thấy ảnh biển số")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    def camera_path(camera_id: str) -> Path:
        if not CAMERA_ID_RE.match(camera_id) or camera_id in RESERVED_IDS:
            raise HTTPException(400, "camera_id chỉ gồm chữ, số, '_' hoặc '-' (tối đa 32 ký tự)")
        return config_dir / f"{camera_id}.json"

    def reference_path(camera_id: str) -> Path:
        return config_dir / f"{camera_id}_ref.jpg"

    def reference_info(camera_id: str) -> dict[str, Any]:
        path = reference_path(camera_id)
        info: dict[str, Any] = {"reference_frame": path.name, "url": f"/api/cameras/{camera_id}/reference.jpg"}
        img = cv2.imread(str(path))
        if img is not None:
            info["height"], info["width"] = img.shape[:2]
        return info

    # ------------------------------------------------------------------ cameras
    @app.get("/api/cameras")
    def list_cameras() -> list[dict[str, Any]]:
        items = []
        for path in sorted(config_dir.glob("*.json")):
            if path.stem in RESERVED_IDS:
                continue
            try:
                cam = CameraConfig.load(path)
            except (ValueError, TypeError, KeyError) as exc:
                log.warning("Skipping invalid camera config %s: %s", path.name, exc)
                continue
            items.append({
                "camera_id": path.stem,
                "has_reference": reference_path(path.stem).exists(),
                "has_stop_line": cam.stop_line is not None,
                "n_zones": len(cam.zones),
                "has_light_roi": cam.light_roi is not None,
            })
        return items

    @app.get("/api/cameras/{camera_id}")
    def get_camera(camera_id: str) -> dict[str, Any]:
        path = camera_path(camera_id)
        if not path.exists():
            raise HTTPException(404, f"Chưa có cấu hình cho camera '{camera_id}'")
        try:
            return CameraConfig.load(path).to_dict()
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(500, f"File cấu hình hỏng: {exc}") from exc

    @app.put("/api/cameras/{camera_id}")
    def put_camera(camera_id: str, data: dict[str, Any] = Body(...)) -> dict[str, Any]:
        path = camera_path(camera_id)
        data = dict(data)
        data["camera_id"] = camera_id
        if not data.get("reference_frame") and reference_path(camera_id).exists():
            data["reference_frame"] = reference_path(camera_id).name
        try:
            cam = CameraConfig.from_dict(data)
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(422, f"Cấu hình không hợp lệ: {exc}") from exc
        cam.save(path)
        log.info("Saved camera config %s", path)
        return cam.to_dict()

    # ------------------------------------------------------------------ reference frame
    @app.post("/api/cameras/{camera_id}/reference/from-video")
    def reference_from_video(camera_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        camera_path(camera_id)  # validates the id
        video = resolve_path(str(body.get("video_path") or ""))
        try:
            time_s = float(body.get("time_s") or 0.0)
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, "time_s phải là số") from exc
        if not video.is_file():
            raise HTTPException(404, f"Không tìm thấy video: {video}")
        try:
            save_reference_frame(video, reference_path(camera_id), time_s=time_s)
        except (FileNotFoundError, IndexError, OSError) as exc:
            raise HTTPException(400, f"Không lấy được frame: {exc}") from exc
        return reference_info(camera_id)

    @app.post("/api/cameras/{camera_id}/reference/upload")
    async def reference_upload(camera_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
        camera_path(camera_id)
        raw = await file.read()
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "File tải lên không phải ảnh hợp lệ")
        cv2.imwrite(str(reference_path(camera_id)), img)
        return reference_info(camera_id)

    @app.get("/api/cameras/{camera_id}/reference.jpg")
    def get_reference(camera_id: str) -> FileResponse:
        path = reference_path(camera_id)
        if not path.exists():
            raise HTTPException(404, "Chưa có frame tham chiếu")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    # ------------------------------------------------------------------ traffic light
    @app.post("/api/cameras/{camera_id}/detect-light")
    def detect_light(camera_id: str, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        path = camera_path(camera_id)
        cam = None
        if path.exists():
            try:
                cam = CameraConfig.load(path)
            except Exception:
                pass

        roi = cam.light_roi if cam else None
        if body and "roi" in body and body["roi"]:
            try:
                roi = LightROI(
                    x=float(body["roi"]["x"]),
                    y=float(body["roi"]["y"]),
                    w=float(body["roi"]["w"]),
                    h=float(body["roi"]["h"]),
                )
            except Exception as exc:
                raise HTTPException(422, f"ROI không hợp lệ: {exc}")

        if roi is None:
            raise HTTPException(400, "Camera chưa có cấu hình light_roi")

        ref_p = reference_path(camera_id)
        if not ref_p.exists():
            raise HTTPException(404, "Chưa có ảnh tham chiếu để nhận diện đèn")

        img = cv2.imread(str(ref_p))
        if img is None:
            raise HTTPException(500, "Không thể đọc ảnh tham chiếu")

        h, w = img.shape[:2]
        x0, y0, x1, y1 = roi.to_pixels(w, h)
        crop = img[y0:y1, x0:x1]

        clf = TrafficLightClassifier(history_len=1)
        res = clf.classify_crop(crop)

        crop_b64 = ""
        if crop is not None and crop.size > 0:
            ret, buf = cv2.imencode(".jpg", crop)
            if ret:
                crop_b64 = base64.b64encode(buf).decode("utf-8")

        return {
            "camera_id": camera_id,
            "state": res.state,
            "confidence": res.confidence,
            "dominant_color": res.dominant_color,
            "scores": res.scores,
            "lit_ratio": res.lit_ratio,
            "crop_base64": f"data:image/jpeg;base64,{crop_b64}" if crop_b64 else None,
            "crop_size": [int(x1 - x0), int(y1 - y0)],
        }

    # ------------------------------------------------------------------ web ui
    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(web_dir / "index.html")

    @app.get("/editor", include_in_schema=False)
    def editor() -> FileResponse:
        return FileResponse(web_dir / "editor.html")

    if web_dir.is_dir():
        app.mount("/static", StaticFiles(directory=web_dir), name="static")
    return app


app = create_app()
