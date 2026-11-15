"""FastAPI backend.

Stage 2: camera-geometry editor (stop line, zones, light ROI) with reference-frame
extraction and JSON save/load. Stage 5 adds violation browsing.

Run from the project root:  python scripts/run_web.py --port 8000
"""
from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.common.config import PROJECT_ROOT, CameraConfig, LightROI, resolve_path
from src.detect.traffic_light import TrafficLightClassifier
from src.ingest.video import save_reference_frame

log = logging.getLogger(__name__)

CAMERA_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
RESERVED_IDS = {"pipeline"}  # config/pipeline.json is not a camera


def create_app(config_dir: Path | None = None, web_dir: Path | None = None) -> FastAPI:
    """Build the application; directories are injectable for tests."""
    config_dir = Path(config_dir or PROJECT_ROOT / "config")
    web_dir = Path(web_dir or PROJECT_ROOT / "web")
    config_dir.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="Red-Light Violation Detection", version="0.2.0")

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

    if web_dir.is_dir():
        app.mount("/static", StaticFiles(directory=web_dir), name="static")
    return app


app = create_app()
