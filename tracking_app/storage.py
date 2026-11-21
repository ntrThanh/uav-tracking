"""SQLite persistence and Excel export for sessions, tracks and plate reads."""
from __future__ import annotations

import sqlite3
from shutil import copy2
from pathlib import Path

import cv2
import numpy as np


class SessionStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, source_kind TEXT NOT NULL, source_label TEXT NOT NULL,
                    started_at TEXT NOT NULL, ended_at TEXT, status TEXT NOT NULL,
                    error TEXT, settings_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS points (
                    session_id TEXT NOT NULL, track_id INTEGER NOT NULL, frame_id INTEGER NOT NULL,
                    video_time REAL NOT NULL, x REAL NOT NULL, y REAL NOT NULL,
                    x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
                    class_name TEXT NOT NULL, confidence REAL NOT NULL,
                    PRIMARY KEY (session_id, track_id, frame_id)
                );
                CREATE TABLE IF NOT EXISTS plates (
                    session_id TEXT NOT NULL, track_id INTEGER NOT NULL, plate TEXT NOT NULL,
                    confidence REAL NOT NULL, first_frame INTEGER NOT NULL,
                    PRIMARY KEY (session_id, track_id)
                );
                CREATE TABLE IF NOT EXISTS evidence (
                    session_id TEXT NOT NULL, track_id INTEGER NOT NULL, kind TEXT NOT NULL,
                    frame_id INTEGER NOT NULL, video_time REAL NOT NULL, confidence REAL NOT NULL,
                    path TEXT NOT NULL,
                    PRIMARY KEY (session_id, track_id, kind)
                );
                CREATE INDEX IF NOT EXISTS idx_points_session ON points(session_id, frame_id);
            """)

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def start(self, session_id: str, kind: str, label: str, started: str, settings_json: str) -> None:
        with self._db() as db:
            db.execute("INSERT INTO sessions(id,source_kind,source_label,started_at,status,settings_json) VALUES(?,?,?,?,?,?)",
                       (session_id, kind, label, started, "starting", settings_json))

    def finish(self, session_id: str, status: str, ended: str, error: str | None = None) -> None:
        with self._db() as db:
            db.execute("UPDATE sessions SET status=?,ended_at=?,error=? WHERE id=?", (status, ended, error, session_id))

    def running(self, session_id: str) -> None:
        with self._db() as db:
            db.execute("UPDATE sessions SET status='running' WHERE id=?", (session_id,))

    def add_points(self, session_id: str, frame_id: int, timestamp: float, tracks: tuple) -> None:
        rows = []
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            rows.append((session_id, track.track_id, frame_id, timestamp, (x1+x2)/2, y2,
                         x1, y1, x2, y2, track.class_name, track.confidence))
        if rows:
            with self._db() as db:
                db.executemany("INSERT OR REPLACE INTO points VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", rows)

    def add_plate(self, session_id: str, track_id: int, plate: str, confidence: float, frame_id: int) -> None:
        with self._db() as db:
            db.execute("""INSERT INTO plates VALUES(?,?,?,?,?) ON CONFLICT(session_id,track_id) DO UPDATE SET
                plate=excluded.plate, confidence=excluded.confidence
                WHERE excluded.confidence > plates.confidence""",
                       (session_id, track_id, plate, confidence, frame_id))

    def set_plate(self, session_id: str, track_id: int, plate: str, confidence: float, frame_id: int) -> None:
        """Replace a plate when the user explicitly retries OCR on saved evidence."""
        with self._db() as db:
            db.execute("""INSERT INTO plates VALUES(?,?,?,?,?) ON CONFLICT(session_id,track_id) DO UPDATE SET
                plate=excluded.plate, confidence=excluded.confidence, first_frame=excluded.first_frame""",
                       (session_id, track_id, plate, confidence, frame_id))

    def save_evidence(self, session_id: str, track_id: int, kind: str, image: np.ndarray,
                      frame_id: int, video_time: float, confidence: float) -> Path:
        if kind not in ("vehicle", "plate", "trajectory") or track_id <= 0 or not session_id.isalnum():
            raise ValueError("Invalid evidence identifier")
        if image.size == 0:
            raise ValueError("Cannot save an empty evidence image")
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("Could not encode evidence image")
        relative = Path("evidence") / session_id / f"{track_id}_{kind}.jpg"
        target = self.path.parent / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(encoded.tobytes())
        temporary.replace(target)
        with self._db() as db:
            db.execute("""INSERT INTO evidence(session_id,track_id,kind,frame_id,video_time,confidence,path)
                VALUES(?,?,?,?,?,?,?) ON CONFLICT(session_id,track_id,kind) DO UPDATE SET
                frame_id=excluded.frame_id, video_time=excluded.video_time,
                confidence=excluded.confidence, path=excluded.path""",
                       (session_id, track_id, kind, frame_id, video_time, confidence, str(relative)))
        return target

    def evidence(self, session_id: str, track_id: int) -> dict[str, dict]:
        with self._db() as db:
            rows = db.execute("SELECT kind,frame_id,video_time,confidence,path FROM evidence "
                              "WHERE session_id=? AND track_id=?", (session_id, track_id)).fetchall()
        return {row["kind"]: {**dict(row), "path": self.path.parent / row["path"]} for row in rows}

    def sessions(self) -> list[dict]:
        with self._db() as db:
            rows = db.execute("SELECT * FROM sessions ORDER BY started_at DESC LIMIT 100").fetchall()
        return [dict(row) for row in rows]

    def tracks(self, session_id: str, plate_filter: str = "") -> list[dict]:
        query = """SELECT p.track_id, MIN(p.video_time) AS first_time, MAX(p.video_time) AS last_time,
            COUNT(*) AS points, MAX(p.class_name) AS class_name, l.plate, l.confidence AS plate_confidence
            FROM points p LEFT JOIN plates l ON l.session_id=p.session_id AND l.track_id=p.track_id
            WHERE p.session_id=? AND COALESCE(l.plate,'') LIKE ? GROUP BY p.track_id ORDER BY first_time"""
        with self._db() as db:
            rows = db.execute(query, (session_id, f"%{plate_filter}%")).fetchall()
        return [dict(row) for row in rows]

    def points(self, session_id: str, track_id: int) -> list[dict]:
        with self._db() as db:
            rows = db.execute("""SELECT frame_id,video_time,x,y,x1,y1,x2,y2,class_name,confidence
                FROM points WHERE session_id=? AND track_id=? ORDER BY frame_id""", (session_id, track_id)).fetchall()
        return [dict(row) for row in rows]

    def export_excel(self, session_id: str, target: Path) -> Path:
        from .excel import write_xlsx
        summary = [["Track ID", "License plate", "OCR confidence", "Vehicle type", "Start (s)",
                    "End (s)", "Points", "Trajectory image", "Vehicle image", "Plate image"]]
        tracks = self.tracks(session_id)
        evidence_target = target.with_name(f"{target.stem}_evidence")
        for item in tracks:
            images = self.evidence(session_id, item["track_id"])
            image_paths = []
            for kind in ("trajectory", "vehicle", "plate"):
                source = images.get(kind, {}).get("path")
                if source and source.is_file():
                    destination = evidence_target / source.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    copy2(source, destination)
                    image_paths.append(str(destination.relative_to(target.parent)))
                else:
                    image_paths.append(None)
            summary.append([item["track_id"], item["plate"], item["plate_confidence"], item["class_name"],
                            item["first_time"], item["last_time"], item["points"], *image_paths])
        detail = [["Track ID", "License plate", "Frame", "Time (s)", "X", "Y", "X1", "Y1", "X2", "Y2", "Vehicle type", "Confidence"]]
        for item in tracks:
            for point in self.points(session_id, item["track_id"]):
                detail.append([item["track_id"], item["plate"], point["frame_id"], point["video_time"],
                               point["x"], point["y"], point["x1"], point["y1"], point["x2"], point["y2"],
                               point["class_name"], point["confidence"]])
        return write_xlsx(target, [("Vehicles", summary), ("Trajectory", detail)])
