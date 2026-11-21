"""Tkinter desktop interface for local inference sessions."""
from __future__ import annotations

import io
import tkinter as tk
from dataclasses import asdict, replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from tracking_app.config import DATA_DIR, Settings, load_settings, save_settings
from tracking_app.runner import TIMING_STAGES, SessionRunner
from tracking_app.storage import SessionStore
from tracking_app.plugins import load_plugins


class DesktopApp:
    def __init__(self, root: tk.Tk) -> None:
        load_plugins()
        self.root = root
        self.root.title("Vehicle Tracking")
        self.root.geometry("1420x880")
        self.root.minsize(1050, 660)
        self.root.configure(bg="#f2f5f3")
        self._style()
        self.store = SessionStore(DATA_DIR / "sessions.sqlite3")
        self.runner = SessionRunner(self.store)
        self.settings = load_settings()
        self.vars = {name: tk.StringVar(value=str(value)) for name, value in asdict(self.settings).items()}
        self.kind = tk.StringVar(value="video")
        self.uri = tk.StringVar()
        self.search = tk.StringVar()
        self.session = tk.StringVar()
        self.status = tk.StringVar(value="Ready")
        self.counts = tk.StringVar(value="0 frames | 0 plates")
        self.fps_value = tk.StringVar(value="0.0 FPS")
        self.inference_frame_value = tk.StringVar(value="0 ms")
        self.inference_total_value = tk.StringVar(value="0.0 s")
        self.stage_values = {name: {field: tk.StringVar(value="-" if field != "calls" else "0")
                                    for field in ("last", "average", "calls")}
                             for name in TIMING_STAGES}
        self._details_visible = False
        self.message = tk.StringVar(value="Select a source and start AI.")
        self._session_ids: list[str] = []
        self._selected_session_id: str | None = None
        self._last_preview: bytes | None = None
        self._photo = None
        self._selected_points: list[dict] = []
        self._evidence_images: dict[str, Image.Image] = {}
        self._evidence_photos: dict[str, ImageTk.PhotoImage] = {}
        self._evidence_tokens: dict[str, tuple] = {}
        self._evidence_key: tuple[str, int] | None = None
        self._trajectory_image: Image.Image | None = None
        self._trajectory_photo: ImageTk.PhotoImage | None = None
        self._rebuilding_table = False
        self._last_status = ""
        self._last_frame_count = -1
        self._ticks = 0
        self._build()
        self._refresh_sessions()
        self.root.after(300, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _style(self) -> None:
        from .layout import apply_style
        apply_style(self.root)

    def _build(self) -> None:
        from .layout import build
        build(self)

    def _toggle_inference_details(self) -> None:
        self._details_visible = not self._details_visible
        if self._details_visible:
            self.inference_details_panel.place(relx=1, x=-12, y=12, anchor="ne")
            self.inference_details_panel.lift()
            self.details_button.configure(text="Hide details")
        else:
            self.inference_details_panel.place_forget()
            self.details_button.configure(text="Show details")

    def _save_group(self, group: str) -> None:
        from .layout import GROUP_FIELDS, GROUP_LABELS
        try:
            values = asdict(self.settings)
            values.update({key: self.vars[key].get() for key in GROUP_FIELDS[group]})
            settings = Settings.from_dict(values)
            save_settings(settings)
            self.settings = settings
            for key in GROUP_FIELDS[group]:
                self.vars[key].set(str(getattr(settings, key)))
            self.message.set(f"{GROUP_LABELS[group]} settings saved.")
        except Exception as exc:
            messagebox.showerror("Could not save settings", str(exc), parent=self.root)

    def _reset_group(self, group: str) -> None:
        from .layout import GROUP_FIELDS, GROUP_LABELS
        defaults = Settings()
        for key in GROUP_FIELDS[group]:
            self.vars[key].set(str(getattr(defaults, key)))
        self.message.set(f"{GROUP_LABELS[group]} reset to defaults. Click Save to keep these values.")

    def _source_mode(self) -> None:
        self.uri.set("0" if self.kind.get() == "usb" else "")
        self.source_hint.configure(text={"video": "Select a video file on this computer", "rtsp": "Enter an rtsp:// URL", "usb": "Enter a local camera index"}[self.kind.get()])
        self.browse_button.configure(state="normal" if self.kind.get() == "video" else "disabled")

    def _browse(self) -> None:
        path = filedialog.askopenfilename(title="Select video", filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All files", "*.*")])
        if path:
            self.uri.set(path)

    def _start(self) -> None:
        try:
            settings = Settings.from_dict({key: var.get() for key, var in self.vars.items()})
            save_settings(settings)
            self.settings = settings
            uri = self.uri.get().strip()
            kind = self.kind.get()
            label = Path(uri).name if kind == "video" else (f"Camera USB {uri}" if kind == "usb" else "Camera RTSP")
            session_id = self.runner.start(kind, uri, settings, label)
            self._selected_session_id = session_id
            self.message.set("Starting AI...")
            self._refresh_sessions()
        except Exception as exc:
            messagebox.showerror("Could not start AI", str(exc), parent=self.root)

    def _stop(self) -> None:
        self.runner.stop()
        self.message.set("Stopping session...")

    def _refresh_sessions(self) -> None:
        rows = self.store.sessions()
        selected = self._selected_session_id
        self._session_ids = [row["id"] for row in rows]
        self.session_box.configure(values=[f"{row['started_at'][:19].replace('T', ' ')}  |  {row['source_label']}  |  {row['status']}" for row in rows])
        if selected in self._session_ids:
            self.session_box.current(self._session_ids.index(selected))
        elif rows:
            self.session_box.current(0)
            self._selected_session_id = rows[0]["id"]
        self._refresh_tracks()

    def _session_changed(self, _event=None) -> None:
        self._selected_session_id = self._current_session()
        self._selected_points = []
        self._clear_evidence()
        self._draw_trajectory()
        self._refresh_tracks()

    def _current_session(self) -> str | None:
        index = self.session_box.current()
        return self._session_ids[index] if 0 <= index < len(self._session_ids) else None

    def _refresh_tracks(self) -> None:
        session_id = self._current_session()
        if not session_id:
            return
        selected = self.table.selection()
        selected_id = selected[0] if selected else None
        if self._evidence_key and self._evidence_key[0] != session_id:
            selected_id = None
        rows = self.store.tracks(session_id, self.search.get().upper().strip())
        self._rebuilding_table = True
        try:
            self.table.delete(*self.table.get_children())
            for row in rows:
                track_id = str(row["track_id"])
                self.table.insert("", "end", iid=track_id,
                                  values=(f"#{track_id}", row["plate"] or "-", row["class_name"],
                                          f"{row['first_time']:.1f}-{row['last_time']:.1f} s", row["points"]))
            if selected_id and self.table.exists(selected_id):
                self.table.selection_set(selected_id)
        finally:
            self._rebuilding_table = False
        self._select_track()

    def _select_track(self) -> None:
        if self._rebuilding_table:
            return
        selected = self.table.selection()
        session_id = self._current_session()
        if selected and session_id:
            track_id = int(selected[0])
            self._selected_points = self.store.points(session_id, track_id)
            self._draw_trajectory()
            self._load_evidence(session_id, track_id)
        else:
            self._selected_points = []
            self._draw_trajectory()
            self._clear_evidence()

    def _clear_evidence(self) -> None:
        self._evidence_key = None
        self._evidence_tokens.clear()
        self._evidence_images.clear()
        self._evidence_photos.clear()
        self._trajectory_image = None
        self._trajectory_photo = None
        self.trajectory_info.set("")
        for kind, empty in (("vehicle", "Select a vehicle"), ("plate", "No plate image")):
            getattr(self, f"{kind}_evidence_label").configure(image="", text=empty)
            getattr(self, f"{kind}_evidence_info").set("")

    def _load_evidence(self, session_id: str, track_id: int) -> None:
        key = (session_id, track_id)
        if key != self._evidence_key:
            self._clear_evidence()
            self._evidence_key = key
        records = self.store.evidence(session_id, track_id)
        trajectory = records.get("trajectory")
        trajectory_token = (key, trajectory["frame_id"]) if trajectory else (key, None)
        if self._evidence_tokens.get("trajectory") != trajectory_token:
            self._evidence_tokens["trajectory"] = trajectory_token
            path = trajectory["path"] if trajectory else None
            self._trajectory_image = None
            if path and path.is_file():
                try:
                    with Image.open(path) as source:
                        self._trajectory_image = source.copy()
                    self.trajectory_info.set(
                        f"Last seen: frame {trajectory['frame_id']} | {trajectory['video_time']:.1f} s"
                        f" | {len(self._selected_points)} points")
                except OSError:
                    self.trajectory_info.set("")
            self._draw_trajectory()
        for kind in ("vehicle", "plate"):
            record = records.get(kind)
            token = (key, record["frame_id"]) if record else (key, None)
            if self._evidence_tokens.get(kind) == token:
                continue
            self._evidence_tokens[kind] = token
            path = record["path"] if record else None
            if path and path.is_file():
                try:
                    with Image.open(path) as source:
                        self._evidence_images[kind] = source.copy()
                    getattr(self, f"{kind}_evidence_info").set(
                        f"Frame {record['frame_id']} | {record['video_time']:.1f} s")
                    self._render_evidence(kind)
                    continue
                except OSError:
                    pass
            self._evidence_images.pop(kind, None)
            self._evidence_photos.pop(kind, None)
            getattr(self, f"{kind}_evidence_label").configure(
                image="", text="No image captured" if kind == "vehicle" else "No plate detected")
            getattr(self, f"{kind}_evidence_info").set("")

    def _render_evidence(self, kind: str) -> None:
        source = self._evidence_images.get(kind)
        if source is None:
            return
        label = getattr(self, f"{kind}_evidence_label")
        width, height = label.winfo_width() - 12, label.winfo_height() - 12
        if width < 20 or height < 20:
            return
        scale = min(width / source.width, height / source.height)
        image = source.resize((max(1, round(source.width * scale)),
                               max(1, round(source.height * scale))), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self._evidence_photos[kind] = photo
        label.configure(image=photo, text="")

    def _show_plate_preprocessing(self) -> None:
        selected = self.table.selection()
        session_id = self._current_session()
        if not selected or not session_id:
            self.message.set("Select a track with a plate image first.")
            return
        track_id = int(selected[0])
        evidence = self.store.evidence(session_id, track_id).get("plate")
        if not evidence or not evidence["path"].is_file():
            self.message.set("No plate image is available for this track.")
            return
        try:
            import cv2
            import numpy as np
            from tracking_app.ocr.preprocess import (PreprocessStage, StandardPlatePreprocessor,
                                                     create_preprocessor)
            from .plate_preprocessing import show_plate_preprocessing

            saved = self.store.session_settings(session_id) or {}
            ocr_fields = ("ocr_upscale", "ocr_preprocess_mode", "ocr_clahe_clip", "ocr_sharpen")
            settings = replace(Settings(), **{key: saved[key] for key in ocr_fields if key in saved})
            with Image.open(evidence["path"]) as source:
                crop = cv2.cvtColor(np.asarray(source.convert("RGB")), cv2.COLOR_RGB2BGR)
            processor = create_preprocessor(settings)
            if isinstance(processor, StandardPlatePreprocessor):
                stages = list(processor.stages(crop))
            else:
                stages = [PreprocessStage("Original crop", crop)]
                stages.extend(PreprocessStage(f"OCR input {index}", image, index)
                              for index, image in enumerate(processor.variants(crop), 1))
            show_plate_preprocessing(self.root, stages, track_id, evidence)
        except Exception as exc:
            messagebox.showerror("Could not show preprocessing", str(exc), parent=self.root)

    def _draw_trajectory(self) -> None:
        canvas = self.path_canvas
        canvas.delete("all")
        source = self._trajectory_image
        if source is None:
            message = ("No last-seen frame captured yet for this track."
                       if self._selected_points else "Select a vehicle to view its trajectory.")
            canvas.create_text(18, 20, anchor="nw", text=message, fill="#75857a",
                               width=max(canvas.winfo_width() - 36, 100))
            return
        width, height = max(canvas.winfo_width() - 8, 1), max(canvas.winfo_height() - 8, 1)
        scale = min(width / source.width, height / source.height)
        image = source.resize((max(1, round(source.width * scale)),
                               max(1, round(source.height * scale))), Image.Resampling.LANCZOS)
        self._trajectory_photo = ImageTk.PhotoImage(image)
        canvas.create_image(canvas.winfo_width() // 2, canvas.winfo_height() // 2,
                            image=self._trajectory_photo)

    def _export(self) -> None:
        session_id = self._current_session()
        if not session_id:
            messagebox.showinfo("Export Excel", "No session is available to export.", parent=self.root)
            return
        path = filedialog.asksaveasfilename(title="Save report", defaultextension=".xlsx",
                                            initialfile=f"tracking_{session_id}.xlsx", filetypes=[("Excel", "*.xlsx")])
        if path:
            try:
                self.store.export_excel(session_id, Path(path))
                self.message.set(f"Report saved: {path}")
            except Exception as exc:
                messagebox.showerror("Could not export Excel", str(exc), parent=self.root)

    def _retry_ocr(self) -> None:
        selected = self.table.selection()
        session_id = self._current_session()
        if not selected or not session_id:
            self.message.set("Select a track with a plate image first.")
            return
        state = self.runner.snapshot()
        if state["session_id"] == session_id and state["status"] in ("starting", "running", "stopping"):
            self.message.set("Stop this session before retrying OCR.")
            return
        track_id = int(selected[0])
        evidence = self.store.evidence(session_id, track_id).get("plate")
        if not evidence or not evidence["path"].is_file():
            self.message.set("No plate image is available for this track.")
            return
        try:
            import cv2
            from tracking_app.ocr import create_reader

            settings = Settings.from_dict({key: var.get() for key, var in self.vars.items()})
            crop = cv2.imread(str(evidence["path"]))
            if crop is None:
                raise ValueError("Could not open the saved plate image")
            result = create_reader(settings).read(crop)
            if not result:
                self.message.set(f"OCR could not read the plate for track #{track_id}.")
                return
            self.store.set_plate(session_id, track_id, result[0], result[1], evidence["frame_id"])
            self._refresh_tracks()
            self.message.set(f"Track #{track_id}: {result[0]} (OCR confidence {result[1]:.2f}).")
        except Exception as exc:
            messagebox.showerror("Could not retry OCR", str(exc), parent=self.root)

    def _poll(self) -> None:
        state = self.runner.snapshot()
        labels = {"idle": "Ready", "starting": "Starting", "running": "Running",
                  "stopping": "Stopping", "stopped": "Stopped", "completed": "Completed", "failed": "Error"}
        self.status.set(labels.get(state["status"], state["status"]))
        status_colors = {
            "failed": ("#FDE8E5", "#A33D31"),
            "starting": ("#FFF1D7", "#8B5A12"),
            "stopping": ("#FFF1D7", "#8B5A12"),
        }
        background, foreground = status_colors.get(state["status"], ("#E1F1E8", "#126B52"))
        self.status_label.configure(bg=background, fg=foreground)
        self.counts.set(f"{state['frames']} frames | {state['plates']} plates")
        self.fps_value.set(f"{state['processing_fps']:.1f} FPS")
        self.inference_frame_value.set(f"{state['avg_inference_ms']:.0f} ms")
        self.inference_total_value.set(f"{state['inference_total_s']:.1f} s")
        for name, values in self.stage_values.items():
            timing = state["stage_timings"][name]
            values["last"].set(f"{timing['last_ms']:.1f} ms" if timing["last_calls"] else "-")
            values["average"].set(f"{timing['avg_ms']:.1f} ms" if timing["calls"] else "-")
            values["calls"].set(str(timing["calls"]))
        active = state["status"] in ("starting", "running", "stopping")
        self.start_button.configure(state="disabled" if active else "normal")
        self.stop_button.configure(state="normal" if state["status"] in ("starting", "running") else "disabled")
        if state["error"]:
            self.message.set(state["error"])
        if state["frames"] != self._last_frame_count:
            self._last_frame_count = state["frames"]
            with self.runner.lock:
                preview = self.runner.preview
            if preview and preview is not self._last_preview:
                self._last_preview = preview
                image = Image.open(io.BytesIO(preview))
                image.thumbnail((max(self.preview_label.winfo_width()-12, 100), max(self.preview_label.winfo_height()-12, 100)))
                self._photo = ImageTk.PhotoImage(image)
                self.preview_label.configure(image=self._photo, text="")
        self._ticks += 1
        if state["status"] != self._last_status or self._ticks % 4 == 0:
            self._last_status = state["status"]
            self._refresh_sessions()
        self.root.after(500, self._poll)

    def _close(self) -> None:
        self.runner.stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    DesktopApp(root)
    root.mainloop()
