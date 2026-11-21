"""Desktop layout and visual styles, separate from session behavior."""
from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

from tracking_app.engine.models import ModelRegistry
from tracking_app.engine.trackers import available_trackers

from tracking_app.config import CATALOG
from tracking_app.ocr import available_backends as ocr_backends
from tracking_app.ocr.preprocess import available_preprocessors
from tracking_app.plate import available_backends as plate_backends
from tracking_app.tracking import available_backends as tracker_backends
from tracking_app.vehicle import available_backends as vehicle_backends

BG = "#F4F6F5"
WHITE = "#FFFFFF"
INK = "#172622"
MUTED = "#6C7D76"
LINE = "#E3EAE6"
ACCENT = "#167C61"
SIDEBAR = "#15241F"

DETECTION_FIELDS = (
    ("vehicle_backend", "Vehicle detector backend"),
    ("vehicle_model", "Vehicle detection model"),
    ("vehicle_classes", "Vehicle classes"),
    ("confidence", "Vehicle confidence threshold"),
    ("iou", "Vehicle IoU threshold"),
    ("plate_backend", "Plate detector backend"),
    ("plate_model", "Plate detection model"),
    ("plate_confidence", "Plate confidence threshold"),
    ("plate_iou", "Plate IoU threshold"),
    ("plate_interval", "Detect plates every N frames"),
    ("device", "Compute device"),
)
TRACKING_FIELDS = (
    ("tracker_backend", "Tracking backend"),
    ("tracker", "Tracking algorithm"),
    ("track_threshold", "Tracker threshold"),
    ("frame_rate", "Tracker frame rate"),
    ("stride", "Process every Nth frame"),
)
OCR_FIELDS = (
    ("ocr_backend", "Backend OCR"),
    ("ocr_detection_model_name", "Text detection model"),
    ("ocr_model_name", "Text recognition model"),
    ("ocr_detection_dir", "Text detection ONNX directory"),
    ("ocr_model_dir", "Text recognition ONNX directory"),
    ("ocr_confidence", "OCR confidence threshold"),
)
OCR_PREPROCESS_FIELDS = (
    ("ocr_upscale", "Plate image resize factor (1-4)"),
    ("ocr_preprocess_mode", "Contrast mode"),
    ("ocr_clahe_clip", "CLAHE contrast limit"),
    ("ocr_sharpen", "Sharpen strength (0-2)"),
)
OCR_DETECTION_FIELDS = (
    ("ocr_text_threshold", "Text pixel threshold"),
    ("ocr_box_threshold", "Text box score threshold"),
    ("ocr_unclip_ratio", "Text box expansion ratio"),
)
DISPLAY_FIELDS = (("preview_width", "Preview width"),)

SETTINGS_GROUPS = (
    ("detection", "Detection", DETECTION_FIELDS, 0),
    ("ocr", "License plate OCR", OCR_FIELDS, 0),
    ("preprocess", "Plate image preprocessing", OCR_PREPROCESS_FIELDS, 0),
    ("tracking", "Tracking", TRACKING_FIELDS, 1),
    ("text_detection", "Direct ONNX text detection", OCR_DETECTION_FIELDS, 1),
    ("display", "Display", DISPLAY_FIELDS, 1),
)
GROUP_FIELDS = {group: tuple(key for key, _ in fields) for group, _, fields, _ in SETTINGS_GROUPS}
GROUP_LABELS = {group: title for group, title, _, _ in SETTINGS_GROUPS}


def apply_style(root: tk.Tk) -> None:
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
        tkfont.nametofont(name).configure(family="Helvetica", size=10)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TFrame", background=WHITE)
    style.configure("Card.TFrame", background=WHITE)
    style.configure("TLabel", background=WHITE, foreground=INK, font=("Helvetica", 10))
    style.configure("Muted.TLabel", background=WHITE, foreground=MUTED, font=("Helvetica", 9))
    style.configure("Section.TLabel", background=WHITE, foreground=INK, font=("Helvetica", 12, "bold"))
    style.configure("TEntry", padding=(10, 9), bordercolor=LINE, lightcolor=LINE, darkcolor=LINE)
    style.configure("TCombobox", padding=(9, 8), fieldbackground=WHITE, background=WHITE,
                    bordercolor=LINE, lightcolor=LINE, darkcolor=LINE)
    style.map("TCombobox", fieldbackground=[("readonly", WHITE)])
    style.configure("TButton", padding=(14, 10), background=WHITE, foreground=INK,
                    bordercolor=LINE, lightcolor=LINE, darkcolor=LINE, font=("Helvetica", 10))
    style.map("TButton", background=[("active", "#F1F5F2")])
    style.configure("Primary.TButton", background=ACCENT, foreground=WHITE,
                    bordercolor=ACCENT, lightcolor=ACCENT, darkcolor=ACCENT)
    style.map("Primary.TButton", background=[("active", "#10654F"), ("disabled", "#B5CEC4")],
              foreground=[("disabled", WHITE)])
    style.configure("Treeview", background=WHITE, fieldbackground=WHITE, foreground=INK,
                    rowheight=36, borderwidth=0, font=("Helvetica", 9))
    style.configure("Treeview.Heading", background="#F1F5F2", foreground=MUTED,
                    borderwidth=0, padding=(8, 12), font=("Helvetica", 9, "bold"))
    style.map("Treeview", background=[("selected", "#DDF0E7")], foreground=[("selected", INK)])


def _card(parent, padding=22) -> ttk.Frame:
    frame = ttk.Frame(parent, style="Card.TFrame", padding=padding)
    return frame


def _text(parent, value, *, size=10, weight="normal", color=INK, bg=WHITE) -> tk.Label:
    return tk.Label(parent, text=value, bg=bg, fg=color, font=("Helvetica", size, weight), anchor="w")


def build(app) -> None:
    root = app.root
    root.configure(bg=BG)
    root.geometry("1440x900")
    root.minsize(1050, 700)

    shell = tk.Frame(root, bg=BG)
    shell.pack(fill="both", expand=True)
    shell.columnconfigure(1, weight=1)
    shell.rowconfigure(0, weight=1)

    sidebar = tk.Frame(shell, bg=SIDEBAR, width=218)
    sidebar.grid(row=0, column=0, sticky="ns")
    sidebar.grid_propagate(False)
    _text(sidebar, "VEHICLE", size=11, weight="bold", color="#8DC6AA", bg=SIDEBAR).pack(anchor="w", padx=26, pady=(34, 0))
    _text(sidebar, "Tracking", size=25, weight="bold", color=WHITE, bg=SIDEBAR).pack(anchor="w", padx=24, pady=(0, 40))
    nav = tk.Frame(sidebar, bg=SIDEBAR)
    nav.pack(fill="x", padx=13)
    app._nav_buttons = {}
    for key, label in (("live", "Live monitor"), ("history", "Session history"),
                       ("settings", "Model settings")):
        button = tk.Button(nav, text=label, anchor="w", padx=16, pady=14, bd=0,
                           bg=SIDEBAR, fg="#B5C8BD", activebackground="#244136",
                           activeforeground=WHITE, font=("Helvetica", 10, "bold"),
                           relief="flat", cursor="hand2", command=lambda page=key: show_page(app, page))
        button.pack(fill="x", pady=3)
        app._nav_buttons[key] = button
    tk.Label(sidebar, text="LOCAL AI  /  ONNX", bg=SIDEBAR, fg="#8AA498",
             font=("Helvetica", 9)).pack(side="bottom", anchor="w", padx=26, pady=25)

    content = tk.Frame(shell, bg=BG)
    content.grid(row=0, column=1, sticky="nsew", padx=28, pady=24)
    content.columnconfigure(0, weight=1)
    content.rowconfigure(1, weight=1)
    header = tk.Frame(content, bg=BG)
    header.grid(row=0, column=0, sticky="ew", pady=(0, 18))
    header.columnconfigure(0, weight=1)
    app._page_title = _text(header, "Live monitor", size=23, weight="bold", bg=BG)
    app._page_title.grid(row=0, column=0, sticky="w")
    app.status_label = tk.Label(header, textvariable=app.status, bg="#E1F1E8", fg="#126B52",
                                padx=14, pady=8, font=("Helvetica", 10, "bold"))
    app.status_label.grid(row=0, column=1, sticky="e")

    stack = tk.Frame(content, bg=BG)
    stack.grid(row=1, column=0, sticky="nsew")
    stack.columnconfigure(0, weight=1)
    stack.rowconfigure(0, weight=1)
    app._pages = {}
    for name in ("live", "history", "settings"):
        page = tk.Frame(stack, bg=BG)
        page.grid(row=0, column=0, sticky="nsew")
        app._pages[name] = page
    _live_page(app, app._pages["live"])
    _history_page(app, app._pages["history"])
    _settings_page(app, app._pages["settings"])
    tk.Label(content, textvariable=app.message, bg=BG, fg=MUTED,
             font=("Helvetica", 9), anchor="w").grid(row=2, column=0, sticky="ew", pady=(14, 0))
    show_page(app, "live")


def show_page(app, name: str) -> None:
    app._pages[name].tkraise()
    app.root.unbind_all("<MouseWheel>")
    app.root.unbind_all("<Button-4>")
    app.root.unbind_all("<Button-5>")
    if name == "settings":
        app.root.bind_all("<MouseWheel>", lambda event: app._settings_canvas.yview_scroll(
            -int(event.delta / 120) if event.delta else 0, "units"))
        app.root.bind_all("<Button-4>", lambda _event: app._settings_canvas.yview_scroll(-1, "units"))
        app.root.bind_all("<Button-5>", lambda _event: app._settings_canvas.yview_scroll(1, "units"))
    app._page_title.configure(text={"live": "Live monitor", "history": "Session history",
                                    "settings": "Model settings"}[name])
    for key, button in app._nav_buttons.items():
        button.configure(bg="#244536" if key == name else SIDEBAR,
                         fg=WHITE if key == name else "#B5C8BD")


def _live_page(app, page) -> None:
    page.columnconfigure(0, weight=1)
    page.rowconfigure(0, weight=1)
    preview = _card(page, 20)
    preview.grid(row=0, column=0, sticky="nsew")
    preview.columnconfigure(0, weight=1)
    preview.rowconfigure(2, weight=1)
    top = ttk.Frame(preview)
    top.grid(row=0, column=0, sticky="ew", pady=(0, 14))
    _text(top, "Live preview", size=12, weight="bold").pack(side="left")
    app.details_button = ttk.Button(top, text="Show details", command=app._toggle_inference_details)
    app.details_button.pack(side="right", padx=(12, 0))
    tk.Label(top, textvariable=app.counts, bg=WHITE, fg=MUTED,
             font=("Helvetica", 10)).pack(side="right")
    metrics = tk.Frame(preview, bg=WHITE)
    metrics.grid(row=1, column=0, sticky="ew", pady=(0, 14))
    for column, (title, value) in enumerate((("Processing speed", app.fps_value),
                                             ("Avg inference / frame", app.inference_frame_value),
                                             ("Total inference time", app.inference_total_value))):
        metrics.columnconfigure(column, weight=1, uniform="live_metrics")
        tile = tk.Frame(metrics, bg="#F7FAF8", padx=14, pady=8,
                        highlightbackground=LINE, highlightthickness=1)
        tile.grid(row=0, column=column, sticky="ew", padx=(0, 10) if column < 2 else 0)
        _text(tile, title, size=9, color=MUTED, bg="#F7FAF8").pack(anchor="w")
        tk.Label(tile, textvariable=value, bg="#F7FAF8", fg=INK,
                 font=("Helvetica", 14, "bold"), anchor="w").pack(anchor="w", pady=(3, 0))
    app.preview_label = tk.Label(preview, text="Select a source, then start AI", bg="#101C18",
                                 fg="#90A59A", font=("Helvetica", 11))
    app.preview_label.grid(row=2, column=0, sticky="nsew")
    details = tk.Frame(app.preview_label, bg=WHITE, padx=10, pady=6,
                       highlightbackground=LINE, highlightthickness=1)
    app.inference_details_panel = details
    for column, title in enumerate(("Stage", "Last frame", "Avg / call", "Calls")):
        _text(details, title, size=9, weight="bold", color=MUTED).grid(
            row=0, column=column, sticky="w", padx=(0, 12) if column < 3 else 0, pady=(0, 3))
    stages = (("vehicle_detection", "Vehicle detection"), ("tracking", "Tracking"),
              ("plate_detection", "Plate detection"), ("ocr", "OCR"),
              ("other", "Other pipeline work"))
    for row, (name, title) in enumerate(stages, start=1):
        _text(details, title, size=9).grid(row=row, column=0, sticky="w", pady=1)
        for column, field in enumerate(("last", "average", "calls"), start=1):
            tk.Label(details, textvariable=app.stage_values[name][field], bg=WHITE, fg=INK,
                     font=("Helvetica", 9), anchor="w").grid(row=row, column=column, sticky="w", pady=1)
    tk.Label(preview, text="Orange lines show vehicle trajectories in the frame.",
             bg=WHITE, fg=MUTED, font=("Helvetica", 9)).grid(row=3, column=0, sticky="w", pady=(12, 0))

    controls = _card(page, 20)
    controls.grid(row=1, column=0, sticky="ew", pady=(16, 0))
    controls.columnconfigure(1, weight=1)
    _text(controls, "Video source", size=12, weight="bold").grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 16))
    app.kind_box = ttk.Combobox(controls, textvariable=app.kind, values=("video", "rtsp", "usb"), state="readonly", width=13)
    app.kind_box.grid(row=1, column=0, sticky="ew", padx=(0, 10))
    app.kind_box.bind("<<ComboboxSelected>>", lambda _e: app._source_mode())
    app.uri_entry = ttk.Entry(controls, textvariable=app.uri)
    app.uri_entry.grid(row=1, column=1, sticky="ew", padx=(0, 10))
    app.browse_button = ttk.Button(controls, text="Browse video", command=app._browse)
    app.browse_button.grid(row=1, column=2, sticky="ew", padx=(0, 18))
    app.start_button = ttk.Button(controls, text="Start AI", style="Primary.TButton", command=app._start)
    app.start_button.grid(row=1, column=3, sticky="ew", padx=(0, 8))
    app.stop_button = ttk.Button(controls, text="Stop", command=app._stop, state="disabled")
    app.stop_button.grid(row=1, column=4, sticky="ew")
    app.source_hint = ttk.Label(controls, text="Select a video file on this computer", style="Muted.TLabel")
    app.source_hint.grid(row=2, column=0, columnspan=5, sticky="w", pady=(10, 0))


def _history_page(app, page) -> None:
    page.columnconfigure(0, weight=1)
    page.rowconfigure(0, weight=1)
    panel = _card(page, 22)
    panel.grid(row=0, column=0, sticky="nsew")
    panel.columnconfigure(0, weight=1)
    panel.rowconfigure(2, weight=1)
    top = ttk.Frame(panel)
    top.grid(row=0, column=0, sticky="ew", pady=(0, 20))
    _text(top, "Detected vehicles and license plates", size=12, weight="bold").pack(side="left")
    ttk.Button(top, text="Export Excel", command=app._export).pack(side="right")
    ttk.Button(top, text="Retry selected OCR", command=app._retry_ocr).pack(side="right", padx=(0, 10))
    filters = ttk.Frame(panel)
    filters.grid(row=1, column=0, sticky="ew", pady=(0, 16))
    filters.columnconfigure(0, weight=2)
    filters.columnconfigure(1, weight=1)
    app.session_box = ttk.Combobox(filters, textvariable=app.session, state="readonly")
    app.session_box.grid(row=0, column=0, sticky="ew", padx=(0, 12))
    app.session_box.bind("<<ComboboxSelected>>", app._session_changed)
    search_entry = ttk.Entry(filters, textvariable=app.search)
    search_entry.grid(row=0, column=1, sticky="ew")
    app.search.trace_add("write", lambda *_: app._refresh_tracks())
    columns = ("track", "plate", "class", "time", "points")
    app.table = ttk.Treeview(panel, columns=columns, show="headings", height=6)
    for key, title, width in (("track", "Track", 85), ("plate", "License plate", 160),
                              ("class", "Vehicle type", 120), ("time", "Time", 170), ("points", "Points", 75)):
        app.table.heading(key, text=title)
        app.table.column(key, width=width, stretch=key in ("plate", "time"))
    app.table.grid(row=2, column=0, sticky="nsew")
    app.table.bind("<<TreeviewSelect>>", lambda _e: app._select_track())
    details = tk.Frame(panel, bg=WHITE)
    details.grid(row=3, column=0, sticky="ew", pady=(22, 0))
    for column, weight in enumerate((2, 1, 1)):
        details.columnconfigure(column, weight=weight, minsize=190)
    for column, title in enumerate(("Trajectory on last frame", "Vehicle evidence", "Plate evidence")):
        _text(details, title, size=12, weight="bold").grid(row=0, column=column, sticky="w", pady=(0, 10))
    app.path_canvas = tk.Canvas(details, height=220, bg="#F7FAF8", highlightbackground=LINE, highlightthickness=1)
    app.path_canvas.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
    app.path_canvas.bind("<Configure>", lambda _e: app._draw_trajectory())
    app.trajectory_info = tk.StringVar(value="")
    tk.Label(details, textvariable=app.trajectory_info, bg=WHITE, fg=MUTED,
             font=("Helvetica", 9), anchor="w").grid(row=2, column=0, sticky="ew", pady=(7, 0))
    for column, kind, empty in ((1, "vehicle", "Select a vehicle"), (2, "plate", "No plate image")):
        holder = tk.Frame(details, height=220, bg="#F7FAF8", highlightbackground=LINE, highlightthickness=1)
        holder.grid(row=1, column=column, sticky="nsew", padx=(0, 12) if column == 1 else 0)
        holder.grid_propagate(False)
        label = tk.Label(holder, text=empty, bg="#F7FAF8", fg=MUTED, font=("Helvetica", 10))
        label.pack(fill="both", expand=True)
        label.bind("<Configure>", lambda _event, image_kind=kind: app._render_evidence(image_kind))
        setattr(app, f"{kind}_evidence_label", label)
        info = tk.StringVar(value="")
        setattr(app, f"{kind}_evidence_info", info)
        tk.Label(details, textvariable=info, bg=WHITE, fg=MUTED,
                 font=("Helvetica", 9), anchor="w").grid(row=2, column=column, sticky="ew", pady=(7, 0))


def _settings_page(app, page) -> None:
    page.columnconfigure(0, weight=1)
    page.rowconfigure(0, weight=1)
    canvas = tk.Canvas(page, bg=BG, highlightthickness=0)
    app._settings_canvas = canvas
    canvas.grid(row=0, column=0, sticky="nsew")
    bar = ttk.Scrollbar(page, orient="vertical", command=canvas.yview)
    bar.grid(row=0, column=1, sticky="ns")
    canvas.configure(yscrollcommand=bar.set)
    inside = tk.Frame(canvas, bg=BG)
    window = canvas.create_window((0, 0), window=inside, anchor="nw")
    inside.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
    inside.columnconfigure(0, weight=1)
    inside.columnconfigure(1, weight=1)
    registry = ModelRegistry.from_json(CATALOG)
    choices = {
        "vehicle_backend": vehicle_backends(),
        "vehicle_model": [m.model_id for m in registry._models.values() if m.task == "object_detection"],
        "plate_backend": plate_backends(),
        "plate_model": [m.model_id for m in registry._models.values() if m.task == "license_plate_detection"],
        "tracker_backend": tracker_backends(),
        "tracker": available_trackers(),
        "device": ("auto", "cpu", "cuda"),
        "ocr_backend": ocr_backends(),
        "ocr_preprocess_mode": available_preprocessors(),
    }
    columns = []
    for column in range(2):
        column_frame = tk.Frame(inside, bg=BG)
        column_frame.grid(row=0, column=column, sticky="new", padx=(0, 12) if column == 0 else 0)
        columns.append(column_frame)
    for group, title, fields, column in SETTINGS_GROUPS:
        card = _card(columns[column], 22)
        card.pack(fill="x", pady=(0, 14))
        card.columnconfigure(0, weight=1)
        header = ttk.Frame(card)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        header.columnconfigure(0, weight=1)
        _text(header, title, size=12, weight="bold").grid(row=0, column=0, sticky="w")
        actions = ttk.Frame(header)
        actions.grid(row=1, column=0, sticky="e", pady=(10, 0))
        ttk.Button(actions, text="Save", style="Primary.TButton",
                   command=lambda key=group: app._save_group(key)).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Reset defaults",
                   command=lambda key=group: app._reset_group(key)).pack(side="left")
        for index, (key, label) in enumerate(fields, 1):
            _text(card, label, size=9, color=MUTED).grid(row=2*index-1, column=0, sticky="w", pady=(10, 5))
            if key in choices:
                state = "normal" if key in ("vehicle_model", "plate_model", "tracker") else "readonly"
                entry = ttk.Combobox(card, textvariable=app.vars[key], values=choices[key], state=state)
            else:
                entry = ttk.Entry(card, textvariable=app.vars[key])
            entry.grid(row=2*index, column=0, sticky="ew")
