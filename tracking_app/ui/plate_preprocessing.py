"""Inspect the OCR preprocessing stages for a saved plate crop."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from tracking_app.ocr.preprocess import PreprocessStage


def show_plate_preprocessing(root: tk.Tk, stages: list[PreprocessStage],
                             track_id: int, evidence: dict) -> tk.Toplevel:
    window = tk.Toplevel(root)
    window.title(f"Plate preprocessing — track #{track_id}")
    window.geometry("900x680")
    window.minsize(560, 400)
    window.transient(root)

    heading = ttk.Frame(window, padding=(20, 16))
    heading.pack(fill="x")
    ttk.Label(heading, text=f"Plate preprocessing — track #{track_id}",
              style="Section.TLabel").pack(anchor="w")
    ttk.Label(heading, text=f"Frame {evidence['frame_id']}  |  {evidence['video_time']:.1f} s",
              style="Muted.TLabel").pack(anchor="w", pady=(5, 0))
    ttk.Label(heading, text="Recreated from the saved plate JPEG with this session's OCR settings.",
              style="Muted.TLabel").pack(anchor="w", pady=(5, 0))
    if any(stage.ocr_attempt == 2 for stage in stages):
        ttk.Label(heading, text="OCR input 2 is tried only if input 1 returns no plate.",
                  style="Muted.TLabel").pack(anchor="w", pady=(5, 0))

    body = ttk.Frame(window)
    body.pack(fill="both", expand=True, padx=20, pady=(0, 20))
    canvas = tk.Canvas(body, bg="#F4F6F5", highlightthickness=0)
    scrollbar = ttk.Scrollbar(body, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    cards = ttk.Frame(canvas)
    for column in (0, 1):
        cards.columnconfigure(column, weight=1, uniform="stages")
    card_window = canvas.create_window((0, 0), window=cards, anchor="nw")
    cards.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda event: canvas.itemconfigure(card_window, width=event.width))
    window.bind("<MouseWheel>", lambda event: canvas.yview_scroll(-int(event.delta / 120), "units"))
    window.bind("<Button-4>", lambda _event: canvas.yview_scroll(-1, "units"))
    window.bind("<Button-5>", lambda _event: canvas.yview_scroll(1, "units"))

    window._photos = []
    for index, stage in enumerate(stages):
        card = tk.Frame(cards, bg="white", padx=12, pady=12)
        card.grid(row=index // 2, column=index % 2, sticky="nsew", padx=6, pady=6)
        title = stage.label
        if stage.ocr_attempt is not None:
            title += f"  ·  OCR input {stage.ocr_attempt}"
        ttk.Label(card, text=title, style="Section.TLabel").pack(fill="x")
        image = (Image.fromarray(stage.image) if stage.image.ndim == 2 else
                 Image.fromarray(cv2.cvtColor(stage.image, cv2.COLOR_BGR2RGB)))
        scale = min(380 / image.width, 240 / image.height, 4)
        image = image.resize((max(1, round(image.width * scale)),
                              max(1, round(image.height * scale))), Image.Resampling.NEAREST)
        photo = ImageTk.PhotoImage(image, master=window)
        window._photos.append(photo)
        tk.Label(card, image=photo, bg="white").pack(pady=(10, 0))
        tk.Label(card, text=f"{stage.image.shape[1]} × {stage.image.shape[0]} px",
                 bg="white", fg="#6C7D76", anchor="w").pack(fill="x", pady=(8, 0))

    window.focus_set()
    return window
