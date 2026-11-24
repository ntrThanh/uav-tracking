"""Build the readable, illustrated oral-exam handout from its Markdown source."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/ON_VAN_DAP_CHI_TIET.md"
FIG = ROOT / "docs/figures/on_van_dap"
OUT = ROOT / "docs/On_van_dap_UAV_Tracking_chi_tiet.docx"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "savefig.dpi": 180})
BLUE, LIGHT, DARK, ORANGE, GREEN = "#24557a", "#eaf2f7", "#193247", "#d97c31", "#328576"


def box(ax, xy, wh, label, color=BLUE, size=12):
    x, y = xy
    w, h = wh
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.018,rounding_size=0.025",
                                facecolor=color, edgecolor="none"))
    ax.text(x + w / 2, y + h / 2, label, color="white", ha="center", va="center",
            fontsize=size, fontweight="medium", linespacing=1.35)


def arrow(ax, a, b, color="#687e8f"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16,
                                 lw=1.8, color=color, connectionstyle="arc3"))


def make_pipeline():
    fig, ax = plt.subplots(figsize=(11.4, 3.25))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    labels = ["Video UAV\n(frame t)", "Detector\n10 lớp", "DET cache\nscore ≥ 0,05", "Tracker\n5 lớp", "Quỹ đạo\nID theo thời gian", "TrackEval\nHOTA · IDF1"]
    xs = np.linspace(.018, .85, 6)
    for i, (x, label) in enumerate(zip(xs, labels)):
        box(ax, (x, .46), (.13, .26), label, [DARK, BLUE, GREEN, BLUE, GREEN, DARK][i], 10.5)
        if i:
            arrow(ax, (xs[i - 1] + .135, .59), (x - .006, .59))
    ax.text(.50, .20, "5 tracker dùng chung một cache của từng detector; chọn ngưỡng trên MOT val, chấm trên MOT test-dev.",
            ha="center", color=DARK, fontsize=11)
    fig.savefig(FIG / "01_pipeline.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_split():
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    ax.set(xlim=(0, 10), ylim=(0, 3.3))
    ax.axis("off")
    box(ax, (.25, 1.87), (2.0, .92), "DET train\n6.414 ảnh", BLUE, 12)
    box(ax, (3.05, 1.87), (2.0, .92), "DET val\n548 ảnh", GREEN, 12)
    box(ax, (6.42, 1.87), (2.9, .92), "Checkpoint\n3 detector", DARK, 12)
    arrow(ax, (2.3, 2.33), (2.99, 2.33)); arrow(ax, (5.1, 2.33), (6.35, 2.33))
    box(ax, (.25, .34), (2.0, .92), "MOT val\n2.846 frame", ORANGE, 12)
    box(ax, (3.05, .34), (2.0, .92), "Chọn ngưỡng\n15 tổ hợp", ORANGE, 12)
    box(ax, (6.42, .34), (2.9, .92), "MOT test-dev\n6.635 frame · báo cáo", DARK, 12)
    arrow(ax, (2.3, .80), (2.99, .80)); arrow(ax, (5.1, .80), (6.35, .80))
    ax.text(.30, 3.08, "57 ảnh DET trùng nguồn video MOT val/test-dev đã loại khỏi tập train.", fontsize=11, color=DARK)
    fig.savefig(FIG / "02_data_protocol.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_distribution():
    names = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
    train = np.array([21.2, 8.5, 3.6, 45.7, 4.2, 2.8, 2.6, 1.2, .9, 9.3])
    val = np.array([28.4, 15.7, 5.3, 27.9, 6.0, 1.2, 3.3, 1.5, .2, 10.5])
    test = np.array([34.5, 11.0, 1.2, 32.4, 3.6, 4.5, 2.2, 1.4, .5, 8.7])
    fig, ax = plt.subplots(figsize=(10.4, 5.7))
    y = np.arange(len(names))
    ax.barh(y - .26, train, height=.24, label="MOT train", color=BLUE)
    ax.barh(y, val, height=.24, label="MOT val", color=GREEN)
    ax.barh(y + .26, test, height=.24, label="MOT test-dev", color=ORANGE)
    ax.set_yticks(y, names); ax.invert_yaxis(); ax.set_xlabel("Tỉ lệ box trong split (%)")
    ax.set_xlim(0, 50); ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis="x", alpha=.18); ax.set_axisbelow(True); ax.legend(ncol=3, loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "03_class_distribution.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_small():
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    vals = [24.9, 19.2, 47.8]
    bars = ax.barh([2, 1, 0], vals, height=.53, color=[BLUE, GREEN, ORANGE])
    ax.set_yticks([2, 1, 0], ["MOT train", "MOT val", "MOT test-dev"])
    ax.set_xlim(0, 55); ax.set_xlabel("Tỉ lệ box nhỏ hơn 32 × 32 px (%)")
    for bar, value in zip(bars, vals):
        ax.text(value + 1, bar.get_y() + bar.get_height()/2, f"{value:.1f}%", va="center", color=DARK, fontsize=12)
    ax.spines[['top', 'right', 'left']].set_visible(False); ax.grid(axis="x", alpha=.18); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(FIG / "04_small_objects.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_architecture():
    fig, axs = plt.subplots(3, 1, figsize=(11.6, 6.3))
    rows = [
        ("Faster R-CNN · hai giai đoạn", ["Ảnh 800–1333", "ResNet-50 + FPN", "RPN · proposal", "RoIAlign + head", "NMS · box"], BLUE),
        ("YOLO11m · một giai đoạn", ["Ảnh 640", "C3k2 + C2PSA", "Neck đa mức", "Head stride 8/16/32", "NMS · box"], GREEN),
        ("RF-DETR-M · dự đoán tập", ["Ảnh 576", "ViT DINOv2", "Object queries", "Deformable decoder", "300 box · không NMS"], ORANGE),
    ]
    for ax, (title, labels, color) in zip(axs, rows):
        ax.set(xlim=(0, 10.5), ylim=(0, 1)); ax.axis("off")
        ax.text(.03, .95, title, fontsize=12, color=DARK, va="top")
        for j, label in enumerate(labels):
            x = .02 + j * 2.1
            box(ax, (x, .16), (1.80, .48), label, color if j in (1, 3) else DARK, 9.4)
            if j: arrow(ax, (x - .30, .40), (x - .04, .40))
    fig.tight_layout(h_pad=.6); fig.savefig(FIG / "05_detector_architectures.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_trackers():
    names = ["SORT", "DeepSORT", "ByteTrack", "OC-SORT", "BoT-SORT"]
    cols = ["Kalman + IoU", "ReID", "Score thấp", "Quan sát", "GMC"]
    values = np.array([[1,0,0,0,0],[1,1,0,0,0],[1,0,1,0,0],[1,0,0,1,0],[1,1,1,0,1]])
    fig, ax = plt.subplots(figsize=(9.8, 3.9))
    ax.imshow(values, cmap=matplotlib.colors.ListedColormap(["#edf2f6", "#5a9c93"]), aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(5), cols); ax.set_yticks(range(5), names)
    for i in range(5):
        for j in range(5): ax.text(j, i, "Có" if values[i,j] else "—", ha="center", va="center", color="white" if values[i,j] else "#8b9aa6")
    ax.set_xticks(np.arange(-.5, 5, 1), minor=True); ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="both", length=0)
    fig.tight_layout(); fig.savefig(FIG / "06_tracker_components.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_heatmap():
    order_det = ["YOLO11m", "Faster R-CNN", "RF-DETR-M"]
    order_track = ["SORT", "DeepSORT", "ByteTrack", "OC-SORT", "BoT-SORT"]
    vals = np.array([[45.32,46.38,47.09,46.51,51.67],[45.10,45.51,46.64,46.22,51.68],[43.11,42.87,45.97,43.78,49.85]])
    fig, ax = plt.subplots(figsize=(9.8, 3.55))
    ax.imshow(vals, cmap="YlGnBu", vmin=40, vmax=53, aspect="auto")
    ax.set_xticks(range(5), order_track); ax.set_yticks(range(3), order_det)
    for i in range(3):
        for j in range(5): ax.text(j, i, f"{vals[i,j]:.2f}", ha="center", va="center", color="white" if vals[i,j] > 47 else DARK, fontsize=11)
    ax.tick_params(length=0); fig.tight_layout(); fig.savefig(FIG / "07_hota_matrix.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_tradeoff():
    rows = list(csv.DictReader((ROOT / "results/tables/matrix_testdev_mean.csv").open()))
    ids = {"yolo11m": "YOLO11m", "frcnn_r50": "Faster R-CNN", "rfdetr_m": "RF-DETR-M"}
    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    for det, label, color in [("yolo11m", "YOLO11m", BLUE), ("frcnn_r50", "Faster R-CNN", ORANGE), ("rfdetr_m", "RF-DETR-M", GREEN)]:
        subset = [r for r in rows if r["det"] == det and "_thr" in r["tracker"]]
        ax.scatter([float(r["FPS_e2e"]) for r in subset], [float(r["HOTA"]) for r in subset], s=78, color=color, label=label)
        for r in subset:
            if r["tracker"].startswith(("botsort", "bytetrack")):
                name = "BoT-SORT" if r["tracker"].startswith("botsort") else "ByteTrack"
                ax.annotate(name, (float(r["FPS_e2e"]), float(r["HOTA"])), xytext=(4, 6), textcoords="offset points", fontsize=8.5)
    ax.set_xscale("log"); ax.set_xlabel("FPS xử lý ghép detector + tracker (L4, log)")
    ax.set_ylabel("HOTA trên MOT test-dev (%)"); ax.set_ylim(41, 53); ax.grid(alpha=.2)
    ax.legend(frameon=False, ncol=3, loc="lower right"); fig.tight_layout()
    fig.savefig(FIG / "08_speed_quality.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def split_sentences(text):
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-ỸĐ])", text)
    chunks = []
    current = ""
    for sentence in pieces:
        if len((current + " " + sentence).split()) > 54 and current:
            chunks.append(current.strip()); current = sentence
        else:
            current = (current + " " + sentence).strip()
    if current: chunks.append(current)
    out = []
    for chunk in chunks:
        if len(chunk.split()) > 95 and "; " in chunk:
            bits = chunk.split("; "); group = ""
            for bit in bits:
                if len((group + "; " + bit).split()) > 75 and group:
                    out.append(group.strip()); group = bit
                else:
                    group = (group + "; " + bit).strip("; ")
            if group: out.append(group)
        else: out.append(chunk)
    return out


def reflow_source():
    original = SOURCE.read_text(encoding="utf-8")
    blocks = original.split("\n\n")
    new = []
    for block in blocks:
        if block.startswith("#"):
            new.append(block)
        else:
            new.extend(split_sentences(block.replace("\n", " ")))
    SOURCE.write_text("\n\n".join(new).strip() + "\n", encoding="utf-8")


def build_doc():
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(2.1); section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.3); section.right_margin = Cm(2.2)
    section.header_distance = Cm(1); section.footer_distance = Cm(1)
    normal = doc.styles["Normal"]
    normal.font.name = "Arial"; normal.font.size = Pt(10.5); normal.font.color.rgb = RGBColor(31, 47, 61)
    normal.paragraph_format.line_spacing = 1.24
    normal.paragraph_format.space_after = Pt(6)
    h1 = doc.styles["Title"]
    h1.font.name = "Arial"; h1.font.size = Pt(20); h1.font.bold = False; h1.font.color.rgb = RGBColor(25, 50, 71)
    h1.paragraph_format.space_after = Pt(13)
    h2 = doc.styles["Heading 1"]
    h2.font.name = "Arial"; h2.font.size = Pt(13.5); h2.font.bold = False; h2.font.color.rgb = RGBColor(36, 85, 122)
    h2.paragraph_format.space_before = Pt(16); h2.paragraph_format.space_after = Pt(7)
    h2.paragraph_format.keep_with_next = True
    caption_style = doc.styles["Caption"]
    caption_style.font.name = "Arial"; caption_style.font.size = Pt(9); caption_style.font.italic = False
    caption_style.font.color.rgb = RGBColor(85, 101, 112)
    caption_style.paragraph_format.space_after = Pt(12)
    fig_for_section = {
        2: ("01_pipeline.png", "Hình 1. Luồng tracking by detection và giao thức cache dùng chung."),
        3: ("02_data_protocol.png", "Hình 2. Vai trò của DET và MOT trong huấn luyện, chọn tham số và đánh giá."),
        4: ("03_class_distribution.png", "Hình 3. Phân bố lớp thay đổi giữa các split MOT."),
        6: ("04_small_objects.png", "Hình 4. Tỉ lệ vật thể nhỏ tăng mạnh ở tập MOT test-dev."),
        9: ("05_detector_architectures.png", "Hình 5. Đường xử lý chính của ba kiến trúc detector."),
        15: ("06_tracker_components.png", "Hình 6. Thành phần khác nhau giữa năm tracker trong cấu hình nghiên cứu."),
        16: ("07_hota_matrix.png", "Hình 7. HOTA của 15 tổ hợp trên MOT test-dev sau chọn ngưỡng bằng val."),
        18: ("08_speed_quality.png", "Hình 8. Đánh đổi HOTA và FPS xử lý trên NVIDIA L4."),
    }
    current_section = None

    def add_figure(section_number):
        if section_number not in fig_for_section: return
        filename, caption = fig_for_section[section_number]
        para = doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_before = Pt(7)
        para.add_run().add_picture(str(FIG / filename), width=Cm(15.5))
        cp = doc.add_paragraph(caption, style="Caption")
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.keep_with_next = True

    for block in SOURCE.read_text(encoding="utf-8").strip().split("\n\n"):
        if block.startswith("# "):
            doc.add_paragraph(block[2:], style="Title")
        elif block.startswith("## "):
            if current_section is not None: add_figure(current_section)
            title = block[3:]
            current_section = int(title.split(".", 1)[0])
            doc.add_paragraph(title, style="Heading 1")
        else:
            p = doc.add_paragraph(block.replace("\n", " "))
            p.paragraph_format.widow_control = True
    if current_section is not None: add_figure(current_section)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run("Ôn vấn đáp UAV Tracking · 24/11/2026").font.size = Pt(8)
    doc.save(OUT)


if __name__ == "__main__":
    reflow_source()
    for func in (make_pipeline, make_split, make_distribution, make_small, make_architecture, make_trackers, make_heatmap, make_tradeoff):
        func()
    build_doc()
    print(OUT)
