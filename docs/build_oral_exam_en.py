"""Generate English figures and the equation-rich Word examination guide."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from docx import Document
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/ORAL_EXAM_EN_MATHEMATICAL_GUIDE.md"
FIG = ROOT / "docs/figures/oral_exam_en"
OUT = ROOT / "docs/UAV_Tracking_Oral_Exam_English_Mathematical_Guide.docx"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "savefig.dpi": 180})
NAVY, BLUE, GREEN, ORANGE = "#193247", "#24557a", "#328576", "#d97c31"


def box(ax, xy, wh, label, color=NAVY, font=10):
    x, y = xy
    w, h = wh
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.025",
                                linewidth=0, facecolor=color))
    ax.text(x+w/2, y+h/2, label, color="white", ha="center", va="center", fontsize=font, linespacing=1.3)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=17, color="#79909f", lw=1.7))


def protocol():
    fig, ax = plt.subplots(figsize=(10.3, 4.2)); ax.axis("off"); ax.set(xlim=(0, 10), ylim=(0, 4))
    box(ax, (.15, 2.35), (1.8, .86), "DET train\n6,414 images", BLUE)
    box(ax, (2.65, 2.35), (1.8, .86), "DET val\n548 images", GREEN)
    box(ax, (5.17, 2.35), (2.18, .86), "Choose detector\ncheckpoint", NAVY)
    box(ax, (7.87, 2.35), (1.92, .86), "3 detector\ncaches", BLUE)
    for start, end in [((1.98,2.78),(2.60,2.78)),((4.48,2.78),(5.10,2.78)),((7.38,2.78),(7.80,2.78))]: arrow(ax,start,end)
    box(ax, (.15, .77), (1.8, .86), "MOT val\n2,846 frames", ORANGE)
    box(ax, (2.65, .77), (1.8, .86), "Tune 15\nthresholds", ORANGE)
    box(ax, (5.17, .77), (2.18, .86), "Freeze settings\nfor all 15 runs", NAVY)
    box(ax, (7.87, .77), (1.92, .86), "MOT test-dev\n6,635 frames", GREEN)
    for start, end in [((1.98,1.20),(2.60,1.20)),((4.48,1.20),(5.10,1.20)),((7.38,1.20),(7.80,1.20))]: arrow(ax,start,end)
    ax.text(5, 3.68, "57 overlapping-source DET images were excluded from training.", ha="center", color=NAVY, fontsize=11)
    fig.savefig(FIG/"01_protocol.png", bbox_inches="tight", facecolor="white"); plt.close(fig)


def detectors():
    fig, axes = plt.subplots(3, 1, figsize=(11.2, 6.1))
    rows = [
        ("Faster R-CNN / two stage", ["Image\n800–1333", "ResNet-50\n+ FPN", "RPN\nproposals", "RoIAlign\n+ class head", "NMS\nfinal boxes"], BLUE),
        ("YOLO11m / one stage", ["Image\n640", "C3k2\n+ C2PSA", "Multi-scale\nneck", "Dense head\nstrides 8/16/32", "NMS\nfinal boxes"], GREEN),
        ("RF-DETR-M / set prediction", ["Image\n576", "DINOv2\nViT", "Object\nqueries", "Deformable\ndecoder", "Set output\nno NMS"], ORANGE),
    ]
    for ax, (title, labels, color) in zip(axes, rows):
        ax.axis("off"); ax.set(xlim=(0, 10.5), ylim=(0, 1))
        ax.text(.02, .95, title, fontsize=12.5, color=NAVY, va="top")
        for i, label in enumerate(labels):
            x=.02+2.1*i; box(ax,(x,.13),(1.8,.49),label,color if i in (1,3) else NAVY,9.5)
            if i: arrow(ax,(x-.29,.375),(x-.04,.375))
    fig.tight_layout(h_pad=.7); fig.savefig(FIG/"02_detectors.png", bbox_inches="tight", facecolor="white"); plt.close(fig)


def trackers():
    names=["SORT","DeepSORT","ByteTrack","OC-SORT","BoT-SORT"]
    cols=["Kalman + IoU","ReID","Low-score\nsecond stage","Observation\nrecovery","Camera motion\ncompensation"]
    vals=np.array([[1,0,0,0,0],[1,1,0,0,0],[1,0,1,0,0],[1,0,0,1,0],[1,1,1,0,1]])
    fig,ax=plt.subplots(figsize=(10.7,4.1))
    ax.imshow(vals,cmap=matplotlib.colors.ListedColormap(["#edf2f6","#5a9c93"]),vmin=0,vmax=1,aspect="auto")
    ax.set_xticks(range(5),cols); ax.set_yticks(range(5),names)
    for i in range(5):
        for j in range(5): ax.text(j,i,"Yes" if vals[i,j] else "—",ha="center",va="center",color="white" if vals[i,j] else "#9baab3")
    ax.set_xticks(np.arange(-.5,5,1),minor=True); ax.set_yticks(np.arange(-.5,5,1),minor=True)
    ax.grid(which="minor",color="white",linewidth=2); ax.tick_params(which="both",length=0)
    fig.tight_layout(); fig.savefig(FIG/"03_trackers.png",bbox_inches="tight",facecolor="white"); plt.close(fig)


def hota():
    vals=np.array([[45.32,46.38,47.09,46.51,51.67],[45.10,45.51,46.64,46.22,51.68],[43.11,42.87,45.97,43.78,49.85]])
    fig,ax=plt.subplots(figsize=(10.2,3.7)); ax.imshow(vals,cmap="YlGnBu",vmin=40,vmax=53,aspect="auto")
    ax.set_xticks(range(5),["SORT","DeepSORT","ByteTrack","OC-SORT","BoT-SORT"])
    ax.set_yticks(range(3),["YOLO11m","Faster R-CNN","RF-DETR-M"])
    for i in range(3):
        for j in range(5): ax.text(j,i,f"{vals[i,j]:.2f}",ha="center",va="center",color="white" if vals[i,j]>47 else NAVY,fontsize=11)
    ax.tick_params(length=0); fig.tight_layout(); fig.savefig(FIG/"04_hota.png",bbox_inches="tight",facecolor="white"); plt.close(fig)


def tradeoff():
    yolo=[(101.09,45.32,"SORT"),(25.65,46.38,"DeepSORT"),(101.01,47.09,"ByteTrack"),(92.29,46.51,"OC-SORT"),(25.87,51.67,"BoT-SORT")]
    frcnn=[(13.38,45.10,"SORT"),(9.79,45.51,"DeepSORT"),(13.33,46.64,"ByteTrack"),(13.14,46.22,"OC-SORT"),(9.15,51.68,"BoT-SORT")]
    rf=[(31.3,43.11,"SORT"),(17.94,42.87,"DeepSORT"),(31.49,45.97,"ByteTrack"),(30.42,43.78,"OC-SORT"),(16.25,49.85,"BoT-SORT")]
    fig,ax=plt.subplots(figsize=(9.8,5.0))
    for label,data,color in [("YOLO11m",yolo,BLUE),("Faster R-CNN",frcnn,ORANGE),("RF-DETR-M",rf,GREEN)]:
        ax.scatter([x[0] for x in data],[x[1] for x in data],s=82,label=label,color=color)
        for fps,h,name in data:
            if name in ("ByteTrack","BoT-SORT"): ax.annotate(name,(fps,h),xytext=(4,6),textcoords="offset points",fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("Combined detector + tracker processing FPS (L4, log scale)")
    ax.set_ylabel("MOT test-dev HOTA (%)"); ax.set_ylim(41,53); ax.grid(alpha=.2)
    ax.legend(frameon=False,ncol=3,loc="lower right"); fig.tight_layout()
    fig.savefig(FIG/"05_tradeoff.png",bbox_inches="tight",facecolor="white"); plt.close(fig)


def bytetrack():
    fig, ax = plt.subplots(figsize=(10.6, 4.0)); ax.axis("off"); ax.set(xlim=(0, 10), ylim=(0, 4))
    box(ax, (.1, 2.65), (1.65, .77), "Detector boxes\nscore ≥ 0.05", NAVY, 10)
    box(ax, (2.35, 2.65), (1.65, .77), "High score\ns > τH", BLUE, 10)
    box(ax, (4.75, 2.65), (2.0, .77), "Stage 1\ntrack ↔ high", BLUE, 10)
    box(ax, (7.54, 2.65), (2.2, .77), "Matched tracks\nkeep their IDs", GREEN, 10)
    arrow(ax,(1.78,3.03),(2.30,3.03)); arrow(ax,(4.03,3.03),(4.70,3.03)); arrow(ax,(6.78,3.03),(7.49,3.03))
    box(ax, (2.35, .72), (1.65, .77), "Low score\nτL < s < τH", ORANGE, 10)
    box(ax, (4.75, .72), (2.0, .77), "Stage 2\nunmatched ↔ low", ORANGE, 10)
    box(ax, (7.54, .72), (2.2, .77), "Recovered old ID\nno new track", GREEN, 10)
    arrow(ax,(1.40,2.60),(2.70,1.55)); arrow(ax,(4.03,1.10),(4.70,1.10)); arrow(ax,(6.78,1.10),(7.49,1.10))
    ax.text(5, .20, "Only unmatched high-score boxes above the birth threshold can start new IDs.", ha="center", color=NAVY)
    fig.savefig(FIG/"06_bytetrack.png",bbox_inches="tight",facecolor="white"); plt.close(fig)


def reflow():
    original=SOURCE.read_text(encoding="utf-8")
    blocks=original.split("\n\n")
    result=[]
    for block in blocks:
        if block.startswith(("#","![","$$")) or block.startswith("    "):
            result.append(block); continue
        sentences=re.split(r"(?<=[.!?])\s+(?=[A-Z])",block.replace("\n"," "))
        current=""
        for sentence in sentences:
            if current and len((current+" "+sentence).split())>68:
                result.append(current); current=sentence
            else: current=(current+" "+sentence).strip()
        if current: result.append(current)
    SOURCE.write_text("\n\n".join(result).strip()+"\n",encoding="utf-8")


def build():
    subprocess.run(["pandoc",str(SOURCE),"--from=markdown+tex_math_dollars", "--resource-path="+str(ROOT/"docs"),
                    "--standalone","--output="+str(OUT)],check=True)
    doc=Document(OUT)
    sec=doc.sections[0]; sec.top_margin=Cm(2.0); sec.bottom_margin=Cm(2.0); sec.left_margin=Cm(2.25); sec.right_margin=Cm(2.2)
    norm=doc.styles["Normal"]
    norm.font.name="Arial"; norm.font.size=Pt(10.5); norm.font.color.rgb=RGBColor(31,47,61)
    norm.paragraph_format.line_spacing=1.22; norm.paragraph_format.space_after=Pt(6)
    for name,size,before,after in [("Title",20,0,14),("Heading 1",14,15,7),("Heading 2",11.5,11,4)]:
        st=next(s for s in doc.styles if s.name==name)
        st.font.name="Arial"; st.font.size=Pt(size); st.font.bold=False; st.font.color.rgb=RGBColor(36,85,122)
        st.paragraph_format.space_before=Pt(before); st.paragraph_format.space_after=Pt(after); st.paragraph_format.keep_with_next=True
    for p in doc.paragraphs:
        if p.style.name=="Image Caption":
            p.style.font.name="Arial"; p.style.font.size=Pt(9)
        if not p.text and not p._p.xpath('.//m:oMath') and not p._p.xpath('.//w:drawing'):
            p.paragraph_format.space_after=Pt(0)
    footer=sec.footer.paragraphs[0]; footer.alignment=2
    footer.add_run("UAV Tracking oral examination · English mathematical guide").font.size=Pt(8)
    doc.save(OUT)
    print(OUT)


if __name__=="__main__":
    reflow()
    for fn in (protocol,detectors,trackers,hota,tradeoff,bytetrack): fn()
    build()
