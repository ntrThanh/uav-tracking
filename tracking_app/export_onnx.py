"""Regenerate app ONNX checkpoints from the existing YOLO training weights."""
from pathlib import Path

from .config import ROOT

SOURCES = (
    ROOT / "tracking_app/models/vehicle/yolo11n.pt",
    ROOT / "weights/turning/T1_yolo11m_640_det/best_yolo11m.pt",
    ROOT / "tracking_app/models/plate/plate_yolo11s.pt",
)


def main() -> None:
    from ultralytics import YOLO

    for source in SOURCES:
        if not source.is_file():
            raise FileNotFoundError(source)
        output = Path(YOLO(str(source)).export(format="onnx", imgsz=640,
                                               dynamic=False, simplify=False, opset=17))
        print(f"{source.name} -> {output}")


if __name__ == "__main__":
    main()
