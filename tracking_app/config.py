"""Validated, editable runtime settings. Model definitions stay in the engine catalog."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from tracking_app.engine.models import ModelRegistry

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "tracking_app/models.json"
DATA_DIR = ROOT / "outputs/tracking_app"
SETTINGS = DATA_DIR / "settings.json"
OCR_MODELS = ROOT / "tracking_app/models/ocr"


@dataclass(frozen=True)
class Settings:
    vehicle_backend: str = "engine"
    vehicle_model: str = "coco_yolo11n_onnx"
    vehicle_classes: str = "bicycle,car,motorcycle,bus,truck"
    plate_backend: str = "engine"
    plate_model: str = "traffic_plate_yolo11s_onnx"
    tracker_backend: str = "engine"
    tracker: str = "bytetrack"
    device: str = "auto"
    confidence: float = 0.35
    plate_confidence: float = 0.25
    iou: float = 0.6
    plate_iou: float = 0.6
    track_threshold: float = 0.35
    frame_rate: int = 30
    stride: int = 1
    plate_interval: int = 5
    ocr_backend: str = "paddle_onnx_direct"
    ocr_model_name: str = "latin_PP-OCRv5_mobile_rec"
    ocr_detection_model_name: str = "PP-OCRv5_mobile_det"
    ocr_model_dir: str = str(OCR_MODELS / "latin_PP-OCRv5_mobile_rec")
    ocr_detection_dir: str = str(OCR_MODELS / "PP-OCRv5_mobile_det")
    ocr_confidence: float = 0.2
    ocr_upscale: float = 2.0
    ocr_preprocess_mode: str = "auto"
    ocr_clahe_clip: float = 2.0
    ocr_sharpen: float = 0.0
    ocr_text_threshold: float = 0.3
    ocr_box_threshold: float = 0.6
    ocr_unclip_ratio: float = 1.5
    preview_width: int = 960

    @classmethod
    def from_dict(cls, raw: dict) -> "Settings":
        from .plugins import load_plugins
        from tracking_app.engine.trackers import available_trackers
        from .vehicle import available_backends as vehicle_backends
        from .plate import available_backends as plate_backends
        from .tracking import available_backends as tracker_backends
        from .ocr import available_backends as ocr_backends
        from .ocr.preprocess import available_preprocessors

        if not isinstance(raw, dict) or set(raw) - set(cls.__dataclass_fields__):
            raise ValueError("Settings contain an invalid field")
        load_plugins()
        values = {**asdict(cls()), **raw}
        registry = ModelRegistry.from_json(CATALOG)
        for backend_key, model_key, expected in (("vehicle_backend", "vehicle_model", "object_detection"),
                                                 ("plate_backend", "plate_model", "license_plate_detection")):
            if values[backend_key] == "engine":
                model = registry.get(values[model_key])
                if model.task != expected or model.runtime != "onnxruntime" or not model.weights.endswith(".onnx"):
                    raise ValueError(f"{model_key} must be an ONNX model for the selected task")
        if values["tracker_backend"] == "engine" and values["tracker"] not in available_trackers():
            raise ValueError("Unsupported tracker")
        for key, available in (("vehicle_backend", vehicle_backends()),
                               ("plate_backend", plate_backends()),
                               ("tracker_backend", tracker_backends()),
                               ("ocr_backend", ocr_backends())):
            if values[key] not in available:
                raise ValueError(f"Unsupported {key}")
        if values["device"] not in ("auto", "cpu", "cuda"):
            raise ValueError("Invalid compute device")
        values["vehicle_classes"] = str(values["vehicle_classes"]).strip()
        if not values["vehicle_classes"]:
            raise ValueError("Select at least one vehicle class")
        if values["vehicle_backend"] == "engine":
            selected = {name.strip().lower() for name in values["vehicle_classes"].split(",") if name.strip()}
            known = {name.lower() for name in registry.get(values["vehicle_model"]).classes.values()}
            unknown = selected - known
            if unknown:
                raise ValueError(f"Vehicle classes not found in the model: {', '.join(sorted(unknown))}")
        values["ocr_model_name"] = str(values["ocr_model_name"]).strip()
        values["ocr_detection_model_name"] = str(values["ocr_detection_model_name"]).strip()
        values["ocr_model_dir"] = str(values["ocr_model_dir"]).strip()
        values["ocr_detection_dir"] = str(values["ocr_detection_dir"]).strip()
        if values["ocr_backend"] in ("paddleocr_onnx", "paddle_onnx_direct") and not all(
            (values["ocr_model_name"], values["ocr_detection_model_name"])):
            raise ValueError("PaddleOCR text detection and recognition model names are required")
        for key in ("ocr_model_dir", "ocr_detection_dir"):
            if values[key] and not Path(values[key]).is_dir():
                raise ValueError(f"Directory not found: {key}")
            if values["ocr_backend"] in ("paddleocr_onnx", "paddle_onnx_direct") and values[key]:
                for filename in ("inference.onnx", "inference.yml"):
                    if not (Path(values[key]) / filename).is_file():
                        raise ValueError(f"Missing {filename} in {key}")
        for key in ("confidence", "plate_confidence", "iou", "plate_iou",
                    "track_threshold", "ocr_confidence"):
            values[key] = float(values[key])
            if not 0 < values[key] <= 1:
                raise ValueError(f"{key} must be in the range (0, 1]")
        if values["ocr_preprocess_mode"] not in available_preprocessors():
            raise ValueError("Unsupported OCR preprocessing mode")
        for key, minimum, maximum in (("ocr_upscale", 1.0, 4.0),
                                       ("ocr_clahe_clip", 1.0, 8.0),
                                       ("ocr_sharpen", 0.0, 2.0),
                                       ("ocr_unclip_ratio", 1.0, 3.0)):
            values[key] = float(values[key])
            if not minimum <= values[key] <= maximum:
                raise ValueError(f"{key} must be between {minimum} and {maximum}")
        for key in ("ocr_text_threshold", "ocr_box_threshold"):
            values[key] = float(values[key])
            if not 0 < values[key] <= 1:
                raise ValueError(f"{key} must be in the range (0, 1]")
        for key in ("stride", "plate_interval", "preview_width", "frame_rate"):
            values[key] = int(values[key])
            if values[key] < 1:
                raise ValueError(f"{key} must be greater than 0")
        if values["preview_width"] > 1920:
            raise ValueError("preview_width must be at most 1920")
        return cls(**values)


def load_settings() -> Settings:
    if not SETTINGS.exists():
        return Settings.from_dict({})
    values = json.loads(SETTINGS.read_text(encoding="utf-8"))
    if "plate_iou" not in values:
        values["plate_iou"] = values.get("iou", Settings.plate_iou)
    # Settings saved by the previous desktop version used PyTorch IDs/EasyOCR.
    migration = {
        "vehicle_model": {"visdrone_yolo11m": "visdrone_yolo11m_onnx"},
        "plate_model": {"traffic_plate_yolo11s": "traffic_plate_yolo11s_onnx"},
        "ocr_backend": {"easyocr": "paddle_onnx_direct", "paddleocr_onnx": "paddle_onnx_direct"},
    }
    for key, mapping in migration.items():
        values[key] = mapping.get(values.get(key), values.get(key, getattr(Settings, key)))
    if values.get("ocr_model_name") == "en_PP-OCRv5_mobile_rec":
        values["ocr_model_name"] = "latin_PP-OCRv5_mobile_rec"
    for key in ("ocr_model_dir", "ocr_detection_dir"):
        if not values.get(key):
            values[key] = getattr(Settings, key)
    return Settings.from_dict(values)


def save_settings(settings: Settings) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary = SETTINGS.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(settings), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(SETTINGS)
