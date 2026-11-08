"""T2: Faster R-CNN R50-FPN v2 fine-tuned on VisDrone2019-DET (torchvision). Labels 1-10 are VisDrone ids, 0 is
background. The model is built by the training code so that the architecture is the one that was trained."""
from __future__ import annotations

import numpy as np

from .base import Detector


class FasterRCNN(Detector):
    default_weights = "weights/turning/T2_frcnn_r50_v2_det_seed0/best_frcnn.pt"

    def __init__(self, weights=None, device="cuda:0", conf=0.05):
        super().__init__(weights, device, conf)
        import torch
        from torch.torch_version import TorchVersion
        from src.training.frcnn_colab import make_model

        with torch.serialization.safe_globals([TorchVersion]):
            payload = torch.load(self.weights, map_location="cpu", weights_only=True)
        saved = payload.get("config", {})
        self.min_size, self.max_size = int(saved.get("min_size", 800)), int(saved.get("max_size", 1333))
        self.epoch = payload.get("epoch")
        model = make_model({"min_size": self.min_size, "max_size": self.max_size}, pretrained=False)
        model.load_state_dict(payload["model"], strict=True)
        self.model = model.to(device).eval()
        self._torch = torch

    def _predict(self, image: np.ndarray) -> np.ndarray:
        torch = self._torch
        x = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1])).to(self.device)
        x = x.permute(2, 0, 1).float().div_(255)
        with torch.inference_mode():
            out = self.model([x])[0]
        return np.c_[out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(), out["labels"].cpu().numpy()]

    def settings(self) -> dict:
        return {"min_size": self.min_size, "max_size": self.max_size, "nms_iou": 0.5, "max_det": 1000,
                "checkpoint_epoch": self.epoch, "precision": "fp32"}
