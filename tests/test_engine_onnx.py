"""Optional executable ONNX Runtime smoke test; needs onnx and onnxruntime."""
from __future__ import annotations

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from src.engine.detectors import OnnxDetector  # noqa: E402
from src.engine.models import ModelRegistry  # noqa: E402
from src.engine.schema import Frame, FrameMetadata, ModelMetadata  # noqa: E402


def test_onnx_cpu_load_and_infer(tmp_path):
    from onnx import TensorProto, helper
    input_info = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])
    output_info = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 6])
    boxes = helper.make_tensor("value", TensorProto.FLOAT, [1, 6], [10, 10, 30, 30, .9, 0])
    graph = helper.make_graph([helper.make_node("Constant", [], ["boxes"], value=boxes)],
                              "box_fixture", [input_info], [output_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 8
    path = tmp_path / "constant.onnx"
    onnx.save(model, path)
    registry = ModelRegistry(tmp_path)
    registry.register(ModelMetadata("box", "object_detection", "fixture", path.name, "onnxruntime",
                                    (64, 64), "fp32", {0:"car"}, postprocess={"format":"decoded_nx6"}))
    adapter = OnnxDetector(registry.get("box"), registry, device="cuda", fallback=("cpu",))
    try:
        frame = Frame(np.zeros((64, 64, 3), np.uint8), FrameMetadata("test", 0, 0.0, 64, 64))
        result = adapter.detect(frame)
        assert len(result.detections) == 1
        assert result.detections[0].bbox == (10.0, 10.0, 30.0, 30.0)
        assert adapter.runtime_choice.provider == "CPUExecutionProvider"
        assert adapter.runtime_choice.fallback_used
        assert adapter.runtime_choice.attempts == ("cuda", "cpu")
    finally:
        adapter.close()
