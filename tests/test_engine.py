"""Core engine tests: no weights, GPU, camera or external service required."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.engine.config import build_session, load_config
from src.engine.detectors import BaseDetector, LegacyTorchDetector, OnnxDetector
from src.engine.graph import NodeSpec, PipelineGraph, PipelineSession, RuleNode, SignalNode, TrackerNode, SinkNode
from src.engine.events import JsonlSink
from src.engine.models import ModelRegistry
from src.engine.rules import RuleEngine
from src.engine.runtime import UnsupportedBackendError, choose_pytorch, device_candidates, require_supported
from src.engine.schema import (Detection, DetectionResult, Event, Frame, FrameMetadata, ModelMetadata, Track,
                               TrackingResult, detections_from_numpy, detections_to_numpy, tracks_from_numpy)
from src.engine.tracking import TrackerAdapter


def frame(source="cam1", idx=0, t=0.0):
    return Frame(np.zeros((100, 100, 3), np.uint8), FrameMetadata(source, idx, t, 100, 100))


def model(**overrides):
    data = dict(model_id="test", task="object_detection", architecture="yolo11m", weights="weights.pt",
                runtime="pytorch", input_size=(64, 64), precision="fp32", classes={4: "car"})
    data.update(overrides)
    return ModelMetadata(**data)


def test_schema_numpy_contract_and_class_mapping():
    meta = frame().meta
    result = detections_from_numpy(np.array([[1, 2, 21, 22, .8, 4]]), meta, "det", {4:"car"})
    assert result.detections[0].class_name == "car"
    assert detections_to_numpy(result).shape == (1, 6)
    assert tracks_from_numpy(np.array([[1,2,21,22,9,.8,4]]), meta, "sort", {4:"car"}).tracks[0].track_id == 9
    with pytest.raises(ValueError):
        detections_from_numpy(np.array([[1,2,21,22,.8,3]]), meta, "det", {4:"car"})
    with pytest.raises(ValueError):
        Detection((5, 5, 1, 2), .8, 4, "car", "det")


def test_model_registry_validation_and_checksum(tmp_path):
    weight = tmp_path / "weights.pt"
    weight.write_bytes(b"test")
    registry = ModelRegistry(tmp_path)
    registry.register(model())
    assert registry.path("test") == weight
    with pytest.raises(ValueError):
        registry.register(model())
    bad = ModelRegistry(tmp_path)
    bad.register(model(checksum="0" * 64))
    with pytest.raises(ValueError, match="checksum"):
        bad.path("test")


class FakeDetector(BaseDetector):
    def load(self):
        self.loaded = True
    def predict(self, image):
        return np.array([[10, 10, 30, 30, .9, 4]])
    def postprocess(self, rows, current):
        return detections_from_numpy(rows, current.meta, self.model.model_id, self.model.classes)


def test_detector_adapter_lifecycle_and_roi(tmp_path):
    registry = ModelRegistry(tmp_path)
    registry.register(model())
    detector = FakeDetector(registry.get("test"), registry)
    result = detector.detect(frame())
    assert result.detections[0].bbox == (10,10,30,30)
    assert result.latency_ms >= 0
    from src.engine.graph import DetectorNode
    output = DetectorNode(detector, (.2,.2,.5,.5)).run(frame(), {})
    assert output.detections[0].bbox == (30,30,50,50)
    detector.close()
    assert not detector.loaded


def test_rfdetr_legacy_adapter_maps_to_shared_schema(tmp_path, monkeypatch):
    (tmp_path / "weights.pt").write_bytes(b"fixture")
    registry = ModelRegistry(tmp_path)
    registry.register(model(architecture="rfdetr_m", classes={1:"pedestrian"}))
    class FakeLegacy:
        def predict(self, image):
            return np.array([[2, 3, 20, 30, .75, 1]])
    monkeypatch.setattr("src.detectors.build_detector", lambda *args, **kwargs: FakeLegacy())
    detector = LegacyTorchDetector(registry.get("test"), registry, device="cpu", fallback=())
    result = detector.detect(frame())
    assert result.detections[0].class_name == "pedestrian"
    assert detector.runtime_choice.device == "cpu"


def test_yolo_onnx_output_decoding_without_runtime(tmp_path):
    registry = ModelRegistry(tmp_path)
    registry.register(model(runtime="onnxruntime", architecture="ultralytics_yolo", classes={0:"car"},
                            postprocess={"format":"ultralytics_yolo","conf":.25,"iou":.5}))
    detector = OnnxDetector(registry.get("test"), registry)
    detector._transform = (1.0,0,0)
    output = np.array([[[20,20],[20,20],[10,10],[10,10],[.9,.8]]],dtype=np.float32)
    detections = detector.postprocess([output], frame())
    assert len(detections.detections) == 1
    assert detections.detections[0].bbox == (15,15,25,25)


def test_device_policy_is_explicit(monkeypatch):
    assert device_candidates("auto", ("cpu",)) == ("cuda", "cpu")
    monkeypatch.setattr("src.engine.runtime.discover", lambda: {"pytorch":True,"cuda":False})
    selected = choose_pytorch("auto", ("cpu",))
    assert selected.device == "cpu" and selected.fallback_used and selected.attempts == ("cuda", "cpu")
    with pytest.raises(UnsupportedBackendError):
        require_supported("qnn")


def test_tracker_factory_and_reset():
    tracker = TrackerAdapter("sort", {4:"car"}, det_thresh=.1)
    f = frame()
    detections = detections_from_numpy(np.array([[10,10,30,30,.9,4]]), f.meta, "det", {4:"car"})
    assert tracker.update(f, detections).frame == f.meta
    old = tracker._tracker
    tracker.reset()
    assert tracker._tracker is not old
    tracker.close()


def test_rules_deduplicate_and_unknown_light():
    rules = RuleEngine.from_config([{"type":"red_light","rule_id":"red","a":[.2,.5],"b":[.8,.5],
                                     "direction":"down","event_type":"red_light_candidate","cooldown_s":10}])
    def context(y, idx, light):
        f = frame(idx=idx,t=float(idx))
        tr = Track(1,(40,y-10,60,y),.9,4,"car","active",float(idx))
        from src.engine.schema import RuleContext
        return RuleContext(f, tracks=(tr,), signals={"light_state":light})
    assert not rules.evaluate(context(40,0,"red"))
    assert not rules.evaluate(context(60,1,"unknown"))
    assert not rules.evaluate(context(40,2,"red"))
    assert len(rules.evaluate(context(60,3,"red"))) == 1
    assert not rules.evaluate(context(40,4,"red"))
    assert not rules.evaluate(context(60,5,"red"))  # cooldown/dedup within window


def test_graph_dependency_cycle_and_order():
    specs = [NodeSpec("signals","signal"),NodeSpec("det","detector"),
             NodeSpec("track","tracker",("det",)),NodeSpec("rules","rule",("track","signals"))]
    assert PipelineGraph(specs).order == ("signals","det","track","rules")
    with pytest.raises(ValueError, match="cycle"):
        PipelineGraph([NodeSpec("a","detector",("b",),{"track_rois":True}),
                       NodeSpec("b","tracker",("a",))])
    with pytest.raises(ValueError, match="missing dependency"):
        PipelineGraph([NodeSpec("a","tracker",("missing",))])


def test_two_streams_have_independent_state(tmp_path):
    registry = ModelRegistry(tmp_path)
    registry.register(model())
    from src.engine.graph import DetectorNode
    graph = PipelineGraph([NodeSpec("det","detector"),NodeSpec("track","tracker",("det",)),
                           NodeSpec("rules","rule",("track",))])
    def session(source):
        return PipelineSession(graph, {"det":DetectorNode(FakeDetector(registry.get("test"),registry)),
            "track":TrackerNode(TrackerAdapter("sort",{4:"car"},det_thresh=.1)),
            "rules":RuleNode(RuleEngine.from_config([{"type":"object_count","rule_id":"count",
                    "a":[.2,.5],"b":[.8,.5],"direction":"down"}]))}, source)
    one, two = session("one"), session("two")
    try:
        assert one.nodes["track"].tracker is not two.nodes["track"].tracker
        a,_ = one.run(frame("one"))
        b,_ = two.run(frame("two"))
        assert a["track"].frame.source_id == "one"
        assert b["track"].frame.source_id == "two"
        assert one.metrics.frames == two.metrics.frames == 1
    finally:
        one.close();two.close()


def test_example_configs_validate_without_loading_models():
    root = Path(__file__).resolve().parents[1]
    for name in ("uav_tracking", "traffic_monitoring"):
        data, models, graph = load_config(root / f"src/engine/examples/{name}.json")
        assert graph.order and models.get("visdrone_yolo11m")


def test_end_to_end_synthetic_dag_writes_event(tmp_path):
    graph = PipelineGraph([NodeSpec("det","detector"), NodeSpec("track","tracker",("det",)),
                           NodeSpec("rule","rule",("track",)), NodeSpec("sink","sink",("rule",),
                                                                          when="has_events")])
    class DetectionNode:
        def run(self, frame, inputs):
            return DetectionResult(frame.meta, (), "fake")
        def close(self):pass
    class MovingTrackNode:
        def run(self, frame, inputs):
            y = 40 if frame.meta.frame_id == 0 else 60
            track = Track(1,(40,y-10,60,y),.9,4,"car","active",frame.meta.timestamp)
            return TrackingResult(frame.meta,(track,),"fake")
        def close(self):pass
    path = tmp_path / "events.jsonl"
    rules = RuleEngine.from_config([{"type":"line_crossing","rule_id":"line","a":[.2,.5],
                                     "b":[.8,.5],"direction":"down"}])
    with PipelineSession(graph,{"det":DetectionNode(),"track":MovingTrackNode(),
                                "rule":RuleNode(rules),"sink":SinkNode([JsonlSink(path)])},"cam") as pipeline:
        first,_ = pipeline.run(frame("cam",0,0.0))
        second,_ = pipeline.run(frame("cam",1,1.0))
        assert first["rule"] == ()
        assert len(second["rule"]) == 1
        assert pipeline.metrics.frames == 2
    assert '"event_type": "line_crossing"' in path.read_text()
