"""Deterministic synchronous DAG with validated contracts and lifecycle."""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from .detectors import BaseDetector, build_detector
from .events import EventSink, TrackJsonlSink
from .metrics import Metrics
from .models import ModelRegistry
from .rules import RuleEngine
from .schema import Detection, DetectionResult, Event, Frame, RuleContext, SystemError, TrackingResult
from .tracking import TrackerAdapter
from .traffic_light import TrafficLightClassifier


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    kind: str
    inputs: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)
    when: str = "always"
    on_error: str = "raise"


class Node(Protocol):
    def run(self, frame: Frame, inputs: dict[str, Any]) -> Any: ...
    def close(self) -> None: ...


class PipelineGraph:
    def __init__(self, specs: list[NodeSpec]) -> None:
        if not specs or len({s.node_id for s in specs}) != len(specs):
            raise ValueError("pipeline needs unique nodes")
        self.specs = {s.node_id: s for s in specs}
        kinds = {"detector", "tracker", "rule", "signal", "traffic_light", "sink", "track_sink"}
        for spec in specs:
            if not spec.node_id or spec.kind not in kinds:
                raise ValueError(f"invalid node: {spec.node_id}/{spec.kind}")
            if spec.when not in {"always", "has_tracks", "has_events"} or spec.on_error not in {"raise", "skip"}:
                raise ValueError(f"invalid condition/error policy: {spec.node_id}")
            if any(dep not in self.specs for dep in spec.inputs):
                raise ValueError(f"missing dependency for {spec.node_id}")
            parent_kinds = [self.specs[d].kind for d in spec.inputs]
            if spec.kind == "detector" and spec.inputs and not spec.config.get("track_rois"):
                raise ValueError("detector dependencies require track_rois")
            if spec.kind == "detector" and spec.config.get("track_rois") and parent_kinds != ["tracker"]:
                raise ValueError("track ROI detector needs exactly one tracker input")
            if spec.kind in {"signal", "traffic_light"} and spec.inputs:
                raise ValueError("signal nodes cannot depend on other nodes")
            if spec.kind == "tracker" and parent_kinds != ["detector"]:
                raise ValueError("tracker needs exactly one detector input")
            if spec.kind == "rule" and ("tracker" not in parent_kinds or any(k not in {"tracker", "detector", "signal", "traffic_light"} for k in parent_kinds)):
                raise ValueError("rule needs tracker input and optional detector/signal inputs")
            if spec.kind == "sink" and (not parent_kinds or any(k != "rule" for k in parent_kinds)):
                raise ValueError("sink needs rule inputs")
            if spec.kind == "track_sink" and parent_kinds != ["tracker"]:
                raise ValueError("track sink needs exactly one tracker input")
        ordered: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()
        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ValueError("pipeline graph contains a cycle")
            if node_id in visited:
                return
            visiting.add(node_id)
            for dep in self.specs[node_id].inputs:
                visit(dep)
            visiting.remove(node_id)
            visited.add(node_id)
            ordered.append(node_id)
        for spec in specs:
            visit(spec.node_id)
        self.order = tuple(ordered)


class DetectorNode:
    def __init__(self, detector: BaseDetector, roi: tuple[float, float, float, float] | None = None,
                 track_rois: bool = False) -> None:
        self.detector, self.roi, self.track_rois = detector, roi, track_rois
        if roi and (len(roi) != 4 or min(roi) < 0 or roi[2] <= 0 or roi[3] <= 0
                    or roi[0] + roi[2] > 1 or roi[1] + roi[3] > 1):
            raise ValueError("ROI must be normalized x,y,w,h")

    def run(self, frame: Frame, inputs: dict[str, Any]) -> DetectionResult:
        if self.track_rois:
            tracks = [track for value in inputs.values() if isinstance(value, TrackingResult) for track in value.tracks]
            merged = []
            for track in tracks:
                x0, y0, x1, y1 = (int(v) for v in track.bbox)
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(frame.meta.original_width, x1), min(frame.meta.original_height, y1)
                if x1 <= x0 or y1 <= y0:
                    continue
                result = self._crop(frame, x0, y0, x1, y1)
                merged.extend(replace(d, metadata={**d.metadata, "parent_track_id": track.track_id})
                              for d in result.detections)
            return DetectionResult(frame.meta, tuple(merged), self.detector.model.model_id)
        if not self.roi:
            return self.detector.detect(frame)
        x, y, w, h = self.roi
        x0, y0 = int(x * frame.meta.original_width), int(y * frame.meta.original_height)
        x1, y1 = int((x+w) * frame.meta.original_width), int((y+h) * frame.meta.original_height)
        return self._crop(frame, x0, y0, x1, y1)

    def _crop(self, frame: Frame, x0: int, y0: int, x1: int, y1: int) -> DetectionResult:
        cropped = frame.image[y0:y1, x0:x1]
        if cropped.size == 0:
            raise ValueError("ROI has no pixels")
        sub = Frame(cropped, replace(frame.meta, original_width=x1-x0, original_height=y1-y0))
        result = self.detector.detect(sub)
        translated = tuple(replace(d, bbox=(d.bbox[0]+x0, d.bbox[1]+y0,
                                            d.bbox[2]+x0, d.bbox[3]+y0)) for d in result.detections)
        return replace(result, frame=frame.meta, detections=translated)

    def close(self) -> None:
        self.detector.close()


class TrackerNode:
    def __init__(self, tracker: TrackerAdapter) -> None:
        self.tracker = tracker

    def run(self, frame: Frame, inputs: dict[str, Any]) -> TrackingResult:
        result = next(iter(inputs.values()))
        return self.tracker.update(frame, result)

    def close(self) -> None:
        self.tracker.close()


class SignalNode:
    """Signals injected by the source or caller."""
    def __init__(self, keys: tuple[str, ...]) -> None:
        self.keys = keys

    def run(self, frame: Frame, inputs: dict[str, Any]) -> dict[str, Any]:
        return {key: frame.meta.camera.get(key) for key in self.keys if key in frame.meta.camera}

    def close(self) -> None:
        pass


class TrafficLightNode:
    """Classify a traffic light with HSV in a configured normalized ROI."""
    def __init__(self, roi: tuple[float, float, float, float], history_len: int = 5) -> None:
        if len(roi) != 4 or min(roi) < 0 or roi[2] <= 0 or roi[3] <= 0 or roi[0]+roi[2] > 1 or roi[1]+roi[3] > 1:
            raise ValueError("traffic light ROI must be normalized x,y,w,h")
        self.roi = roi
        self.classifier = TrafficLightClassifier(history_len=history_len)

    def run(self, frame: Frame, inputs: dict[str, Any]) -> dict[str, Any]:
        x, y, w, h = self.roi
        width, height = frame.meta.original_width, frame.meta.original_height
        crop = frame.image[int(y*height):max(int((y+h)*height),int(y*height)+1),
                           int(x*width):max(int((x+w)*width),int(x*width)+1)]
        result = self.classifier.classify_crop(crop)
        return {"light_state": result.state, "light_confidence": result.confidence}

    def close(self) -> None:
        self.classifier.reset()


class RuleNode:
    def __init__(self, rules: RuleEngine, model_versions: dict[str, str] | None = None,
                 pipeline_version: str = "1") -> None:
        self.rules = rules
        self.model_versions = model_versions or {}
        self.pipeline_version = pipeline_version

    def run(self, frame: Frame, inputs: dict[str, Any]) -> tuple[Event, ...]:
        detections: list[Detection] = []
        tracks = []
        signals = {}
        for value in inputs.values():
            if isinstance(value, DetectionResult):
                detections.extend(value.detections)
            elif isinstance(value, TrackingResult):
                tracks.extend(value.tracks)
            elif isinstance(value, dict):
                signals.update(value)
        events = self.rules.evaluate(RuleContext(frame, tuple(detections), tuple(tracks), signals))
        return tuple(replace(event, model_versions=self.model_versions,
                             metadata={**event.metadata, "pipeline_version": self.pipeline_version})
                     for event in events)

    def close(self) -> None:
        self.rules.reset()


class SinkNode:
    def __init__(self, sinks: list[EventSink]) -> None:
        self.sinks = sinks

    def run(self, frame: Frame, inputs: dict[str, Any]) -> tuple[Event, ...]:
        events = tuple(event for value in inputs.values() for event in value)
        for event in events:
            for sink in self.sinks:
                sink.write(event, frame)
        return events

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()


class TrackSinkNode:
    def __init__(self, sink: TrackJsonlSink) -> None:
        self.sink = sink

    def run(self, frame: Frame, inputs: dict[str, Any]) -> TrackingResult:
        result = next(iter(inputs.values()))
        self.sink.write(result)
        return result

    def close(self) -> None:
        self.sink.close()


class PipelineSession:
    """Owns one stream's state. Create a separate session per camera."""
    def __init__(self, graph: PipelineGraph, nodes: dict[str, Node], source_id: str) -> None:
        if set(nodes) != set(graph.specs):
            raise ValueError("node implementations must match graph")
        self.graph, self.nodes = graph, nodes
        self.source_id = source_id
        self.metrics = Metrics(source_id)
        self.closed = False

    def run(self, frame: Frame) -> tuple[dict[str, Any], tuple[SystemError, ...]]:
        if self.closed:
            raise RuntimeError("pipeline is closed")
        if frame.meta.source_id != self.source_id:
            raise ValueError("frame belongs to another stream")
        outputs: dict[str, Any] = {}
        errors = []
        start = time.perf_counter()
        for node_id in self.graph.order:
            spec = self.graph.specs[node_id]
            if any(dep not in outputs for dep in spec.inputs):
                continue
            inputs = {dep: outputs[dep] for dep in spec.inputs}
            if spec.when == "has_tracks" and not any(isinstance(v, TrackingResult) and v.tracks for v in inputs.values()):
                continue
            if spec.when == "has_events" and not any(isinstance(v, tuple) and v and isinstance(v[0], Event) for v in inputs.values()):
                continue
            t0 = time.perf_counter()
            try:
                outputs[node_id] = self.nodes[node_id].run(frame, inputs)
                node = self.nodes[node_id]
                if isinstance(node, DetectorNode) and node.detector.runtime_choice is not None:
                    choice = node.detector.runtime_choice
                    self.metrics.runtime[node_id] = {
                        "model_id": node.detector.model.model_id,
                        "backend": choice.runtime, "device": choice.device,
                        "provider": choice.provider, "attempts": choice.attempts,
                        "fallback_used": choice.fallback_used,
                        "model_load_ms": node.detector.load_ms,
                    }
            except Exception as exc:
                self.metrics.errors += 1
                errors.append(SystemError(self.source_id, node_id, str(exc)))
                if spec.on_error == "raise":
                    raise
            finally:
                self.metrics.record(node_id, (time.perf_counter() - t0) * 1000)
        self.metrics.record("end_to_end", (time.perf_counter() - start) * 1000)
        self.metrics.frames += 1
        return outputs, tuple(errors)

    def close(self) -> None:
        if not self.closed:
            error = None
            for node_id in reversed(self.graph.order):
                try:
                    self.nodes[node_id].close()
                except Exception as exc:
                    if error is None:
                        error = exc
            self.closed = True
            if error is not None:
                raise error

    def __enter__(self) -> "PipelineSession":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
