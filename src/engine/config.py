"""JSON pipeline schema and concrete node assembly."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .detectors import build_detector
from .events import EvidenceSink, JsonlSink, TrackJsonlSink
from .graph import DetectorNode, NodeSpec, PipelineGraph, PipelineSession, RuleNode, SignalNode, TrafficLightNode, SinkNode, TrackerNode, TrackSinkNode
from .models import ModelRegistry
from .rules import RuleEngine
from .tracking import TrackerAdapter


def load_config(path: str | Path) -> tuple[dict[str, Any], ModelRegistry, PipelineGraph]:
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "models" not in data or "nodes" not in data:
        raise ValueError("config needs models and nodes")
    models_path = Path(data["models"])
    if not models_path.is_absolute():
        models_path = path.parent / models_path
    registry = ModelRegistry.from_json(models_path)
    specs = [NodeSpec(node_id=item["id"], kind=item["type"],
                      inputs=tuple(item.get("inputs", [])), config=item.get("config", {}),
                      when=item.get("when", "always"), on_error=item.get("on_error", "raise"))
             for item in data["nodes"]]
    graph = PipelineGraph(specs)
    # Validate references before loading a model or opening a stream.
    for spec in specs:
        if spec.kind == "detector":
            registry.get(spec.config["model"])
        if spec.kind == "tracker" and "algorithm" not in spec.config:
            raise ValueError(f"tracker {spec.node_id} needs algorithm")
        if spec.kind == "rule":
            RuleEngine.from_config(spec.config.get("rules", []))
    return data, registry, graph


def build_session(path: str | Path, source_id: str, *, overrides: dict[str, Any] | None = None) -> PipelineSession:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", source_id):
        raise ValueError("source ID must contain only letters, digits, _ or -")
    data, models, graph = load_config(path)
    nodes = {}
    base = Path(path).resolve().parent
    overrides = overrides or {}
    try:
        for node_id in graph.order:
            spec = graph.specs[node_id]
            cfg = spec.config
            if spec.kind == "detector":
                device = str(overrides.get("device", cfg.get("device", "auto")))
                fallback = tuple(cfg.get("fallback", ["cpu"]))
                adapter = build_detector(cfg["model"], models, device, fallback)
                node = DetectorNode(adapter, tuple(cfg["roi"]) if "roi" in cfg else None,
                                    bool(cfg.get("track_rois", False)))
            elif spec.kind == "tracker":
                parent = graph.specs[spec.inputs[0]]
                model = models.get(parent.config["model"])
                path_value = cfg.get("config_path")
                if path_value and not Path(path_value).is_absolute():
                    path_value = str((base / path_value).resolve())
                node = TrackerNode(TrackerAdapter(cfg["algorithm"], model.classes, path_value,
                                                  int(cfg.get("frame_rate", 30)), cfg.get("det_thresh")))
            elif spec.kind == "signal":
                node = SignalNode(tuple(cfg.get("keys", [])))
            elif spec.kind == "traffic_light":
                node = TrafficLightNode(tuple(cfg["roi"]), int(cfg.get("history_len", 5)))
            elif spec.kind == "rule":
                versions = {models.get(item.config["model"]).model_id: models.get(item.config["model"]).version
                            for item in graph.specs.values() if item.kind == "detector"}
                node = RuleNode(RuleEngine.from_config(cfg.get("rules", [])), versions,
                                str(data.get("version", "1")))
            elif spec.kind == "sink":
                sinks = []
                for sink in cfg.get("sinks", []):
                    target = str(sink["path"]).replace("{source_id}", source_id)
                    target_path = Path(target)
                    if not target_path.is_absolute():
                        target_path = base / target_path
                    if sink["type"] == "jsonl":
                        sinks.append(JsonlSink(target_path))
                    elif sink["type"] == "evidence":
                        sinks.append(EvidenceSink(target_path))
                    else:
                        raise ValueError(f"unknown sink: {sink['type']}")
                node = SinkNode(sinks)
            elif spec.kind == "track_sink":
                target = str(cfg["path"]).replace("{source_id}", source_id)
                path_value = Path(target)
                if not path_value.is_absolute():
                    path_value = base / path_value
                node = TrackSinkNode(TrackJsonlSink(path_value))
            nodes[node_id] = node
    except Exception:
        for node in reversed(tuple(nodes.values())):
            node.close()
        raise
    return PipelineSession(graph, nodes, source_id)
