"""Per-session counters and latency summaries; no background sampler."""
from __future__ import annotations

import time
from collections import defaultdict


class Metrics:
    def __init__(self, source_id: str) -> None:
        self.source_id = source_id
        self.started = time.perf_counter()
        self.cpu_started = time.process_time()
        self.frames = 0
        self.errors = 0
        self.dropped = 0
        self.stage_ms: dict[str, list[float]] = defaultdict(list)
        self.runtime: dict[str, dict] = {}

    def record(self, stage: str, milliseconds: float) -> None:
        self.stage_ms[stage].append(milliseconds)
        if len(self.stage_ms[stage]) > 1000:
            self.stage_ms[stage] = self.stage_ms[stage][-1000:]

    def snapshot(self) -> dict:
        elapsed = max(time.perf_counter() - self.started, 1e-9)
        pipeline_ms = self.stage_ms.get("end_to_end", [])
        inference_fps = 1000 * len(pipeline_ms) / sum(pipeline_ms) if pipeline_ms and sum(pipeline_ms) else 0.0
        report = {"source_id": self.source_id, "frames": self.frames, "input_fps": self.frames / elapsed,
                "inference_fps": inference_fps, "errors": self.errors, "dropped_frames": self.dropped,
                "queue_depth": 0, "runtime": self.runtime,
                "process_cpu_seconds": time.process_time() - self.cpu_started,
                "stage_ms_mean": {name: sum(values) / len(values) for name, values in self.stage_ms.items() if values}}
        try:
            import resource
            report["process_max_rss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except ImportError:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                report["cuda_memory_allocated_bytes"] = torch.cuda.memory_allocated()
        except ImportError:
            pass
        return report
