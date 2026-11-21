"""Stateful, model-independent rule plugins; one instance belongs to one stream."""
from __future__ import annotations

import math
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable

from .schema import Event, RuleContext, Track

Point = tuple[float, float]


def _point(track: Track, context: RuleContext) -> Point:
    x1, _, x2, y2 = track.bbox
    return ((x1 + x2) / (2 * context.frame.meta.original_width), y2 / context.frame.meta.original_height)


def _inside(point: Point, polygon: list[Point]) -> bool:
    hit = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > point[1]) != (yj > point[1]) and point[0] < (xj-xi) * (point[1]-yi) / (yj-yi) + xi:
            hit = not hit
        j = i
    return hit


def _crossed(a: Point, b: Point, previous: Point, current: Point) -> bool:
    def side(p: Point) -> float:
        return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])
    s0, s1 = side(previous), side(current)
    if abs(s0) < 1e-6 or abs(s1) < 1e-6 or s0*s1 >= 0:
        return False
    fraction = s0/(s0-s1)
    intersection = (previous[0]+fraction*(current[0]-previous[0]),
                    previous[1]+fraction*(current[1]-previous[1]))
    dx, dy = b[0]-a[0], b[1]-a[1]
    projection = ((intersection[0]-a[0])*dx+(intersection[1]-a[1])*dy)/(dx*dx+dy*dy)
    return 0 <= projection <= 1


def _event(rule: "BaseRule", context: RuleContext, track: Track, event_type: str,
           confidence: float | None = None, metadata: dict[str, Any] | None = None) -> Event:
    return Event(uuid.uuid4().hex, event_type, context.frame.meta.source_id, rule.rule_id,
                 context.frame.meta.timestamp, track_id=track.track_id, confidence=confidence,
                 metadata=metadata or {})


class BaseRule(ABC):
    def __init__(self, rule_id: str, cooldown_s: float = 2.0) -> None:
        self.rule_id = rule_id
        self.cooldown_s = cooldown_s
        self._last_emitted: dict[int, float] = {}
        self._seen: dict[int, float] = {}
        self.last_outcome = "ready"

    def _allow(self, track_id: int, timestamp: float) -> bool:
        last = self._last_emitted.get(track_id, -math.inf)
        if timestamp - last < self.cooldown_s:
            return False
        self._last_emitted[track_id] = timestamp
        return True

    @abstractmethod
    def evaluate(self, context: RuleContext) -> list[Event]: ...

    def reset(self) -> None:
        self._last_emitted.clear()
        self._seen.clear()
        self.last_outcome = "ready"

    def prune(self, context: RuleContext, timeout_s: float = 60.0) -> None:
        now = context.frame.meta.timestamp
        for track in context.tracks:
            self._seen[track.track_id] = now
        expired = [track_id for track_id, seen in self._seen.items() if now - seen > timeout_s]
        for track_id in expired:
            self._seen.pop(track_id, None)
            self._last_emitted.pop(track_id, None)
            for attr in ("_last_points", "_inside", "_last", "_anchor"):
                state = getattr(self, attr, None)
                if state is not None:
                    state.pop(track_id, None)


class LineCrossingRule(BaseRule):
    def __init__(self, rule_id: str, a: Point, b: Point, direction: str = "any",
                 event_type: str = "line_crossing", min_move: float = 0.002,
                 cooldown_s: float = 2.0) -> None:
        super().__init__(rule_id, cooldown_s)
        self.a, self.b = tuple(a), tuple(b)
        if self.a == self.b or any(not 0 <= v <= 1 for v in (*self.a, *self.b)):
            raise ValueError("line endpoints must differ and lie within image")
        if direction not in {"any", "up", "down", "left", "right"}:
            raise ValueError("invalid crossing direction")
        self.direction, self.event_type, self.min_move = direction, event_type, min_move
        self._last_points: dict[int, Point] = {}

    def _moving_right_way(self, old: Point, new: Point) -> bool:
        vector = {"up": (0,-1), "down": (0,1), "left": (-1,0), "right": (1,0)}.get(self.direction)
        return vector is None or (new[0]-old[0])*vector[0]+(new[1]-old[1])*vector[1] > 0

    def evaluate(self, context: RuleContext) -> list[Event]:
        events = []
        for track in context.tracks:
            point = _point(track, context)
            old = self._last_points.get(track.track_id)
            self._last_points[track.track_id] = point
            if old is None or math.dist(old, point) < self.min_move:
                continue
            if _crossed(self.a, self.b, old, point) and self._moving_right_way(old, point):
                if self._allow(track.track_id, context.frame.meta.timestamp):
                    events.append(_event(self, context, track, self.event_type, track.confidence))
        return events

    def reset(self) -> None:
        super().reset()
        self._last_points.clear()


class RedLightRule(LineCrossingRule):
    def evaluate(self, context: RuleContext) -> list[Event]:
        light = context.signals.get("light_state", "unknown")
        confidence = context.signals.get("light_confidence")
        if light == "unknown" or (confidence is not None and confidence < 0.5):
            # Still update positions so an unknown light cannot generate a delayed event.
            for track in context.tracks:
                self._last_points[track.track_id] = _point(track, context)
            self.last_outcome = "insufficient_light_data"
            return []
        if light != "red":
            for track in context.tracks:
                self._last_points[track.track_id] = _point(track, context)
            self.last_outcome = "light_not_red"
            return []
        self.last_outcome = "ready"
        return super().evaluate(context)


class RegionRule(BaseRule):
    def __init__(self, rule_id: str, polygon: list[Point], mode: str = "entry",
                 cooldown_s: float = 2.0) -> None:
        super().__init__(rule_id, cooldown_s)
        if len(polygon) < 3 or mode not in {"entry", "exit"}:
            raise ValueError("region requires polygon and entry/exit mode")
        self.polygon, self.mode = polygon, mode
        self._inside: dict[int, bool] = {}

    def evaluate(self, context: RuleContext) -> list[Event]:
        events = []
        for track in context.tracks:
            current = _inside(_point(track, context), self.polygon)
            old = self._inside.get(track.track_id)
            self._inside[track.track_id] = current
            changed = old is not None and old != current
            if changed and current == (self.mode == "entry") and self._allow(track.track_id, context.frame.meta.timestamp):
                events.append(_event(self, context, track, f"region_{self.mode}", track.confidence))
        return events

    def reset(self) -> None:
        super().reset()
        self._inside.clear()


class WrongWayRule(BaseRule):
    def __init__(self, rule_id: str, allowed_direction: str, min_move: float = 0.02,
                 cooldown_s: float = 10.0) -> None:
        super().__init__(rule_id, cooldown_s)
        if allowed_direction not in {"up", "down", "left", "right"}:
            raise ValueError("invalid allowed direction")
        self.allowed_direction, self.min_move = allowed_direction, min_move
        self._last: dict[int, Point] = {}

    def evaluate(self, context: RuleContext) -> list[Event]:
        axis = {"up":(0,-1),"down":(0,1),"left":(-1,0),"right":(1,0)}[self.allowed_direction]
        events = []
        for track in context.tracks:
            point = _point(track, context)
            old = self._last.get(track.track_id)
            if old is None:
                self._last[track.track_id] = point
                continue
            delta = (point[0]-old[0], point[1]-old[1])
            if math.dist(old, point) >= self.min_move:
                self._last[track.track_id] = point
                if delta[0]*axis[0]+delta[1]*axis[1] < -self.min_move/2 and self._allow(track.track_id, context.frame.meta.timestamp):
                    events.append(_event(self, context, track, "wrong_way", track.confidence))
        return events

    def reset(self) -> None:
        super().reset()
        self._last.clear()


class StoppedObjectRule(BaseRule):
    def __init__(self, rule_id: str, duration_s: float = 5.0, radius: float = 0.01,
                 cooldown_s: float = 30.0) -> None:
        super().__init__(rule_id, cooldown_s)
        self.duration_s, self.radius = duration_s, radius
        self._anchor: dict[int, tuple[Point, float]] = {}

    def evaluate(self, context: RuleContext) -> list[Event]:
        events = []
        for track in context.tracks:
            point = _point(track, context)
            anchor, since = self._anchor.get(track.track_id, (point, context.frame.meta.timestamp))
            if math.dist(point, anchor) > self.radius:
                anchor, since = point, context.frame.meta.timestamp
            self._anchor[track.track_id] = (anchor, since)
            if context.frame.meta.timestamp - since >= self.duration_s and self._allow(track.track_id, context.frame.meta.timestamp):
                events.append(_event(self, context, track, "stopped_object", track.confidence))
        return events

    def reset(self) -> None:
        super().reset()
        self._anchor.clear()


class CountingRule(LineCrossingRule):
    def __init__(self, rule_id: str, a: Point, b: Point, direction: str = "any") -> None:
        super().__init__(rule_id, a, b, direction, event_type="object_count", cooldown_s=math.inf)
        self.count = 0

    def evaluate(self, context: RuleContext) -> list[Event]:
        events = super().evaluate(context)
        self.count += len(events)
        return events

    def reset(self) -> None:
        super().reset()
        self.count = 0


RuleFactory = Callable[..., BaseRule]
RULES: dict[str, RuleFactory] = {
    "line_crossing": LineCrossingRule,
    "red_light": RedLightRule,
    "region_entry": lambda **kw: RegionRule(mode="entry", **kw),
    "region_exit": lambda **kw: RegionRule(mode="exit", **kw),
    "object_count": CountingRule,
    "wrong_way": WrongWayRule,
    "stopped_object": StoppedObjectRule,
}


def register_rule(name: str, factory: RuleFactory) -> None:
    if name in RULES:
        raise ValueError(f"rule already registered: {name}")
    RULES[name] = factory


class RuleEngine:
    def __init__(self, rules: list[BaseRule]) -> None:
        self.rules = rules

    @classmethod
    def from_config(cls, configs: list[dict[str, Any]]) -> "RuleEngine":
        rules = []
        for item in configs:
            if not item.get("enabled", True):
                continue
            name = item["type"]
            if name not in RULES:
                raise ValueError(f"unknown rule: {name}")
            parameters = {k:v for k,v in item.items() if k not in {"type", "enabled"}}
            rules.append(RULES[name](**parameters))
        return cls(rules)

    def evaluate(self, context: RuleContext) -> tuple[Event, ...]:
        events = []
        for rule in self.rules:
            rule.prune(context)
            events.extend(rule.evaluate(context))
        return tuple(events)

    def reset(self) -> None:
        for rule in self.rules:
            rule.reset()
