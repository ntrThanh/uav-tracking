"""Pure, testable red-light crossing rule in normalized image coordinates."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.common.config import CameraConfig, Point


def _side(a: Point, b: Point, p: Point) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _in_polygon(p: Point, polygon: list[Point]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > p[1]) != (yj > p[1]) and p[0] < (xj - xi) * (p[1] - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def crossed_stop_line(previous: Point, current: Point, config: CameraConfig) -> bool:
    line = config.stop_line
    if line is None:
        return False
    s0, s1 = _side(line.a, line.b, previous), _side(line.a, line.b, current)
    if abs(s0) < 1e-5 or abs(s1) < 1e-5 or s0 * s1 >= 0:
        return False
    direction = (current[0] - previous[0], current[1] - previous[1])
    axis = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[line.allowed_direction]
    if direction[0] * axis[0] + direction[1] * axis[1] <= 0:
        return False
    # The crossing must happen on the finite segment, not its infinite extension.
    t = s0 / (s0 - s1)
    x = previous[0] + t * direction[0]
    y = previous[1] + t * direction[1]
    dx, dy = line.b[0] - line.a[0], line.b[1] - line.a[1]
    projection = ((x - line.a[0]) * dx + (y - line.a[1]) * dy) / (dx * dx + dy * dy)
    if not 0 <= projection <= 1:
        return False
    if config.zones and not any(_in_polygon(previous, z.polygon) or _in_polygon(current, z.polygon)
                                for z in config.zones):
        return False
    return True


@dataclass
class RedLightRule:
    config: CameraConfig
    last_points: dict[int, Point] = field(default_factory=dict)
    emitted: set[int] = field(default_factory=set)

    def update(self, track_id: int, point: Point, light_state: str) -> bool:
        old = self.last_points.get(track_id)
        self.last_points[track_id] = point
        if old is None or track_id in self.emitted:
            return False
        if crossed_stop_line(old, point, self.config) and light_state == "red":
            self.emitted.add(track_id)
            return True
        return False
