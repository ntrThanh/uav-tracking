"""Name -> algorithm lookup.

Adding a tracker means adding one module in algorithms/ with @register("name") and one yaml file. Nothing here or in
factory.py changes, because every module of that package is imported on first use.
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import Callable, TypeVar

from .base import TrackingAlgorithm

A = TypeVar("A", bound=type[TrackingAlgorithm])

_REGISTRY: dict[str, type[TrackingAlgorithm]] = {}
_discovered = False


def register(name: str) -> Callable[[A], A]:
    def add(cls: A) -> A:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"Tracker name already registered: {name}")
        _REGISTRY[name] = cls
        return cls
    return add


def _discover() -> None:
    global _discovered
    if _discovered:
        return
    from . import algorithms
    for module in pkgutil.iter_modules(algorithms.__path__):
        importlib.import_module(f"{algorithms.__name__}.{module.name}")
    _discovered = True


def get_algorithm(name: str) -> type[TrackingAlgorithm]:
    _discover()
    if name not in _REGISTRY:
        raise KeyError(f"Unknown tracker {name!r}; available: {available_trackers()}")
    return _REGISTRY[name]


def available_trackers() -> list[str]:
    _discover()
    return sorted(_REGISTRY)
