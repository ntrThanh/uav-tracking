"""Drop local plugin modules here; each module registers its own backends."""
from __future__ import annotations

import importlib
import pkgutil

_loaded = False


def load_plugins() -> None:
    global _loaded
    if _loaded:
        return
    for module in pkgutil.iter_modules(__path__):
        importlib.import_module(f"{__name__}.{module.name}")
    _loaded = True
