"""Dynamic import helpers for Ray actor configuration."""

from __future__ import annotations

import importlib
from typing import Any


def import_from_path(path: str) -> Any:
    """Load ``module:attribute`` or ``module.attribute`` import paths."""

    if ":" in path:
        module_name, attr_name = path.split(":", 1)
    else:
        module_name, _, attr_name = path.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"invalid import path: {path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)
