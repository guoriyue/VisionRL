"""Unified model registry for staged temporal generation models.

All models — Wan, Cosmos, action-conditioned interactive video generators
(Matrix-Game-3), and future temporal models — implement the
``VideoGenerationModel`` contract and register here.

The registry uses lazy imports: model classes are specified as
``module:ClassName`` strings and only imported when resolved.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wm_infra.models.base import VideoGenerationModel

_REGISTRY: dict[str, str] = {
    "wan-official": "wm_infra.models.families.wan.official:OfficialWanModel",
    "wan-diffusers-i2v": "wm_infra.models.families.wan.diffusers_i2v:DiffusersWanI2VModel",
    "cosmos": "wm_infra.models.families.cosmos.model:CosmosGenerationModel",
}


def resolve_model(name: str) -> type[VideoGenerationModel]:
    """Lazily import and return a model class by name."""
    spec = _REGISTRY[name]
    module_path, cls_name = spec.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def register_model(name: str, spec: str) -> None:
    """Register a model spec (module:Class)."""
    _REGISTRY[name] = spec


def list_models() -> list[str]:
    """List all registered model names."""
    return sorted(_REGISTRY.keys())
