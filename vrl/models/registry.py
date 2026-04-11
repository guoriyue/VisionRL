"""Model registry with lazy imports."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vrl.models.base import VideoGenerationModel

_REGISTRY: dict[str, str] = {
    "wan-official": "vrl.models.families.wan.official:OfficialWanModel",
    "wan-diffusers-i2v": "vrl.models.families.wan.diffusers_i2v:DiffusersWanI2VModel",
    "cosmos": "vrl.models.families.cosmos.model:CosmosGenerationModel",
}


def resolve_model(name: str) -> type[VideoGenerationModel]:
    """Resolve model class by name."""
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
