"""Model registry with lazy imports."""

from __future__ import annotations

import importlib

_REGISTRY: dict[str, str] = {
    "sd3_5-diffusers-t2i": "vrl.models.families.sd3_5.policy:SD3_5Policy",
    "wan-diffusers-t2v": "vrl.models.families.wan_2_1.diffusers_policy:WanT2VDiffusersPolicy",
    "wan-official-t2v": "vrl.models.families.wan_2_1.official_policy:WanT2VOfficialPolicy",
    "cosmos-predict2": "vrl.models.families.cosmos.policy:CosmosPredict2Policy",
}


def resolve_model(name: str) -> type:
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
