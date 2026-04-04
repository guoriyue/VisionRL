"""Model registry for world models.

Allows registering and loading different world model implementations
(e.g., COSMOS, Genie, custom) by name.
"""

from __future__ import annotations

from typing import Callable, Type

import torch.nn as nn

from wm_infra.models.base import WorldModel

_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str) -> Callable:
    """Decorator to register a world model class."""
    def wrapper(cls: Type[nn.Module]) -> Type[nn.Module]:
        _REGISTRY[name] = cls
        return cls
    return wrapper


def get_model(name: str) -> Callable[..., nn.Module]:
    """Get a registered model class by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none)"
        raise KeyError(f"Model '{name}' not found. Available: {available}")
    return _REGISTRY[name]


def list_models() -> list[str]:
    """List all registered model names."""
    return sorted(_REGISTRY.keys())


# Register built-in models
from wm_infra.models.dynamics import LatentDynamicsModel  # noqa: E402
register_model("latent_dynamics")(LatentDynamicsModel)
