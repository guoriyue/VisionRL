"""Model definitions and registry."""

from vrl.models.diffusion import DiffusionPolicy, VideoGenerationRequest
from vrl.models.registry import list_models, register_model, resolve_model

__all__ = [
    "DiffusionPolicy",
    "VideoGenerationRequest",
    "list_models",
    "register_model",
    "resolve_model",
]
