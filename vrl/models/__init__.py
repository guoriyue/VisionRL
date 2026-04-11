"""Model definitions and registry."""

from vrl.models.base import VideoGenerationModel
from vrl.models.registry import list_models, register_model, resolve_model

__all__ = ["VideoGenerationModel", "list_models", "register_model", "resolve_model"]
