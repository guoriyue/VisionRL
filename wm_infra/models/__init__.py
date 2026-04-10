"""Unified model definitions for staged temporal generation."""

from wm_infra.models.base import VideoGenerationModel
from wm_infra.models.registry import list_models, register_model, resolve_model

__all__ = ["VideoGenerationModel", "list_models", "register_model", "resolve_model"]
