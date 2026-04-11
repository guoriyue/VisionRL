"""Shared DTOs for staged video generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class VideoGenerationRequest:
    """Staged generation request. Backend-agnostic scalars; extras in `extra`."""

    prompt: str = ""
    negative_prompt: str = ""
    references: list[str] = field(default_factory=list)
    task_type: str = "text_to_video"
    width: int = 1024
    height: int = 640
    frame_count: int = 16
    num_steps: int = 35
    guidance_scale: float = 5.0
    high_noise_guidance_scale: float | None = None
    seed: int | None = None
    model_name: str = ""
    model_size: str = "A14B"
    ckpt_dir: str | None = None
    fps: int = 16
    sample_solver: str = "dpmpp"
    shift: float = 1.0
    t5_cpu: bool = True
    convert_model_dtype: bool = True
    offload_model: bool = False
    # Action-conditioning
    action_sequence: list[list[float]] | None = None
    action_dim: int | None = None
    action_conditioning_mode: str = "none"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StageResult:
    """Normalized stage output for all model families."""

    state_updates: dict[str, Any] = field(default_factory=dict)
    runtime_state_updates: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    cache_hit: bool | None = None
    status: str = "succeeded"
