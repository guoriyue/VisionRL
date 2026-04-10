"""Cross-layer DTOs for staged temporal generation.

These dataclasses are the shared contract between engine/ and models/.
They carry zero wm_infra dependencies so that both layers can import
them without creating circular edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class VideoGenerationRequest:
    """Model-level request for staged generation.

    The request stays backend-agnostic and uses scalars only. Model-specific
    inputs can be carried in ``extra`` when a generator needs action traces,
    trajectory hints, camera metadata, or other family-specific fields.

    Action-conditioning fields (``action_sequence``, ``action_dim``,
    ``action_conditioning_mode``) support interactive video generators
    such as Matrix-Game-3 that accept per-frame action inputs.
    """

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
    # Action-conditioning for interactive video generators
    action_sequence: list[list[float]] | None = None
    action_dim: int | None = None
    action_conditioning_mode: str = "none"  # "none" | "concat" | "cross_attn" | "adaptive"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StageResult:
    """Normalized stage output used by the shared staged-generation contract.

    The same result shape works for Wan, Cosmos, and action-conditioned
    interactive video generators, even when their execution substrates differ.
    Used by model implementations as the normalized stage output type.
    """

    state_updates: dict[str, Any] = field(default_factory=dict)
    runtime_state_updates: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    cache_hit: bool | None = None
    status: str = "succeeded"
