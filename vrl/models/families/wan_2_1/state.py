"""Wan-specific execution state. Opaque to the engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WanDenoiseState:
    """Per-step denoising state for Wan 2.2 models.

    Stored as ``DenoiseLoopState.model_state``. The engine never
    inspects these fields — only ``denoise_step`` / ``denoise_finalize``
    on ``OfficialWanModel`` read and mutate them.
    """

    latents: Any = None
    timesteps: Any = None
    seed_generator: Any = None
    arg_c: dict = field(default_factory=dict)
    arg_null: dict = field(default_factory=dict)
    boundary: float = 0.0
    high_noise_guidance_scale: float = 5.0
    low_noise_guidance_scale: float = 5.0
    scheduler: Any = None
    seed: int = 0
    seed_policy: str = "randomized"
    solver_name: str = "dpmpp"
    pipeline: Any = None
    task_key: str = ""
