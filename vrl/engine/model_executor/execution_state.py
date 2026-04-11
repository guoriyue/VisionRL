"""Per-request execution state for iterative video generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vrl.engine.types import VideoExecutionPhase
from vrl.schemas.video_generation import StageResult, VideoGenerationRequest


@dataclass(frozen=True)
class PhaseGroupKey:
    """Batch grouping key: (phase, height, width, frame_count)."""

    phase: VideoExecutionPhase
    height: int
    width: int
    frame_count: int


@dataclass
class DenoiseLoopState:
    """Per-step denoising loop state. Populated by denoise_init(), mutated by denoise_step()."""

    latents: Any = None
    timesteps: Any = None
    current_step: int = 0
    total_steps: int = 0
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
    extra: dict = field(default_factory=dict)


@dataclass
class VideoExecutionState:
    """Per-request state wrapping VideoGenerationRequest with phase tracking."""

    request: VideoGenerationRequest
    phase: VideoExecutionPhase = VideoExecutionPhase.ENCODE_TEXT
    pipeline_state: dict = field(default_factory=dict)
    stage_results: list[StageResult] = field(default_factory=list)
    denoise_state: DenoiseLoopState | None = None
    supports_per_step_denoise: bool = False
