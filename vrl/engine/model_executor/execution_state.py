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
    """Per-step denoising progress. Model-agnostic.

    The engine only reads ``current_step`` and ``total_steps`` to decide
    when to transition from DENOISE_STEP to DENOISE_FINALIZE.

    All model-specific state (latents, scheduler, guidance args, etc.)
    lives in ``model_state`` — an opaque object created by
    ``denoise_init()`` and consumed by ``denoise_step()`` /
    ``denoise_finalize()``. The engine never inspects it.
    """

    current_step: int = 0
    total_steps: int = 0
    model_state: Any = None


@dataclass
class VideoExecutionState:
    """Per-request state wrapping VideoGenerationRequest with phase tracking."""

    request: VideoGenerationRequest
    phase: VideoExecutionPhase = VideoExecutionPhase.ENCODE_TEXT
    pipeline_state: dict = field(default_factory=dict)
    stage_results: list[StageResult] = field(default_factory=list)
    denoise_state: DenoiseLoopState | None = None
    supports_per_step_denoise: bool = False
