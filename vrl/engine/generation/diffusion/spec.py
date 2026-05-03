"""Typed sampling specs shared by diffusion generation executors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BaseDiffusionGenerationSpec:
    """Common request fields every diffusion executor needs."""

    num_steps: int
    guidance_scale: float
    height: int
    width: int
    num_frames: int
    fps: int | None
    sample_batch_size: int
    max_sequence_length: int
    seed: int | None
    negative_prompt: str | None


@dataclass(frozen=True, slots=True)
class SDEDiffusionSpec:
    """SDE rollout knobs for diffusion executors that collect logprobs."""

    noise_level: float
    sde_window_size: int
    sde_window_range: tuple[int, int]
    same_latent: bool
    return_kl: bool


@dataclass(frozen=True, slots=True)
class DiffusionGenerationSpec:
    """Parsed generation spec for one diffusion request."""

    base: BaseDiffusionGenerationSpec
    sde: SDEDiffusionSpec | None


__all__ = [
    "BaseDiffusionGenerationSpec",
    "DiffusionGenerationSpec",
    "SDEDiffusionSpec",
]
