"""Typed generation runtime payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class GenerationMetrics:
    """Runtime-only generation metrics.

    These metrics describe engine execution. Reward and trainer metrics stay
    outside this module.
    """

    queue_wait_s: float = 0.0
    execution_s: float = 0.0
    peak_memory_mb: float | None = None
    num_prompts: int = 0
    num_samples: int = 0
    num_steps: int | None = None
    micro_batches: int = 0


@dataclass(slots=True)
class GenerationRequest:
    """One generation request submitted to the engine."""

    request_id: str
    family: str
    task: str
    prompts: list[str]
    samples_per_prompt: int
    sampling: dict[str, Any] = field(default_factory=dict)
    return_artifacts: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    policy_version: int | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("GenerationRequest.request_id must be non-empty")
        if not self.family:
            raise ValueError("GenerationRequest.family must be non-empty")
        if not self.task:
            raise ValueError("GenerationRequest.task must be non-empty")
        if not self.prompts:
            raise ValueError("GenerationRequest.prompts must be non-empty")
        if self.samples_per_prompt < 1:
            raise ValueError("GenerationRequest.samples_per_prompt must be >= 1")
        if self.policy_version is not None and self.policy_version < 0:
            raise ValueError("GenerationRequest.policy_version must be >= 0")


@dataclass(slots=True)
class GenerationSampleSpec:
    """Expanded sample-level unit inside a generation request."""

    prompt_index: int
    sample_index: int
    prompt: str
    prompt_id: str
    group_id: str
    sample_id: str
    trajectory_id: str
    seed: int | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WorkloadSignature:
    """Batch grouping key for generation requests."""

    family: str
    task: str
    height: int | None
    width: int | None
    num_frames: int | None
    num_steps: int | None
    artifact_mode: tuple[str, ...]
    max_new_tokens: int | None = None

    @classmethod
    def from_request(cls, request: GenerationRequest) -> WorkloadSignature:
        sampling = request.sampling
        return cls(
            family=request.family,
            task=request.task,
            height=_optional_int(sampling.get("height")),
            width=_optional_int(sampling.get("width")),
            num_frames=_optional_int(sampling.get("num_frames", sampling.get("frame_count"))),
            num_steps=_optional_int(
                sampling.get("num_steps", sampling.get("num_inference_steps"))
            ),
            artifact_mode=tuple(sorted(request.return_artifacts)),
            max_new_tokens=_optional_int(
                sampling.get("max_new_tokens", sampling.get("max_new_image_tokens"))
            ),
        )


@dataclass(slots=True)
class RolloutDebugTensors:
    """Optional debug tensors collected during rollout generation."""

    rollout_variance_noises: Any | None = None
    rollout_prev_sample_means: Any | None = None
    rollout_noise_std_devs: Any | None = None
    rollout_model_outputs: Any | None = None


@dataclass(slots=True)
class RolloutDenoisingEnv:
    """Replay context needed to re-evaluate diffusion rollout logprobs."""

    image_kwargs: dict[str, Any] | None = None
    pos_cond_kwargs: dict[str, Any] | None = None
    neg_cond_kwargs: dict[str, Any] | None = None
    guidance: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RolloutDitTrajectory:
    """Diffusion trajectory data produced by a DiT rollout."""

    latents: Any | None = None
    timesteps: Any | None = None


@dataclass(slots=True)
class RolloutTrajectoryData:
    """SGLang-style post-training rollout trajectory payload."""

    rollout_log_probs: Any | None = None
    rollout_debug_tensors: RolloutDebugTensors | None = None
    denoising_env: RolloutDenoisingEnv | None = None
    dit_trajectory: RolloutDitTrajectory | None = None


@dataclass(slots=True)
class OutputBatch:
    """Engine runtime output batch.

    This is the generation-side output, not the trainer-side RolloutBatch.
    Reward, advantage, and GRPO group semantics stay outside this type.
    """

    request_id: str
    family: str
    task: str
    prompts: list[str]
    sample_specs: list[GenerationSampleSpec]
    output: Any
    trajectory_timesteps: Any | None = None
    trajectory_latents: Any | None = None
    rollout_trajectory_data: RolloutTrajectoryData | None = None
    trajectory_decoded: list[Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    metrics: GenerationMetrics | None = None
    peak_memory_mb: float = 0.0
    error: str | None = None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
