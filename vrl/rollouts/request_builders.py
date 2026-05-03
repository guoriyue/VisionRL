"""GenerationRequest builders for rollout collectors."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from vrl.engine.generation import GenerationRequest


@dataclass(frozen=True, slots=True)
class RolloutRequestPlan:
    """Generation request plus rollout-side metadata."""

    request: GenerationRequest
    reward_metadata: dict[str, Any] = field(default_factory=dict)
    pack_metadata: dict[str, Any] = field(default_factory=dict)


class RolloutRequestBuilder(Protocol):
    """Build one engine request from rollout call arguments."""

    def build(
        self,
        prompts: list[str],
        group_size: int,
        kwargs: dict[str, Any],
    ) -> RolloutRequestPlan: ...


class DiffusionRequestBuilder:
    """Build diffusion GenerationRequest payloads."""

    def __init__(
        self,
        *,
        family: str,
        task: str,
        request_prefix: str,
        config: Any,
        default_task_type: str,
        include_fps: bool = False,
    ) -> None:
        self.family = family
        self.task = task
        self.request_prefix = request_prefix
        self.config = config
        self.default_task_type = default_task_type
        self.include_fps = include_fps

    def build(
        self,
        prompts: list[str],
        group_size: int,
        kwargs: dict[str, Any],
    ) -> RolloutRequestPlan:
        cfg = self.config
        request_overrides = dict(kwargs.get("request_overrides", {}))
        seed = kwargs.get("seed")
        policy_version = kwargs.get("policy_version")

        sampling: dict[str, Any] = {
            "num_steps": cfg.num_steps,
            "guidance_scale": cfg.guidance_scale,
            "height": cfg.height,
            "width": cfg.width,
            "cfg": cfg.cfg,
            "sample_batch_size": cfg.sample_batch_size,
            "sde_window_size": cfg.sde_window_size,
            "sde_window_range": list(cfg.sde_window_range),
            "same_latent": cfg.same_latent,
            "max_sequence_length": cfg.max_sequence_length,
            "return_kl": cfg.kl_reward > 0,
        }
        if hasattr(cfg, "num_frames"):
            sampling["num_frames"] = cfg.num_frames
        if hasattr(cfg, "noise_level"):
            sampling["noise_level"] = cfg.noise_level
        else:
            sampling["noise_level"] = 1.0
        if self.include_fps:
            sampling["fps"] = cfg.fps
        if seed is not None:
            sampling["seed"] = seed
        sampling.update(request_overrides)

        metadata = _shared_reward_metadata(
            kwargs,
            default_task_type=self.default_task_type,
        )
        reference_image = kwargs.get("reference_image")
        if reference_image is not None:
            metadata["reference_image"] = reference_image

        request = GenerationRequest(
            request_id=f"{self.request_prefix}-{uuid.uuid4()}",
            family=self.family,
            task=self.task,
            prompts=prompts,
            samples_per_prompt=group_size,
            sampling=sampling,
            return_artifacts={
                "output",
                "rollout_trajectory_data",
                "trajectory_timesteps",
                "trajectory_latents",
                "denoising_env",
            },
            metadata=metadata,
            policy_version=policy_version,
        )
        return RolloutRequestPlan(
            request=request,
            reward_metadata=metadata,
            pack_metadata=metadata,
        )


class ARRequestBuilder:
    """Build AR GenerationRequest payloads from registry-declared fields."""

    def __init__(
        self,
        *,
        family: str,
        task: str,
        request_prefix: str,
        config: Any,
        sampling_fields: tuple[str, ...],
        return_artifacts: tuple[str, ...],
        metadata_key: str | None = None,
    ) -> None:
        self.family = family
        self.task = task
        self.request_prefix = request_prefix
        self.config = config
        self.sampling_fields = sampling_fields
        self.return_artifacts = return_artifacts
        self.metadata_key = metadata_key

    def build(
        self,
        prompts: list[str],
        group_size: int,
        kwargs: dict[str, Any],
    ) -> RolloutRequestPlan:
        cfg = self.config
        seed = kwargs.get("seed")
        policy_version = kwargs.get("policy_version")
        sampling = _sampling_from_config(cfg, self.sampling_fields)
        if seed is not None:
            sampling["seed"] = seed
        sampling.update(dict(kwargs.get("request_overrides", {})))

        metadata = _ar_reward_metadata(kwargs)
        request_metadata = dict(metadata)
        if self.metadata_key is not None:
            request_metadata = {self.metadata_key: dict(metadata)}
        request = GenerationRequest(
            request_id=f"{self.request_prefix}-{uuid.uuid4()}",
            family=self.family,
            task=self.task,
            prompts=prompts,
            samples_per_prompt=group_size,
            sampling=sampling,
            return_artifacts=set(self.return_artifacts),
            metadata=request_metadata,
            policy_version=policy_version,
        )
        return RolloutRequestPlan(
            request=request,
            reward_metadata=metadata,
            pack_metadata=metadata,
        )


def _shared_reward_metadata(
    kwargs: dict[str, Any],
    *,
    default_task_type: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = dict(kwargs.get("sample_metadata", {}))
    target_text = kwargs.get("target_text", "")
    references = kwargs.get("references", [])
    task_type = kwargs.get("task_type", default_task_type)
    if target_text:
        metadata["target_text"] = target_text
    if references:
        metadata["references"] = references
    metadata["task_type"] = task_type
    return metadata


def _ar_reward_metadata(kwargs: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    target_text = kwargs.get("target_text")
    if target_text:
        metadata["target_text"] = target_text
    references = kwargs.get("references")
    if references:
        metadata["references"] = references
    sample_metadata = kwargs.get("sample_metadata")
    if sample_metadata:
        metadata.update(sample_metadata)
    return metadata


def _sampling_from_config(config: Any, field_names: tuple[str, ...]) -> dict[str, Any]:
    sampling: dict[str, Any] = {}
    for field_name in field_names:
        if not hasattr(config, field_name):
            raise ValueError(
                f"{type(config).__name__} is missing AR sampling field {field_name!r}",
            )
        sampling[field_name] = getattr(config, field_name)
    return sampling


__all__ = [
    "ARRequestBuilder",
    "DiffusionRequestBuilder",
    "RolloutRequestBuilder",
    "RolloutRequestPlan",
]
