"""Rollout-to-engine request adapter for collectors."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from vrl.engine import GenerationRequest


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


class RolloutEngineRequestBuilder:
    """Build ``GenerationRequest`` payloads from registry-declared fields."""

    def __init__(
        self,
        *,
        family: str,
        task: str,
        request_prefix: str,
        config: Any,
        sampling_fields: tuple[str, ...],
        return_artifacts: tuple[str, ...],
        default_task_type: str | None = None,
        metadata_key: str | None = None,
    ) -> None:
        if not sampling_fields:
            raise ValueError(f"{family} request builder requires sampling_fields")
        if not return_artifacts:
            raise ValueError(f"{family} request builder requires return_artifacts")
        self.family = family
        self.task = task
        self.request_prefix = request_prefix
        self.config = config
        self.sampling_fields = sampling_fields
        self.return_artifacts = return_artifacts
        self.default_task_type = default_task_type
        self.metadata_key = metadata_key

    def build(
        self,
        prompts: list[str],
        group_size: int,
        kwargs: dict[str, Any],
    ) -> RolloutRequestPlan:
        seed = kwargs.get("seed")
        policy_version = kwargs.get("policy_version")
        sampling = _sampling_from_config(self.config, self.sampling_fields)
        if seed is not None:
            sampling["seed"] = seed
        sampling.update(dict(kwargs.get("request_overrides", {})))

        metadata = _rollout_metadata(
            kwargs,
            default_task_type=self.default_task_type,
        )
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


def _rollout_metadata(
    kwargs: dict[str, Any],
    *,
    default_task_type: str | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    sample_metadata = kwargs.get("sample_metadata")
    if sample_metadata:
        metadata.update(sample_metadata)
    target_text = kwargs.get("target_text")
    if target_text:
        metadata["target_text"] = target_text
    references = kwargs.get("references")
    if references:
        metadata["references"] = references
    if default_task_type is not None:
        metadata["task_type"] = kwargs.get("task_type", default_task_type)
        reference_image = kwargs.get("reference_image")
        if reference_image is not None:
            metadata["reference_image"] = reference_image
    return metadata


def _sampling_from_config(config: Any, field_names: tuple[str, ...]) -> dict[str, Any]:
    sampling: dict[str, Any] = {}
    for field_name in field_names:
        if not hasattr(config, field_name):
            raise ValueError(
                f"{type(config).__name__} is missing sampling field {field_name!r}",
            )
        sampling[field_name] = _sampling_value(getattr(config, field_name))
    return sampling


def _sampling_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


__all__ = [
    "RolloutEngineRequestBuilder",
    "RolloutRequestBuilder",
    "RolloutRequestPlan",
]
