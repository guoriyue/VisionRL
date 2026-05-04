"""Shared helpers for batching multiple generation requests."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Protocol

from vrl.engine.core.types import (
    GenerationRequest,
    GenerationSampleSpec,
    OutputBatch,
    RolloutDebugTensors,
    RolloutDenoisingEnv,
    RolloutDitTrajectory,
    RolloutTrajectoryData,
)


class _SingleRequestExecutor(Protocol):
    family: str
    task: str

    def forward(
        self,
        request: GenerationRequest,
        sample_specs: list[GenerationSampleSpec],
    ) -> OutputBatch: ...


def forward_batch_by_merging_prompts(
    executor: _SingleRequestExecutor,
    requests: list[GenerationRequest],
    sample_specs_by_request: dict[str, list[GenerationSampleSpec]],
) -> dict[str, OutputBatch]:
    """Run same-config requests as one prompt-major executor call.

    This is intentionally stricter than ``WorkloadSignature``. The planner may
    select same-shape requests, but the executor only fuses requests whose full
    sampling config and artifact contract are identical. If seeds or CFG differ,
    the worker should fall back to per-request execution.
    """

    if not requests:
        return {}
    if len(requests) == 1:
        request = requests[0]
        return {
            request.request_id: executor.forward(
                request,
                sample_specs_by_request[request.request_id],
            )
        }

    first = requests[0]
    _validate_mergeable(requests)

    prompts: list[str] = []
    merged_specs: list[GenerationSampleSpec] = []
    request_sample_counts: dict[str, int] = {}
    for request in requests:
        prompts.extend(request.prompts)
        specs = sample_specs_by_request[request.request_id]
        merged_specs.extend(specs)
        request_sample_counts[request.request_id] = len(specs)

    merged_request = GenerationRequest(
        request_id=f"batch:{first.request_id}",
        family=first.family,
        task=first.task,
        prompts=prompts,
        samples_per_prompt=first.samples_per_prompt,
        sampling=dict(first.sampling),
        return_artifacts=set(first.return_artifacts),
        metadata={"batched_request_ids": [r.request_id for r in requests]},
        priority=min(r.priority for r in requests),
        policy_version=first.policy_version,
    )
    merged_output = executor.forward(merged_request, merged_specs)

    outputs: dict[str, OutputBatch] = {}
    offset = 0
    for request in requests:
        count = request_sample_counts[request.request_id]
        outputs[request.request_id] = _slice_output_batch(
            merged_output,
            request=request,
            sample_specs=sample_specs_by_request[request.request_id],
            offset=offset,
            count=count,
            total=len(merged_specs),
        )
        offset += count
    return outputs


def _validate_mergeable(requests: list[GenerationRequest]) -> None:
    first = requests[0]
    for request in requests[1:]:
        if request.family != first.family or request.task != first.task:
            raise ValueError("Cannot merge requests from different family/task")
        if request.samples_per_prompt != first.samples_per_prompt:
            raise ValueError("Cannot merge requests with different sample counts")
        if request.sampling != first.sampling:
            raise ValueError("Cannot merge requests with different sampling config")
        if request.return_artifacts != first.return_artifacts:
            raise ValueError("Cannot merge requests with different artifact modes")
        if request.policy_version != first.policy_version:
            raise ValueError("Cannot merge requests with different policy versions")


def _slice_output_batch(
    output: OutputBatch,
    *,
    request: GenerationRequest,
    sample_specs: list[GenerationSampleSpec],
    offset: int,
    count: int,
    total: int,
) -> OutputBatch:
    return OutputBatch(
        request_id=request.request_id,
        family=request.family,
        task=request.task,
        prompts=list(request.prompts),
        sample_specs=sample_specs,
        output=_slice_value(output.output, offset, count, total),
        trajectory_timesteps=_slice_value(
            output.trajectory_timesteps,
            offset,
            count,
            total,
        ),
        trajectory_latents=_slice_value(
            output.trajectory_latents,
            offset,
            count,
            total,
        ),
        rollout_trajectory_data=_slice_rollout_trajectory(
            output.rollout_trajectory_data,
            offset,
            count,
            total,
        ),
        trajectory_decoded=_slice_value(
            output.trajectory_decoded,
            offset,
            count,
            total,
        ),
        extra=_slice_value(output.extra, offset, count, total),
        metrics=replace(output.metrics, num_prompts=len(request.prompts), num_samples=count)
        if output.metrics is not None
        else None,
        peak_memory_mb=output.peak_memory_mb,
        error=output.error,
    )


def _slice_rollout_trajectory(
    data: RolloutTrajectoryData | None,
    offset: int,
    count: int,
    total: int,
) -> RolloutTrajectoryData | None:
    if data is None:
        return None
    return RolloutTrajectoryData(
        rollout_log_probs=_slice_value(
            data.rollout_log_probs,
            offset,
            count,
            total,
        ),
        rollout_debug_tensors=_slice_debug_tensors(
            data.rollout_debug_tensors,
            offset,
            count,
            total,
        ),
        denoising_env=_slice_denoising_env(
            data.denoising_env,
            offset,
            count,
            total,
        ),
        dit_trajectory=_slice_dit_trajectory(
            data.dit_trajectory,
            offset,
            count,
            total,
        ),
    )


def _slice_debug_tensors(
    data: RolloutDebugTensors | None,
    offset: int,
    count: int,
    total: int,
) -> RolloutDebugTensors | None:
    if data is None:
        return None
    return RolloutDebugTensors(
        rollout_variance_noises=_slice_value(
            data.rollout_variance_noises,
            offset,
            count,
            total,
        ),
        rollout_prev_sample_means=_slice_value(
            data.rollout_prev_sample_means,
            offset,
            count,
            total,
        ),
        rollout_noise_std_devs=_slice_value(
            data.rollout_noise_std_devs,
            offset,
            count,
            total,
        ),
        rollout_model_outputs=_slice_value(
            data.rollout_model_outputs,
            offset,
            count,
            total,
        ),
    )


def _slice_denoising_env(
    data: RolloutDenoisingEnv | None,
    offset: int,
    count: int,
    total: int,
) -> RolloutDenoisingEnv | None:
    if data is None:
        return None
    return RolloutDenoisingEnv(
        image_kwargs=_slice_value(data.image_kwargs, offset, count, total),
        pos_cond_kwargs=_slice_value(data.pos_cond_kwargs, offset, count, total),
        neg_cond_kwargs=_slice_value(data.neg_cond_kwargs, offset, count, total),
        guidance=_slice_value(data.guidance, offset, count, total),
        extra=_slice_value(data.extra, offset, count, total),
    )


def _slice_dit_trajectory(
    data: RolloutDitTrajectory | None,
    offset: int,
    count: int,
    total: int,
) -> RolloutDitTrajectory | None:
    if data is None:
        return None
    return RolloutDitTrajectory(
        latents=_slice_value(data.latents, offset, count, total),
        timesteps=_slice_value(data.timesteps, offset, count, total),
    )


def _slice_value(value: Any, offset: int, count: int, total: int) -> Any:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 0 and int(shape[0]) == total:
        return value[offset : offset + count]
    if isinstance(value, list) and len(value) == total:
        return value[offset : offset + count]
    if isinstance(value, tuple) and len(value) == total:
        return value[offset : offset + count]
    if isinstance(value, dict):
        return {key: _slice_value(inner, offset, count, total) for key, inner in value.items()}
    return value
