"""Tests for pure chunk gatherers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from vrl.engine.generation.diffusion import DiffusionChunkResult
from vrl.engine.generation.gather import (
    ChunkGatherer,
    DiffusionChunkGatherer,
    gather_pipeline_chunks,
    require_chunk_gatherer,
)
from vrl.engine.generation.types import GenerationRequest, OutputBatch
from vrl.engine.generation.worker import GenerationIdFactory


class _PureGatherer:
    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_specs: Sequence[Any],
        chunks: Sequence[Any],
    ) -> OutputBatch:
        return OutputBatch(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=list(request.prompts),
            sample_specs=list(sample_specs),
            output=list(chunks),
        )


def test_chunk_gatherer_accepts_pure_object_without_forward_chunk() -> None:
    request = _request()
    sample_specs = GenerationIdFactory().build_sample_specs(request)
    gatherer = _PureGatherer()

    assert isinstance(gatherer, ChunkGatherer)
    assert not hasattr(gatherer, "forward_chunk")
    assert require_chunk_gatherer(gatherer) is gatherer

    output = gather_pipeline_chunks(gatherer, request, sample_specs, ["chunk"])

    assert output.output == ["chunk"]


def test_diffusion_chunk_gatherer_gathers_without_model_object() -> None:
    request = _request(cfg=False)
    sample_specs = GenerationIdFactory().build_sample_specs(request)
    gatherer = DiffusionChunkGatherer(model_family="sd3_5")
    context = {
        "guidance_scale": 4.5,
        "cfg": False,
        "model_family": "sd3_5",
    }

    output = gatherer.gather_chunks(request, sample_specs, _diffusion_chunks(context))

    assert output.output.device.type == "cpu"
    assert output.metrics is not None
    assert output.metrics.num_steps == 2
    assert output.metrics.micro_batches == 2
    assert output.rollout_trajectory_data is not None
    denoising_env = output.rollout_trajectory_data.denoising_env
    assert denoising_env is not None
    assert denoising_env.extra["context"] == context


def test_diffusion_chunk_gatherer_can_ignore_cfg_sampling_flag() -> None:
    request = _request(family="cosmos", task="v2w", cfg=False)
    sample_specs = GenerationIdFactory().build_sample_specs(request)
    gatherer = DiffusionChunkGatherer(
        model_family="cosmos",
        respect_cfg_flag=False,
    )
    context = {
        "guidance_scale": 4.5,
        "cfg": True,
        "model_family": "cosmos",
    }

    output = gatherer.gather_chunks(request, sample_specs, _diffusion_chunks(context))

    assert output.rollout_trajectory_data is not None
    denoising_env = output.rollout_trajectory_data.denoising_env
    assert denoising_env is not None
    assert denoising_env.extra["context"] == context


def _request(
    *,
    family: str = "sd3_5",
    task: str = "t2i",
    cfg: bool = True,
) -> GenerationRequest:
    return GenerationRequest(
        request_id="req",
        family=family,
        task=task,
        prompts=["p0"],
        samples_per_prompt=2,
        sampling={
            "num_steps": 2,
            "guidance_scale": 4.5,
            "cfg": cfg,
            "seed": 1,
        },
    )


def _diffusion_chunks(context: dict[str, Any]) -> list[DiffusionChunkResult]:
    return [_diffusion_chunk(1.0, context), _diffusion_chunk(2.0, context)]


def _diffusion_chunk(value: float, context: dict[str, Any]) -> DiffusionChunkResult:
    return DiffusionChunkResult(
        observations=torch.full((1, 2, 1), value),
        actions=torch.full((1, 2, 1), value + 1),
        log_probs=torch.full((1, 2), value + 2),
        timesteps=torch.arange(2).view(1, 2),
        kl=torch.full((1, 2), value + 3),
        video=torch.full((1, 3, 4, 4), value),
        training_extras={},
        context=context,
    )
