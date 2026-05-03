"""Tests that diffusion collectors can run against an injected runtime."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from vrl.engine.generation import GenerationIdFactory, OutputBatch, RolloutBackend
from vrl.rollouts.collectors import (
    CosmosPredict2CollectorConfig,
    SD3_5CollectorConfig,
    Wan_2_1CollectorConfig,
    build_rollout_collector,
)
from vrl.rollouts.experience import ExperienceBatch


class _FakeRuntime(RolloutBackend):
    def __init__(self) -> None:
        self.requests: list[Any] = []

    async def generate(self, request: Any) -> OutputBatch:
        self.requests.append(request)
        return OutputBatch(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=list(request.prompts),
            sample_specs=GenerationIdFactory().build_sample_specs(request),
            output=torch.zeros(
                len(request.prompts) * request.samples_per_prompt,
                3,
                4,
                4,
            ),
        )

    async def shutdown(self) -> None:
        return None


async def _fake_to_batch(*args: Any, **kwargs: Any) -> ExperienceBatch:
    output = args[0]
    del kwargs
    batch_size = len(output.sample_specs)
    return ExperienceBatch(
        observations=torch.zeros(batch_size, 1, 1),
        actions=torch.zeros(batch_size, 1, 1),
        rewards=torch.zeros(batch_size),
        dones=torch.ones(batch_size, dtype=torch.bool),
        group_ids=torch.tensor([spec.prompt_index for spec in output.sample_specs]),
        prompts=[spec.prompt for spec in output.sample_specs],
    )


@pytest.mark.parametrize(
    ("family", "config_cls"),
    [
        pytest.param("sd3_5", SD3_5CollectorConfig, id="sd3_5"),
        pytest.param("wan_2_1", Wan_2_1CollectorConfig, id="wan_2_1"),
        pytest.param("cosmos", CosmosPredict2CollectorConfig, id="cosmos"),
    ],
)
def test_diffusion_collector_uses_injected_runtime_without_model(
    family: str,
    config_cls: Any,
) -> None:
    import asyncio

    runtime = _FakeRuntime()
    collector = build_rollout_collector(
        family,
        model=None,
        reward_fn=object(),
        config=config_cls(),
        runtime=runtime,
    )
    collector._output_batch_to_experience_batch = _fake_to_batch

    batch = asyncio.run(
        collector.collect(["p0", "p1"], group_size=2, policy_version=9),
    )

    assert len(runtime.requests) == 1
    request = runtime.requests[0]
    assert request.prompts == ["p0", "p1"]
    assert request.samples_per_prompt == 2
    assert request.policy_version == 9
    assert batch.group_ids.tolist() == [0, 0, 1, 1]
