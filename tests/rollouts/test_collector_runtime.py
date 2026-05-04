"""Tests for rollout collector runtime orchestration."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from vrl.engine import GenerationRequest, OutputBatch
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.collector.core import RolloutCollector
from vrl.rollouts.collector.requests import RolloutRequestPlan
from vrl.rollouts.packers.base import RolloutPackContext


class _RequestBuilder:
    def build(
        self,
        prompts: list[str],
        group_size: int,
        kwargs: dict[str, Any],
    ) -> RolloutRequestPlan:
        request = GenerationRequest(
            request_id="unit-request",
            family="unit",
            task="collect",
            prompts=prompts,
            samples_per_prompt=group_size,
            sampling={"seed": kwargs.get("seed")},
            return_artifacts={"output"},
            metadata={"source": "collector-test"},
            policy_version=kwargs.get("policy_version"),
        )
        return RolloutRequestPlan(
            request=request,
            reward_metadata={"reward": "metadata"},
            pack_metadata={"pack": "metadata"},
        )


class _Runtime:
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> OutputBatch:
        self.requests.append(request)
        batch_size = len(request.prompts) * request.samples_per_prompt
        return OutputBatch(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=list(request.prompts),
            sample_specs=[],
            output=torch.ones(batch_size, 1),
        )


class _RewardScorer:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def score(
        self,
        outputs: torch.Tensor,
        prompts: list[str],
        metadata: dict[str, Any],
        device: Any,
    ) -> torch.Tensor:
        self.calls.append(
            {
                "outputs": outputs,
                "prompts": prompts,
                "metadata": metadata,
                "device": device,
            },
        )
        return torch.arange(outputs.shape[0], dtype=torch.float32)


class _Packer:
    def __init__(self) -> None:
        self.reward_contexts: list[RolloutPackContext] = []
        self.pack_contexts: list[RolloutPackContext] = []

    def reward_outputs(
        self,
        output: OutputBatch,
        context: RolloutPackContext,
    ) -> torch.Tensor:
        self.reward_contexts.append(context)
        return output.output

    def reward_prompts(
        self,
        output: OutputBatch,
        context: RolloutPackContext,
    ) -> list[str]:
        del context
        return output.prompts

    async def pack(
        self,
        output: OutputBatch,
        rewards_raw: torch.Tensor,
        context: RolloutPackContext,
    ) -> RolloutBatch:
        self.pack_contexts.append(context)
        batch_size = rewards_raw.shape[0]
        return RolloutBatch(
            observations=torch.zeros(batch_size, 1),
            actions=torch.zeros(batch_size, 1),
            rewards=rewards_raw,
            dones=torch.ones(batch_size, dtype=torch.bool),
            group_ids=torch.arange(batch_size),
            context=dict(context.metadata),
            prompts=output.prompts,
        )


def _collector(
    *,
    runtime: _Runtime | None = None,
    packer: _Packer | None = None,
    reward_scorer: _RewardScorer | None = None,
) -> RolloutCollector:
    return RolloutCollector(
        model=None,
        config=object(),
        family="unit",
        task="collect",
        executor_cls=object,
        request_builder=_RequestBuilder(),
        packer=packer or _Packer(),
        reward_scorer=reward_scorer or _RewardScorer(),
        runtime=runtime,
    )


def test_collector_requires_runtime_before_collect() -> None:
    import asyncio

    collector = _collector()

    with pytest.raises(RuntimeError, match="runtime is not initialized"):
        asyncio.run(collector.collect(["p0"], group_size=1))


def test_collector_routes_request_through_runtime_reward_and_packer() -> None:
    import asyncio

    runtime = _Runtime()
    packer = _Packer()
    reward_scorer = _RewardScorer()
    collector = _collector(
        runtime=runtime,
        packer=packer,
        reward_scorer=reward_scorer,
    )

    batch = asyncio.run(
        collector.collect(["p0", "p1"], group_size=2, seed=5, policy_version=7),
    )

    assert len(runtime.requests) == 1
    request = runtime.requests[0]
    assert request.prompts == ["p0", "p1"]
    assert request.samples_per_prompt == 2
    assert request.sampling == {"seed": 5}
    assert request.policy_version == 7
    assert reward_scorer.calls[0]["metadata"] == {"reward": "metadata"}
    assert packer.reward_contexts[0].metadata == {"pack": "metadata"}
    assert packer.pack_contexts[0].metadata == {"pack": "metadata"}
    assert batch.rewards.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert batch.context == {"pack": "metadata"}
