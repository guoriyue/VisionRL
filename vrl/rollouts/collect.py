"""Shared rollout collector orchestration."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any

from vrl.engine.generation import (
    GenerationRuntime,
    OutputBatch,
    RolloutBackend,
    build_local_generation_runtime,
)
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.engine_requests import RolloutRequestBuilder, RolloutRequestPlan
from vrl.rollouts.packers.base import RolloutPackContext, RolloutPacker
from vrl.rollouts.rewards import RewardScorer


class RolloutCollector:
    """Generic collector: request -> generation runtime -> reward -> pack."""

    def __init__(
        self,
        *,
        model: Any | None,
        config: Any,
        family: str,
        task: str,
        executor_cls: type,
        request_builder: RolloutRequestBuilder,
        packer: RolloutPacker,
        reward_scorer: RewardScorer,
        default_group_size: int = 1,
        runtime: RolloutBackend | None = None,
        executor_kwargs: Mapping[str, Any] | None = None,
        phase_sink: dict[str, float] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.family = family
        self.task = task
        self.executor_cls = executor_cls
        self.request_builder = request_builder
        self.packer = packer
        self.reward_scorer = reward_scorer
        self.default_group_size = max(1, int(default_group_size))
        self._runtime = runtime
        self.executor_kwargs = dict(executor_kwargs or {})
        self.phase_sink = phase_sink

    def build_runtime(self) -> GenerationRuntime:
        return build_local_generation_runtime(
            model=self.model,
            executor_cls=self.executor_cls,
            cfg=self.config,
            executor_kwargs=self.executor_kwargs,
        )

    def set_runtime(self, runtime: RolloutBackend) -> None:
        if not callable(getattr(runtime, "generate", None)):
            raise TypeError(
                "rollout runtime must implement async generate(request) -> OutputBatch",
            )
        self._runtime = runtime

    @property
    def runtime(self) -> RolloutBackend:
        if self._runtime is None:
            self._runtime = self.build_runtime()
        return self._runtime

    async def shutdown(self) -> None:
        shutdown = getattr(self._runtime, "shutdown", None)
        if shutdown is not None:
            await shutdown()
        self._runtime = None

    async def collect(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> RolloutBatch:
        group_size = int(kwargs.get("group_size", self.default_group_size))
        plan = self.request_builder.build(prompts, group_size, dict(kwargs))

        profile = os.environ.get("VRL_PROFILE_COLLECT") == "1"
        phases: dict[str, float] = {}
        phase_t = _sync_time() if profile else None

        output = await self.runtime.generate(plan.request)
        if output.error:
            raise RuntimeError(
                f"{self.family}/{self.task} generation failed "
                f"(request_id={plan.request.request_id}): {output.error}",
            )

        if profile and phase_t is not None:
            now = _sync_time()
            phases["collect.engine_generate"] = now - phase_t
            phase_t = now

        batch = await self._output_batch_to_experience_batch(
            output,
            request_plan=plan,
            phases=phases if profile else None,
            phase_t=phase_t,
        )

        if profile and self.phase_sink is not None:
            self.phase_sink.clear()
            self.phase_sink.update(phases)

        return batch

    async def _output_batch_to_experience_batch(
        self,
        output: OutputBatch,
        *,
        request_plan: RolloutRequestPlan,
        phases: dict[str, float] | None = None,
        phase_t: float | None = None,
    ) -> RolloutBatch:
        context = RolloutPackContext(
            metadata=dict(request_plan.pack_metadata),
            device=_device_from_model(self.model),
            kl_reward=float(getattr(self.config, "kl_reward", 0.0)),
            rescale_to_unit=bool(getattr(self.config, "rescale_to_unit", False)),
        )
        reward_outputs = self.packer.reward_outputs(output, context)
        reward_prompts = self.packer.reward_prompts(output, context)
        rewards = await self.reward_scorer.score(
            reward_outputs,
            reward_prompts,
            request_plan.reward_metadata,
            _infer_device(reward_outputs, context.device),
        )

        if phases is not None and phase_t is not None:
            phases["collect.reward_score"] = _sync_time() - phase_t

        return await self.packer.pack(output, rewards, context)


def _device_from_model(model: Any | None) -> Any | None:
    return getattr(model, "device", None)


def _infer_device(value: Any, fallback: Any | None) -> Any:
    if fallback is not None:
        return fallback
    device = getattr(value, "device", None)
    if device is not None:
        return device
    return "cpu"


def _sync_time() -> float:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


__all__ = ["RolloutCollector"]
