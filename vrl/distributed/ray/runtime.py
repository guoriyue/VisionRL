"""Rollout backend facade for Ray-distributed generation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from vrl.distributed.ray.rollout_executor import DistributedRolloutExecutor
from vrl.distributed.ray.weight_sync import RolloutWeightSync
from vrl.engine.generation.types import GenerationRequest, OutputBatch


class RayDistributedRuntime:
    """Collector-facing runtime backed by Ray rollout workers."""

    def __init__(
        self,
        executor: DistributedRolloutExecutor,
        *,
        weight_sync: RolloutWeightSync | None = None,
    ) -> None:
        self.executor = executor
        self.weight_sync = weight_sync
        self.current_policy_version: int | None = None

    async def generate(self, request: GenerationRequest) -> OutputBatch:
        if request.policy_version is None and self.current_policy_version is not None:
            request = replace(request, policy_version=self.current_policy_version)
        return await self.executor.execute(request)

    async def update_weights(self, state_ref: Any, policy_version: int) -> None:
        if self.weight_sync is None:
            raise RuntimeError("RayDistributedRuntime has no RolloutWeightSync")
        await self.weight_sync.push_to_rollout_workers(state_ref, policy_version)
        self.current_policy_version = int(policy_version)

    async def shutdown(self) -> None:
        return None


__all__ = ["RayDistributedRuntime"]
