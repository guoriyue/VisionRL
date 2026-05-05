"""Rollout backend facade for Ray-distributed generation."""

from __future__ import annotations

import contextlib
from dataclasses import replace
from typing import Any

from vrl.distributed.ray.dependencies import require_ray
from vrl.distributed.ray.rollout.executor import DistributedRolloutExecutor
from vrl.distributed.ray.rollout.types import RayWorkerHandle
from vrl.distributed.ray.rollout.weight_sync import RolloutWeightSync
from vrl.engine.core.runtime import RolloutBackend
from vrl.engine.core.types import GenerationRequest, OutputBatch


class RayDistributedRuntime(RolloutBackend):
    """Collector-facing runtime backed by Ray rollout workers."""

    def __init__(
        self,
        executor: DistributedRolloutExecutor,
        *,
        weight_sync: RolloutWeightSync | None = None,
        owned_workers: list[RayWorkerHandle] | None = None,
        placement_group: Any | None = None,
    ) -> None:
        self.executor = executor
        self.weight_sync = weight_sync
        self._owned_workers = list(owned_workers or [])
        self._placement_group = placement_group
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
        if not self._owned_workers and self._placement_group is None:
            return None
        ray = require_ray()
        release_refs: list[Any] = []
        for worker in self._owned_workers:
            actor = worker.actor
            if actor is None:
                continue
            with contextlib.suppress(Exception):
                release_refs.append(actor.release_policy.remote())
        if release_refs:
            with contextlib.suppress(Exception):
                ray.get(release_refs, timeout=60)
        for worker in self._owned_workers:
            actor = worker.actor
            if actor is None:
                continue
            with contextlib.suppress(Exception):
                ray.kill(actor, no_restart=True)
        self._owned_workers.clear()
        if self._placement_group is not None:
            with contextlib.suppress(Exception):
                from ray.util import remove_placement_group

                remove_placement_group(self._placement_group)
            self._placement_group = None
        return None


__all__ = ["RayDistributedRuntime"]
