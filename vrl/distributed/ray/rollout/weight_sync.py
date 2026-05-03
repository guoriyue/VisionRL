"""Synchronous weight-version coordination for Ray rollout workers."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

from vrl.distributed.ray.dependencies import require_ray
from vrl.distributed.ray.rollout.types import RayWorkerHandle


class RolloutWeightSync(Protocol):
    """Push trainable rollout state to workers with a known policy version."""

    async def push_to_rollout_workers(
        self,
        state_ref: Any,
        policy_version: int,
    ) -> None:
        ...


class RayRolloutWeightSync(RolloutWeightSync):
    """Call ``update_weights`` on every Ray rollout worker."""

    def __init__(self, workers: list[RayWorkerHandle]) -> None:
        self.workers = list(workers)

    async def push_to_rollout_workers(
        self,
        state_ref: Any,
        policy_version: int,
    ) -> None:
        refs: list[Any] = []
        for worker in self.workers:
            actor = worker.actor
            if actor is None:
                raise RuntimeError(f"worker {worker.worker_id!r} has no actor")
            update_weights = actor.update_weights
            remote = getattr(update_weights, "remote", None)
            if callable(remote):
                refs.append(remote(state_ref, policy_version))
            else:
                update_weights(state_ref, policy_version)

        if refs:
            ray = require_ray()
            await asyncio.to_thread(ray.get, refs)


__all__ = [
    "RayRolloutWeightSync",
    "RolloutWeightSync",
]
