"""Manager for Ray collector actors."""

from __future__ import annotations

import asyncio
from typing import Any

from vrl.distributed.ray.collector_actor import CollectorJob, RayCollectorActor
from vrl.distributed.ray.config import RayConfig
from vrl.distributed.ray.placement_group import RayPlacement
from vrl.distributed.ray.utils import require_ray, split_round_robin
from vrl.rollouts.types import ExperienceBatch, stack_batches


class RayCollectorManager:
    """Own Ray collector actors and gather prompt-level rollout jobs."""

    def __init__(
        self,
        cfg: Any,
        family: str,
        config: RayConfig,
        placement: RayPlacement,
    ) -> None:
        self.cfg = cfg
        self.family = family
        self.config = config
        self.placement = placement
        self._ray = require_ray()

        if len(placement.ordered_bundle_indices) < config.num_rollout_workers:
            raise ValueError(
                "placement has fewer bundles than num_rollout_workers: "
                f"{len(placement.ordered_bundle_indices)} < {config.num_rollout_workers}",
            )

        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        RemoteCollectorActor = self._ray.remote(
            num_cpus=config.cpus_per_rollout_worker,
            num_gpus=config.gpus_per_rollout_worker,
        )(RayCollectorActor)

        self._actors = [
            RemoteCollectorActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement.placement_group,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=placement.ordered_bundle_indices[i],
                ),
            ).remote(cfg, family)
            for i in range(config.num_rollout_workers)
        ]

    @property
    def actors(self) -> list[Any]:
        return list(self._actors)

    @staticmethod
    def split_jobs(jobs: list[CollectorJob], num_workers: int) -> list[list[CollectorJob]]:
        return split_round_robin(jobs, num_workers)

    async def collect(self, jobs: list[CollectorJob]) -> ExperienceBatch:
        """Collect jobs across workers and stack partial ``ExperienceBatch`` values."""

        if not jobs:
            raise ValueError("RayCollectorManager.collect requires at least one job")

        shards = self.split_jobs(jobs, self.config.num_rollout_workers)
        refs = [
            actor.collect_jobs.remote(shard)
            for actor, shard in zip(self._actors, shards, strict=False)
            if shard
        ]
        nested_batches: list[list[ExperienceBatch]] = await asyncio.to_thread(
            self._ray.get,
            refs,
        )
        batches = [batch for actor_batches in nested_batches for batch in actor_batches]
        if not batches:
            raise RuntimeError("Ray collector actors returned no batches")
        return stack_batches(batches)

    def close(self) -> None:
        for actor in self._actors:
            self._ray.kill(actor, no_restart=True)
        self._actors.clear()
