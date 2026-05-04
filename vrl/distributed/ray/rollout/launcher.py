"""Launch Ray rollout workers and assemble the collector-facing runtime."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from vrl.distributed.ray.dependencies import require_ray
from vrl.distributed.ray.placement.group import create_rollout_placement_group
from vrl.distributed.ray.rollout.executor import DistributedRolloutExecutor
from vrl.distributed.ray.rollout.planner import DistributedExecutionPlanner
from vrl.distributed.ray.rollout.runtime import RayDistributedRuntime
from vrl.distributed.ray.rollout.types import RayWorkerHandle
from vrl.distributed.ray.rollout.weight_sync import RayRolloutWeightSync
from vrl.distributed.ray.rollout.worker import RayRolloutWorker
from vrl.engine.core.runtime_spec import GenerationRuntimeSpec
from vrl.engine.gather import ChunkGatherer, require_chunk_gatherer
from vrl.rollouts.runtime.config import RolloutBackendConfig


@dataclass(slots=True)
class RayRolloutLauncher:
    """Create Ray rollout actors and return a ``RayDistributedRuntime``."""

    init_ray: bool = True
    ray_init_kwargs: dict[str, Any] = field(default_factory=dict)

    def launch(
        self,
        config: RolloutBackendConfig | Mapping[str, Any],
        runtime_spec: GenerationRuntimeSpec | Mapping[str, Any],
        gatherer: ChunkGatherer,
    ) -> RayDistributedRuntime:
        rollout_config = RolloutBackendConfig.from_cfg(config)
        if rollout_config.backend != "ray":
            raise ValueError(
                "RayRolloutLauncher requires distributed rollout backend='ray', "
                f"got {rollout_config.backend!r}",
            )

        spec = GenerationRuntimeSpec.from_value(runtime_spec)
        if not spec.family:
            raise ValueError("GenerationRuntimeSpec.family is required")
        chunk_gatherer = require_chunk_gatherer(gatherer)

        ray = require_ray()
        if self.init_ray and not ray.is_initialized():
            ray.init(**self.ray_init_kwargs)

        placement = create_rollout_placement_group(rollout_config)

        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        RemoteRolloutWorker = ray.remote(
            num_cpus=rollout_config.cpus_per_worker,
            num_gpus=rollout_config.gpus_per_worker,
        )(RayRolloutWorker)

        actors: list[Any] = []
        worker_ids: list[str] = []
        for logical_idx, bundle_idx in enumerate(placement.ordered_bundle_indices):
            worker_id = f"rollout-{logical_idx}"
            actor = RemoteRolloutWorker.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement.placement_group,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=bundle_idx,
                ),
            ).remote(worker_id, spec)
            actors.append(actor)
            worker_ids.append(worker_id)

        try:
            ray.get([actor.load_policy.remote() for actor in actors])
            metadata = ray.get([actor.worker_metadata.remote() for actor in actors])
        except Exception:
            _kill_actors(ray, actors)
            _remove_placement_group(ray, placement.placement_group)
            raise

        workers = [
            RayWorkerHandle(
                worker_id=worker_id,
                node_id=str(meta.get("node_ip", "unknown")),
                gpu_ids=tuple(int(gpu_id) for gpu_id in meta.get("gpu_ids", ())),
                actor=actor,
            )
            for worker_id, actor, meta in zip(worker_ids, actors, metadata, strict=True)
        ]

        executor = DistributedRolloutExecutor(
            DistributedExecutionPlanner(),
            workers,
            chunk_gatherer,
            max_inflight_chunks_per_worker=rollout_config.max_inflight_chunks_per_worker,
        )
        weight_sync = (
            RayRolloutWeightSync(workers)
            if rollout_config.sync_trainable_state != "disabled"
            else None
        )
        runtime = RayDistributedRuntime(
            executor,
            weight_sync=weight_sync,
            owned_workers=workers,
            placement_group=placement.placement_group,
        )
        if spec.policy_version is not None:
            runtime.current_policy_version = spec.policy_version
        return runtime


def launch_ray_rollout_runtime(
    config: RolloutBackendConfig | Mapping[str, Any],
    runtime_spec: GenerationRuntimeSpec | Mapping[str, Any],
    gatherer: ChunkGatherer,
    *,
    init_ray: bool = True,
    ray_init_kwargs: dict[str, Any] | None = None,
) -> RayDistributedRuntime:
    """Functional wrapper around ``RayRolloutLauncher``."""

    return RayRolloutLauncher(
        init_ray=init_ray,
        ray_init_kwargs={} if ray_init_kwargs is None else dict(ray_init_kwargs),
    ).launch(config, runtime_spec, gatherer)


def _kill_actors(ray: Any, actors: list[Any]) -> None:
    for actor in actors:
        with contextlib.suppress(Exception):
            ray.kill(actor, no_restart=True)


def _remove_placement_group(ray: Any, placement_group: Any) -> None:
    with contextlib.suppress(Exception):
        from ray.util import remove_placement_group

        remove_placement_group(placement_group)


__all__ = [
    "RayRolloutLauncher",
    "launch_ray_rollout_runtime",
]
