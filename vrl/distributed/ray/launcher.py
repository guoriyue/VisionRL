"""Launch Ray rollout workers and assemble the collector-facing runtime."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from vrl.distributed.ray.config import DistributedRolloutConfig, RayConfig
from vrl.distributed.ray.placement_group import create_rollout_placement_group
from vrl.distributed.ray.planning import DistributedExecutionPlanner
from vrl.distributed.ray.rollout_executor import DistributedRolloutExecutor
from vrl.distributed.ray.rollout_worker import RayRolloutWorker
from vrl.distributed.ray.runtime import RayDistributedRuntime
from vrl.distributed.ray.spec import RolloutRuntimeSpec
from vrl.distributed.ray.types import RayWorkerHandle
from vrl.distributed.ray.utils import require_ray
from vrl.distributed.ray.weight_sync import RayRolloutWeightSync
from vrl.engine.generation.gather import ChunkGatherer, require_chunk_gatherer


@dataclass(slots=True)
class RayRolloutLauncher:
    """Create Ray rollout actors and return a ``RayDistributedRuntime``."""

    init_ray: bool = True
    ray_init_kwargs: dict[str, Any] = field(default_factory=dict)

    def launch(
        self,
        config: DistributedRolloutConfig | RayConfig | Mapping[str, Any],
        runtime_spec: RolloutRuntimeSpec | Mapping[str, Any],
        gatherer: ChunkGatherer,
    ) -> RayDistributedRuntime:
        rollout_config = DistributedRolloutConfig.from_cfg(config)
        if rollout_config.backend != "ray":
            raise ValueError(
                "RayRolloutLauncher requires distributed rollout backend='ray', "
                f"got {rollout_config.backend!r}",
            )

        spec = RolloutRuntimeSpec.from_value(runtime_spec)
        if not spec.family:
            raise ValueError("Ray rollout runtime_spec.family is required")
        chunk_gatherer = require_chunk_gatherer(gatherer)

        ray = require_ray()
        if self.init_ray and not ray.is_initialized():
            ray.init(**self.ray_init_kwargs)

        ray_config = _to_ray_config(rollout_config)
        placement = create_rollout_placement_group(ray_config)

        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        RemoteRolloutWorker = ray.remote(
            num_cpus=ray_config.cpus_per_rollout_worker,
            num_gpus=ray_config.gpus_per_rollout_worker,
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
            ).remote(worker_id, spec.family, spec)
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
    config: DistributedRolloutConfig | RayConfig | Mapping[str, Any],
    runtime_spec: RolloutRuntimeSpec | Mapping[str, Any],
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


def _to_ray_config(config: DistributedRolloutConfig) -> RayConfig:
    return RayConfig(
        enable=True,
        num_rollout_workers=config.num_workers,
        gpus_per_rollout_worker=config.gpus_per_worker,
        cpus_per_rollout_worker=config.cpus_per_worker,
        placement_strategy=config.placement_strategy,
    )


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
