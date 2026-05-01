"""Ray placement-group helpers for rollout workers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from vrl.distributed.ray.config import RayConfig
from vrl.distributed.ray.utils import require_ray, sort_node_gpu_key

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RayPlacement:
    """Placement group and stable bundle order for rollout workers."""

    placement_group: Any
    ordered_bundle_indices: list[int]


class _InfoActor:
    def get_ip_and_gpu_id(self) -> tuple[str, int]:
        ray = require_ray()
        gpu_ids = ray.get_gpu_ids()
        gpu_id = int(gpu_ids[0]) if gpu_ids else -1
        return str(ray.util.get_node_ip_address()), gpu_id


def _bundle(config: RayConfig) -> dict[str, float]:
    bundle = {"CPU": float(config.cpus_per_rollout_worker)}
    if config.gpus_per_rollout_worker > 0:
        bundle["GPU"] = float(config.gpus_per_rollout_worker)
    return bundle


def create_rollout_placement_group(config: RayConfig) -> RayPlacement:
    """Create a placement group for rollout workers and stable-sort bundles."""

    ray = require_ray()
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    bundles = [_bundle(config) for _ in range(config.num_rollout_workers)]
    pg = placement_group(bundles, strategy=config.placement_strategy)
    ray.get(pg.ready())

    RemoteInfoActor = ray.remote(
        num_cpus=config.cpus_per_rollout_worker,
        num_gpus=config.gpus_per_rollout_worker,
    )(_InfoActor)

    info_actors = [
        RemoteInfoActor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=i,
            ),
        ).remote()
        for i in range(config.num_rollout_workers)
    ]

    try:
        ip_gpu_pairs = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    finally:
        for actor in info_actors:
            ray.kill(actor, no_restart=True)

    bundle_infos = [
        (idx, node_ip, gpu_id)
        for idx, (node_ip, gpu_id) in enumerate(ip_gpu_pairs)
    ]
    ordered = [idx for idx, _, _ in sorted(bundle_infos, key=sort_node_gpu_key)]

    for logical_idx, actual_idx in enumerate(ordered):
        node_ip, gpu_id = ip_gpu_pairs[actual_idx]
        logger.info(
            "Ray rollout bundle %d -> actual bundle %d node=%s gpu=%s",
            logical_idx,
            actual_idx,
            node_ip,
            gpu_id,
        )

    return RayPlacement(placement_group=pg, ordered_bundle_indices=ordered)
