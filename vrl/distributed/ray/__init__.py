"""Ray distributed rollout utilities."""

from __future__ import annotations

from vrl.distributed.ray.config import DistributedRolloutConfig
from vrl.distributed.ray.placement.group import (
    RayPlacement,
    create_rollout_placement_group,
)
from vrl.distributed.ray.rollout.executor import DistributedRolloutExecutor
from vrl.distributed.ray.rollout.family_runtime import (
    RayRolloutRuntimeInputs,
    build_family_ray_rollout_runtime_inputs,
)
from vrl.distributed.ray.rollout.launcher import (
    RayRolloutLauncher,
    launch_ray_rollout_runtime,
)
from vrl.distributed.ray.rollout.planner import DeviceAssignment, DistributedExecutionPlanner
from vrl.distributed.ray.rollout.runtime import RayDistributedRuntime
from vrl.distributed.ray.rollout.spec import RolloutRuntimeSpec
from vrl.distributed.ray.rollout.types import RayChunkResult, RayWorkerHandle
from vrl.distributed.ray.rollout.weight_sync import RayRolloutWeightSync, RolloutWeightSync
from vrl.distributed.ray.rollout.worker import RayRolloutWorker
from vrl.distributed.ray.train.actor import RayTrainActor
from vrl.distributed.ray.train.group import RayTrainGroup, RayTrainRankSpec

__all__ = [
    "DeviceAssignment",
    "DistributedExecutionPlanner",
    "DistributedRolloutConfig",
    "DistributedRolloutExecutor",
    "RayChunkResult",
    "RayDistributedRuntime",
    "RayPlacement",
    "RayRolloutLauncher",
    "RayRolloutRuntimeInputs",
    "RayRolloutWeightSync",
    "RayRolloutWorker",
    "RayTrainActor",
    "RayTrainGroup",
    "RayTrainRankSpec",
    "RayWorkerHandle",
    "RolloutRuntimeSpec",
    "RolloutWeightSync",
    "build_family_ray_rollout_runtime_inputs",
    "create_rollout_placement_group",
    "launch_ray_rollout_runtime",
]
