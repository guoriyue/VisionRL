"""Ray train actor scaffolding."""

from __future__ import annotations

from vrl.distributed.ray.train.actor import RayTrainActor
from vrl.distributed.ray.train.group import RayTrainGroup, RayTrainRankSpec

__all__ = [
    "RayTrainActor",
    "RayTrainGroup",
    "RayTrainRankSpec",
]
