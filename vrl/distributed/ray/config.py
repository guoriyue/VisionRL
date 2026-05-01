"""Configuration for Ray-backed distributed rollout collection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RayConfig:
    """Resource configuration for Ray rollout workers.

    P0 only owns rollout collector workers. Trainer ownership, weight sync, and
    train/rollout GPU partitioning are intentionally left to the later Ray
    training sprint.
    """

    enable: bool = False
    num_rollout_workers: int = 1
    gpus_per_rollout_worker: float = 1.0
    cpus_per_rollout_worker: float = 1.0
    placement_strategy: str = "PACK"

    def __post_init__(self) -> None:
        if self.num_rollout_workers < 1:
            raise ValueError("num_rollout_workers must be >= 1")
        if self.gpus_per_rollout_worker < 0:
            raise ValueError("gpus_per_rollout_worker must be >= 0")
        if self.cpus_per_rollout_worker <= 0:
            raise ValueError("cpus_per_rollout_worker must be > 0")
        if not self.placement_strategy:
            raise ValueError("placement_strategy must be non-empty")
