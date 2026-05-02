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


@dataclass(slots=True)
class DistributedRolloutConfig:
    """Resource configuration for distributed large rollout execution."""

    backend: str = "local"
    num_workers: int = 1
    gpus_per_worker: float = 1.0
    cpus_per_worker: float = 1.0
    placement_strategy: str = "SPREAD"
    allow_driver_gpu_overlap: bool = False
    max_inflight_chunks_per_worker: int = 1
    sync_trainable_state: str = "disabled"

    def __post_init__(self) -> None:
        if self.backend not in {"local", "ray"}:
            raise ValueError("backend must be 'local' or 'ray'")
        if self.num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        if self.gpus_per_worker < 0:
            raise ValueError("gpus_per_worker must be >= 0")
        if self.cpus_per_worker <= 0:
            raise ValueError("cpus_per_worker must be > 0")
        if not self.placement_strategy:
            raise ValueError("placement_strategy must be non-empty")
        if self.max_inflight_chunks_per_worker < 1:
            raise ValueError("max_inflight_chunks_per_worker must be >= 1")
        if self.sync_trainable_state not in {"disabled", "lora_only"}:
            raise ValueError("sync_trainable_state must be 'disabled' or 'lora_only'")

    @classmethod
    def from_legacy(cls, config: RayConfig) -> DistributedRolloutConfig:
        """Map the old prompt-level collector config to the new rollout config."""

        return cls(
            backend="ray" if config.enable else "local",
            num_workers=config.num_rollout_workers,
            gpus_per_worker=config.gpus_per_rollout_worker,
            cpus_per_worker=config.cpus_per_rollout_worker,
            placement_strategy=config.placement_strategy,
        )
