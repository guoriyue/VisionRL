"""Configuration for Ray-backed distributed rollout collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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

    @classmethod
    def from_cfg(cls, cfg: Any) -> DistributedRolloutConfig:
        """Build rollout config from a full training cfg or rollout cfg slice."""
        if isinstance(cfg, cls):
            return cfg
        if isinstance(cfg, RayConfig):
            return cls.from_legacy(cfg)

        direct_backend = _config_get(cfg, "backend", _MISSING)
        distributed = _config_get(cfg, "distributed", _MISSING)
        if distributed is _MISSING:
            if direct_backend is _MISSING:
                return cls()
            distributed = cfg
        rollout = _config_get(distributed, "rollout", _MISSING)
        if rollout is _MISSING:
            rollout = distributed

        backend = _config_get(distributed, "backend", _config_get(rollout, "backend", "local"))

        return cls(
            backend=str(backend),
            num_workers=int(_rollout_get(distributed, rollout, "num_workers", 1)),
            gpus_per_worker=float(
                _rollout_get(distributed, rollout, "gpus_per_worker", 1.0),
            ),
            cpus_per_worker=float(
                _rollout_get(distributed, rollout, "cpus_per_worker", 1.0),
            ),
            placement_strategy=str(
                _rollout_get(distributed, rollout, "placement_strategy", "SPREAD"),
            ),
            allow_driver_gpu_overlap=bool(
                _rollout_get(
                    distributed,
                    rollout,
                    "allow_driver_gpu_overlap",
                    False,
                ),
            ),
            max_inflight_chunks_per_worker=int(
                _rollout_get(
                    distributed,
                    rollout,
                    "max_inflight_chunks_per_worker",
                    1,
                ),
            ),
            sync_trainable_state=str(
                _rollout_get(distributed, rollout, "sync_trainable_state", "disabled"),
            ),
        )


_MISSING = object()


def _config_get(node: Any, key: str, default: Any) -> Any:
    if node is None:
        return default
    getter = getattr(node, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    try:
        return node[key]
    except (KeyError, IndexError, TypeError):
        pass
    return getattr(node, key, default)


def _rollout_get(distributed: Any, rollout: Any, key: str, default: Any) -> Any:
    return _config_get(rollout, key, _config_get(distributed, key, default))
