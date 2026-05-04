"""Backend configuration for rollout collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RolloutBackendConfig:
    """Resource and backend selection config for rollout collection."""

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
    def from_cfg(cls, cfg: Any) -> RolloutBackendConfig:
        """Build rollout backend config from a full training cfg or rollout cfg slice."""
        if isinstance(cfg, cls):
            return cfg

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

    def to_dict(self) -> dict[str, Any]:
        """Return the flat config shape accepted by rollout backends."""
        return {
            "backend": self.backend,
            "num_workers": self.num_workers,
            "gpus_per_worker": self.gpus_per_worker,
            "cpus_per_worker": self.cpus_per_worker,
            "placement_strategy": self.placement_strategy,
            "allow_driver_gpu_overlap": self.allow_driver_gpu_overlap,
            "max_inflight_chunks_per_worker": self.max_inflight_chunks_per_worker,
            "sync_trainable_state": self.sync_trainable_state,
        }


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


__all__ = ["RolloutBackendConfig"]
