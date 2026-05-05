"""Weight synchronisation between trainer and inference workers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class WeightSyncer(ABC):
    """Pushes updated weights from the trainer to inference workers."""

    @abstractmethod
    async def push(self, state_dict: dict[str, Any]) -> None:
        """Send updated weights to inference workers."""

    @abstractmethod
    async def pull(self) -> dict[str, Any]:
        """Fetch the latest weights."""


class InMemoryWeightSyncer(WeightSyncer):
    """Simple in-process syncer for single-GPU development."""

    def __init__(self) -> None:
        self._state: dict[str, Any] = {}

    async def push(self, state_dict: dict[str, Any]) -> None:
        self._state = dict(state_dict)

    async def pull(self) -> dict[str, Any]:
        return dict(self._state)


class RayRuntimeWeightSyncer(WeightSyncer):
    """Bridge ``OnlineTrainer`` weight pushes to a Ray rollout runtime."""

    def __init__(
        self,
        runtime: Any,
        *,
        initial_policy_version: int | None = None,
    ) -> None:
        update_weights = getattr(runtime, "update_weights", None)
        if not callable(update_weights):
            raise TypeError("runtime must expose async update_weights(state, version)")
        self.runtime = runtime
        self._next_policy_version = _resolve_next_policy_version(
            runtime,
            initial_policy_version,
        )
        self._last_state: dict[str, Any] = {}

    async def push(self, state_dict: dict[str, Any]) -> None:
        state = _cpu_state_dict(state_dict)
        policy_version = self._next_policy_version
        await self.runtime.update_weights(state, policy_version)
        self._next_policy_version = policy_version + 1
        self._last_state = state

    async def pull(self) -> dict[str, Any]:
        return dict(self._last_state)


def build_runtime_weight_syncer(
    runtime: Any,
    *,
    initial_policy_version: int | None = None,
) -> WeightSyncer | None:
    """Return a syncer when a rollout runtime supports weight updates."""

    if not callable(getattr(runtime, "update_weights", None)):
        return None
    if getattr(runtime, "weight_sync", None) is None:
        return None
    return RayRuntimeWeightSyncer(
        runtime,
        initial_policy_version=initial_policy_version,
    )


def _resolve_next_policy_version(
    runtime: Any,
    initial_policy_version: int | None,
) -> int:
    if initial_policy_version is not None:
        return int(initial_policy_version) + 1
    current = getattr(runtime, "current_policy_version", None)
    if current is None:
        return 1
    return int(current) + 1


def _cpu_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {key: _to_cpu(value) for key, value in state_dict.items()}


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_to_cpu(inner) for inner in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(inner) for inner in value)
    return value
