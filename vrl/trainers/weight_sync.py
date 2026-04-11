"""Weight synchronisation between trainer and inference workers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


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
