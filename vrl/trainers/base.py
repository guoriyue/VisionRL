"""Trainer ABC — training loop contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from vrl.algorithms.types import TrainStepMetrics


class Trainer(ABC):
    """Base class for RL trainers."""

    @abstractmethod
    async def step(self) -> TrainStepMetrics:
        """Run one training step (collect → reward → advantage → update)."""

    @abstractmethod
    def state_dict(self) -> dict:
        """Return a serialisable snapshot of trainer state."""

    @abstractmethod
    def load_state_dict(self, state: dict, *, strict: bool = True) -> None:
        """Restore trainer state from a snapshot."""
