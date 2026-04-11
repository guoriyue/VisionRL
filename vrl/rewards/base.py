"""RewardFunction ABC — async scoring of rollouts."""

from __future__ import annotations

from abc import ABC, abstractmethod

from vrl.algorithms.types import Rollout


class RewardFunction(ABC):
    """Base class for reward functions."""

    @abstractmethod
    async def score(self, rollout: Rollout) -> float:
        """Score a single rollout."""

    async def score_batch(self, rollouts: list[Rollout]) -> list[float]:
        """Score a batch of rollouts (default: sequential)."""
        return [await self.score(r) for r in rollouts]
