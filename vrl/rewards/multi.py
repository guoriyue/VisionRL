"""Multi-reward registry — weighted combination of named reward functions.

Ported from the multi_score() pattern in flow_grpo/rewards.py.
"""

from __future__ import annotations

from typing import Any

from vrl.algorithms.types import Rollout
from vrl.rewards.base import RewardFunction


# Registry of reward function factories.
# Each factory takes (device,) and returns a RewardFunction instance.
_REWARD_REGISTRY: dict[str, type[RewardFunction]] = {}


def register_reward(name: str, cls: type[RewardFunction]) -> None:
    """Register a reward function class under a name."""
    _REWARD_REGISTRY[name] = cls


def get_reward(name: str) -> type[RewardFunction]:
    """Look up a registered reward function class by name."""
    if name not in _REWARD_REGISTRY:
        raise KeyError(f"Unknown reward function: {name!r}. Available: {list(_REWARD_REGISTRY)}")
    return _REWARD_REGISTRY[name]


def _register_builtins() -> None:
    """Register the built-in reward functions (lazy to avoid import errors)."""
    from vrl.rewards.aesthetic import AestheticReward
    from vrl.rewards.clip import CLIPScoreReward
    from vrl.rewards.pickscore import PickScoreReward

    register_reward("aesthetic", AestheticReward)
    register_reward("clipscore", CLIPScoreReward)
    register_reward("pickscore", PickScoreReward)


class MultiReward(RewardFunction):
    """Weighted combination of named reward functions.

    Usage::

        reward_fn = MultiReward.from_dict(
            {"pickscore": 1.0, "aesthetic": 0.5},
            device="cuda",
        )
    """

    def __init__(self, rewards: list[tuple[float, RewardFunction]]) -> None:
        self.rewards = rewards

    @classmethod
    def from_dict(cls, score_dict: dict[str, float], device: str = "cuda", **kwargs: Any) -> MultiReward:
        """Build from ``{"name": weight}`` dict, like flow_grpo config.reward_fn."""
        _register_builtins()
        pairs: list[tuple[float, RewardFunction]] = []
        for name, weight in score_dict.items():
            reward_cls = get_reward(name)
            pairs.append((weight, reward_cls(device=device, **kwargs)))
        return cls(pairs)

    async def score(self, rollout: Rollout) -> float:
        total = 0.0
        for weight, fn in self.rewards:
            total += weight * await fn.score(rollout)
        return total

    async def score_batch(self, rollouts: list[Rollout]) -> list[float]:
        totals = [0.0] * len(rollouts)
        for weight, fn in self.rewards:
            sub_scores = await fn.score_batch(rollouts)
            for i, s in enumerate(sub_scores):
                totals[i] += weight * s
        return totals
