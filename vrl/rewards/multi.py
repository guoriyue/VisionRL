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
    from vrl.rewards.ocr import OCRReward
    from vrl.rewards.pickscore import PickScoreReward

    register_reward("aesthetic", AestheticReward)
    register_reward("clipscore", CLIPScoreReward)
    register_reward("ocr", OCRReward)
    register_reward("pickscore", PickScoreReward)


class MultiReward(RewardFunction):
    """Weighted combination of named reward functions.

    Tracks per-component raw scores on the most recent call so the training
    loop can log them — essential for spotting reward hacking early (e.g.
    aesthetic collapses while OCR climbs).

    Usage::

        reward_fn = MultiReward.from_dict(
            {"ocr": 1.0, "aesthetic": 0.3},
            device="cuda",
        )
        total = await reward_fn.score(rollout)
        # reward_fn.last_components -> {"ocr": [0.87], "aesthetic": [5.2]}
    """

    def __init__(
        self,
        rewards: list[tuple[str, float, RewardFunction]],
    ) -> None:
        self.rewards = rewards
        self.last_components: dict[str, list[float]] = {}

    @classmethod
    def from_dict(
        cls,
        score_dict: dict[str, float],
        device: str = "cuda",
        reward_kwargs: dict[str, dict[str, Any]] | None = None,
    ) -> MultiReward:
        """Build from ``{"name": weight}`` dict, like flow_grpo config.reward_fn.

        ``reward_kwargs`` allows passing per-reward init kwargs, keyed by name,
        e.g. ``{"ocr": {"debug_dir": "out/ocr_debug"}}``.
        """
        _register_builtins()
        reward_kwargs = reward_kwargs or {}
        triples: list[tuple[str, float, RewardFunction]] = []
        for name, weight in score_dict.items():
            reward_cls = get_reward(name)
            extra = reward_kwargs.get(name, {})
            triples.append((name, weight, reward_cls(device=device, **extra)))
        return cls(triples)

    async def score(self, rollout: Rollout) -> float:
        total = 0.0
        components: dict[str, list[float]] = {}
        for name, weight, fn in self.rewards:
            s = await fn.score(rollout)
            components[name] = [s]
            total += weight * s
        self.last_components = components
        return total

    async def score_batch(self, rollouts: list[Rollout]) -> list[float]:
        totals = [0.0] * len(rollouts)
        components: dict[str, list[float]] = {}
        for name, weight, fn in self.rewards:
            sub_scores = await fn.score_batch(rollouts)
            components[name] = list(sub_scores)
            for i, s in enumerate(sub_scores):
                totals[i] += weight * s
        self.last_components = components
        return totals
