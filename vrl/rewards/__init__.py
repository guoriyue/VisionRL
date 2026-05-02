"""Reward functions for RL training."""

from typing import Any

from vrl.rewards.base import RewardFunction
from vrl.rewards.composite import CompositeReward
from vrl.rewards.multi import MultiReward, get_reward, register_reward


def __getattr__(name: str) -> Any:
    if name == "RemoteReward":
        from vrl.rewards.remote import RemoteReward

        return RemoteReward
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "CompositeReward",
    "MultiReward",
    "RemoteReward",
    "RewardFunction",
    "get_reward",
    "register_reward",
]
