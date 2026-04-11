"""Reward functions for RL training."""

from vrl.rewards.base import RewardFunction
from vrl.rewards.composite import CompositeReward
from vrl.rewards.remote import RemoteRewardFunction

__all__ = [
    "CompositeReward",
    "RemoteRewardFunction",
    "RewardFunction",
]
