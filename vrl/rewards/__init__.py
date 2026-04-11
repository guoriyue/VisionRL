"""Reward functions for RL training."""

from vrl.rewards.base import RewardFunction
from vrl.rewards.composite import CompositeReward
from vrl.rewards.multi import MultiReward, get_reward, register_reward
from vrl.rewards.remote import RemoteReward

__all__ = [
    "CompositeReward",
    "MultiReward",
    "RemoteReward",
    "RewardFunction",
    "get_reward",
    "register_reward",
]
