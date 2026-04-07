"""Compatibility shim for consumer-side RL env adapters."""

from wm_infra.consumers.rl.env import GoalReward, WorldModelEnv, WorldModelVectorEnv

__all__ = ["GoalReward", "WorldModelEnv", "WorldModelVectorEnv"]
