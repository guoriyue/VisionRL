"""Per-prompt reward stat tracking for GRPO advantage computation.

Ported from flow_grpo/stat_tracking.py.  Tracks per-prompt reward
history and computes group-relative advantages with optional global std.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class PerPromptStatTracker:
    """Track per-prompt reward statistics for advantage normalization.

    When ``global_std=True``, the standard deviation is computed across
    *all* rewards in the batch (not just same-prompt), which is better
    for diverse prompt sets.
    """

    def __init__(self, global_std: bool = False) -> None:
        self.global_std = global_std
        self.stats: dict[str, list[float]] = {}
        self.history_prompts: set[int] = set()

    def update(
        self,
        prompts: list[str] | Any,
        rewards: list[float] | Any,
        method: str = "grpo",
    ) -> Any:
        """Compute advantages for the given prompts and rewards.

        Supported methods:
        - ``"grpo"``: (reward - mean) / std per prompt group
        - ``"rwr"``:  raw rewards (reward-weighted regression)

        Returns a numpy array of advantages, same shape as ``rewards``.
        """
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.zeros_like(rewards)

        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards.tolist())
            self.history_prompts.add(hash(prompt))

        for prompt in unique:
            self.stats[prompt] = list(np.stack(self.stats[prompt]))
            prompt_rewards = rewards[prompts == prompt]
            mean = np.mean(self.stats[prompt], axis=0, keepdims=True)
            if self.global_std:
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4
            else:
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4

            if method == "grpo":
                advantages[prompts == prompt] = (prompt_rewards - mean) / std
            elif method == "rwr":
                advantages[prompts == prompt] = prompt_rewards

        return advantages

    def get_stats(self) -> tuple[float, int]:
        """Return (average group size, number of unique prompts seen)."""
        avg = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0
        return avg, len(self.history_prompts)

    def clear(self) -> None:
        """Clear accumulated stats (call at end of each epoch)."""
        self.stats = {}
