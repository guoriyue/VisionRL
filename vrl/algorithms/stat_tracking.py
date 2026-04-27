"""Per-prompt advantage statistics tracker.

Ported from flow_grpo's ``flow_grpo/stat_tracking.py``. Intended usage
mirrors flow_grpo's ``train_wan2_1.py:829`` pattern: caller invokes
``update()`` on the full gathered batch, reads advantages, then calls
``clear()`` before the next epoch. The tracker groups a single batch's
rewards by prompt and normalizes within each group — it is a per-step
helper, *not* a cross-epoch history pool.

Reference: flow_grpo/stat_tracking.py::PerPromptStatTracker
             flow_grpo/scripts/train_wan2_1.py:812-829 (call site)
"""

from __future__ import annotations

from typing import Any

import numpy as np


class PerPromptStatTracker:
    """Groups a batch's rewards by prompt and computes normalized advantages.

    Within a single ``update()`` call, samples are bucketed by prompt;
    per-bucket mean/std produces the GRPO advantage. The tracker keeps
    the grouped stats alive until ``clear()`` is called (flow_grpo
    invokes clear every outer epoch — see train_wan2_1.py:829).

    Advantage types:
      * ``grpo``  — ``(r - group_mean) / (group_std + 1e-4)``
      * ``rwr``   — identity (reward-weighted regression)
      * ``sft``   — one-hot on batch max
      * ``dpo``   — ±1 on batch max/min (fallback to idx 0/1 if tied)
    """

    def __init__(self, global_std: bool = False) -> None:
        self.global_std = global_std
        self.stats: dict[str, np.ndarray] = {}
        self.history_prompts: set[int] = set()

    def update(
        self,
        prompts: list[str] | np.ndarray,
        rewards: Any,
        type: str = "grpo",
    ) -> np.ndarray:
        """Accumulate reward history and return normalized advantages.

        ``prompts`` and ``rewards`` are per-sample and must align. Rewards
        may be a 1D array (scalar per sample) or 2D (per-timestep); mean
        and std are taken along axis 0.
        """
        import torch

        if isinstance(rewards, torch.Tensor):
            rewards = rewards.detach().cpu().numpy()
        prompts = np.array(prompts)
        rewards = np.asarray(rewards, dtype=np.float64)

        unique = np.unique(prompts)
        advantages = np.zeros_like(rewards)

        # First pass: append this batch's rewards into each prompt's history.
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            # stats may already be a np.ndarray from a prior update; treat as list.
            hist = self.stats[prompt]
            if isinstance(hist, np.ndarray):
                hist = list(hist)
            hist.extend(prompt_rewards)
            self.stats[prompt] = hist
            self.history_prompts.add(hash(prompt))

        # Second pass: compute per-prompt advantages from accumulated history.
        for prompt in unique:
            # Stack history into ndarray for consistent mean/std shape handling.
            self.stats[prompt] = np.stack(self.stats[prompt])
            prompt_rewards = rewards[prompts == prompt]
            mean = np.mean(self.stats[prompt], axis=0, keepdims=True)
            if self.global_std:
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4
            else:
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4

            if type == "grpo":
                advantages[prompts == prompt] = (prompt_rewards - mean) / std
            elif type == "rwr":
                advantages[prompts == prompt] = prompt_rewards
            elif type == "sft":
                pr_t = torch.tensor(prompt_rewards)
                advantages[prompts == prompt] = (pr_t == torch.max(pr_t)).float().numpy()
            elif type == "dpo":
                pr_t = torch.tensor(prompt_rewards)
                max_idx = int(torch.argmax(pr_t))
                min_idx = int(torch.argmin(pr_t))
                if max_idx == min_idx:
                    min_idx = 0
                    max_idx = 1
                result = np.zeros_like(prompt_rewards)
                result[max_idx] = 1.0
                result[min_idx] = -1.0
                advantages[prompts == prompt] = result
            else:
                raise ValueError(f"Unknown advantage type: {type}")

        return advantages

    def get_stats(self) -> tuple[float, int]:
        """Return (avg_group_size, num_unique_prompts_seen)."""
        if not self.stats:
            return 0.0, 0
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats)
        return avg_group_size, len(self.history_prompts)

    def clear(self) -> None:
        self.stats = {}
