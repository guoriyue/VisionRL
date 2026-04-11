"""GRPO — Group Relative Policy Optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from vrl.algorithms.base import Algorithm
from vrl.algorithms.types import (
    Advantages,
    RolloutBatch,
    RolloutGroup,
    TrainStepMetrics,
)


@dataclass(slots=True)
class GRPOConfig:
    """Hyper-parameters for GRPO."""

    clip_eps: float = 0.2
    kl_coeff: float = 0.0
    eps: float = 1e-8
    adv_clip_max: float = 5.0
    global_std: bool = False


class GRPO(Algorithm):
    """Group Relative Policy Optimization.

    Advantages are normalised within each prompt group:
        â_i = (r_i - mean(r)) / max(std(r), eps)

    Loss is the clipped surrogate objective (PPO-style) applied to
    pre-computed log-probabilities stored in ``TrajectoryStep.log_prob``.
    """

    def __init__(self, config: GRPOConfig | None = None) -> None:
        self.config = config or GRPOConfig()

    # ------------------------------------------------------------------
    # Advantages
    # ------------------------------------------------------------------

    def compute_advantages(
        self,
        group: RolloutGroup,
        global_rewards: list[float] | None = None,
    ) -> Advantages:
        """Compute per-rollout advantages for a single prompt group.

        When ``global_std=True`` (or ``global_rewards`` is provided), the
        standard deviation is computed across all rewards in the batch,
        not just within this group.  This matches the flow_grpo
        ``PerPromptStatTracker(global_std=True)`` behavior.
        """
        rewards = [r.reward for r in group.rollouts]
        n = len(rewards)
        if n == 0:
            return Advantages(values=[], method="grpo")

        mean = sum(rewards) / n

        if self.config.global_std and global_rewards is not None:
            g_n = len(global_rewards)
            g_mean = sum(global_rewards) / max(g_n, 1)
            g_var = sum((r - g_mean) ** 2 for r in global_rewards) / max(g_n, 1)
            std = math.sqrt(g_var)
        else:
            var = sum((r - mean) ** 2 for r in rewards) / max(n, 1)
            std = math.sqrt(var)

        denom = max(std, self.config.eps)
        values = [(r - mean) / denom for r in rewards]

        # Clip advantages (from flow_grpo: torch.clamp(adv, -adv_clip_max, adv_clip_max))
        clip = self.config.adv_clip_max
        values = [max(-clip, min(clip, v)) for v in values]

        return Advantages(
            values=values,
            method="grpo",
            stats={"reward_mean": mean, "reward_std": std},
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        batch: RolloutBatch,
        policy: Any,
        ref_policy: Any = None,
    ) -> tuple[Any, TrainStepMetrics]:
        """Clipped surrogate loss over all rollout groups.

        Expects each ``TrajectoryStep`` to carry:
        - ``log_prob``: log π_old(a|s) recorded during rollout collection
        - a corresponding new log-prob accessible as ``step.log_prob``
          (the caller should have refreshed log-probs under the current
          policy before calling this method).

        Returns a scalar loss (float) and ``TrainStepMetrics``.
        """
        cfg = self.config
        total_loss = 0.0
        total_policy_loss = 0.0
        total_kl = 0.0
        total_steps = 0
        clip_count = 0

        all_rewards: list[float] = []
        all_advantages: list[float] = []

        for group in batch.groups:
            if group.advantages is None:
                continue
            for rollout, adv in zip(group.rollouts, group.advantages.values):
                all_rewards.append(rollout.reward)
                all_advantages.append(adv)
                for step in rollout.trajectory.steps:
                    old_lp = step.log_prob
                    # new_log_prob is refreshed by the trainer before calling
                    # compute_loss; fall back to old_lp (ratio=1).
                    new_lp = step.new_log_prob if step.new_log_prob is not None else old_lp

                    ratio = math.exp(new_lp - old_lp)
                    clipped = max(min(ratio, 1.0 + cfg.clip_eps), 1.0 - cfg.clip_eps)
                    surrogate = min(ratio * adv, clipped * adv)
                    total_policy_loss += -surrogate

                    if ratio != clipped:
                        clip_count += 1

                    # Optional KL penalty (π_old ‖ π_ref)
                    if cfg.kl_coeff > 0 and ref_policy is not None:
                        ref_lp = step.ref_log_prob if step.ref_log_prob is not None else old_lp
                        kl = old_lp - ref_lp
                        total_kl += cfg.kl_coeff * kl

                    total_steps += 1

        if total_steps > 0:
            total_policy_loss /= total_steps
            total_kl /= total_steps

        total_loss = total_policy_loss + total_kl

        n_rewards = len(all_rewards)
        reward_mean = sum(all_rewards) / max(n_rewards, 1)
        reward_var = (
            sum((r - reward_mean) ** 2 for r in all_rewards) / max(n_rewards, 1)
        )
        adv_mean = sum(all_advantages) / max(len(all_advantages), 1)

        metrics = TrainStepMetrics(
            loss=total_loss,
            policy_loss=total_policy_loss,
            kl_penalty=total_kl,
            reward_mean=reward_mean,
            reward_std=math.sqrt(reward_var),
            advantage_mean=adv_mean,
            clip_fraction=clip_count / max(total_steps, 1),
        )
        return total_loss, metrics
