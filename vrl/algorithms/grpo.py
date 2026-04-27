"""GRPO — Group Relative Policy Optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vrl.algorithms.base import Algorithm
from vrl.algorithms.types import TrainStepMetrics
from vrl.rollouts.evaluators.types import SignalBatch


@dataclass(slots=True)
class GRPOConfig:
    """Hyper-parameters for GRPO."""

    eps_clip: float = 0.2
    init_kl_coef: float = 0.0
    eps: float = 1e-8
    adv_clip_max: float = 5.0
    global_std: bool = False


class GRPO(Algorithm):
    """Group Relative Policy Optimization.

    Advantages are normalised within each prompt group:
        a_i = (r_i - mean(r)) / max(std(r), eps)

    Loss is the clipped surrogate objective (PPO-style) applied to
    per-sample log-probabilities produced by the evaluator.
    """

    def __init__(self, config: GRPOConfig | None = None) -> None:
        self.config = config or GRPOConfig()
        # Diagnostic stash: last call's policy_loss and (init_kl_coef * kl_loss)
        # tensors, kept alive (not detached) for grad-split diagnostics in
        # the trainer. Set to None when not applicable.
        self._last_policy_loss_tensor: Any = None
        self._last_kl_term_tensor: Any = None

    # ------------------------------------------------------------------
    # Advantages from tensors
    # ------------------------------------------------------------------

    def compute_advantages_from_tensors(
        self,
        rewards: Any,     # [B] tensor
        group_ids: Any,   # [B] tensor
    ) -> Any:
        """Per-group advantage normalization on tensors.

        Groups are identified by ``group_ids`` — samples sharing the same
        group_id are normalized together (GRPO per-prompt normalization).
        """
        import torch

        cfg = self.config
        advantages = torch.zeros_like(rewards)
        unique_groups = torch.unique(group_ids)

        for gid in unique_groups:
            mask = group_ids == gid
            group_rewards = rewards[mask]

            # Single sample → advantage is 0 (no group contrast possible)
            if group_rewards.numel() <= 1:
                advantages[mask] = 0.0
                continue

            mean = group_rewards.mean()

            if cfg.global_std:
                std = rewards.std() if rewards.numel() > 1 else torch.tensor(0.0)
            else:
                std = group_rewards.std()

            denom = torch.clamp(std, min=cfg.eps)
            group_adv = (group_rewards - mean) / denom
            group_adv = torch.clamp(group_adv, -cfg.adv_clip_max, cfg.adv_clip_max)
            advantages[mask] = group_adv

        return advantages

    # ------------------------------------------------------------------
    # Loss from SignalBatch
    # ------------------------------------------------------------------

    def compute_signal_loss(
        self,
        signals: SignalBatch,
        advantages: Any,       # [B] advantages
        old_log_probs: Any,    # [B] old log-probs from collection
    ) -> tuple[Any, TrainStepMetrics]:
        """Clipped surrogate loss from evaluator signals.

        Handles both flow-matching (latent-space KL) and generic (log-prob KL).
        """
        import torch

        from vrl.rollouts.evaluators.diffusion.flow_matching import compute_kl_divergence

        cfg = self.config

        ratio = torch.exp(signals.log_prob - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1.0 - cfg.eps_clip, 1.0 + cfg.eps_clip)
        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * clipped_ratio
        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

        # KL penalty. Hard-error when init_kl_coef > 0 but ref_log_prob is
        # missing — otherwise KL silently becomes 0 and the user thinks
        # regularization is active when it isn't.
        if cfg.init_kl_coef > 0:
            if signals.ref_log_prob is None:
                raise RuntimeError(
                    f"GRPOConfig.init_kl_coef={cfg.init_kl_coef} > 0 but "
                    "signals.ref_log_prob is None. Check: (1) ref_model "
                    "passed to OnlineTrainer, (2) SignalRequest(need_ref=True) "
                    "in the evaluator call."
                )
            if (
                signals.dist_family == "flow_matching"
                and signals.prev_sample_mean is not None
                and signals.ref_prev_sample_mean is not None
            ):
                # Latent-space KL (more principled for continuous diffusion)
                kl = compute_kl_divergence(
                    signals.prev_sample_mean,
                    signals.ref_prev_sample_mean,
                    signals.std_dev_t,
                    signals.dt,
                )
                kl_loss = torch.mean(kl)
            else:
                # Log-prob KL fallback
                kl_loss = torch.mean(signals.log_prob - signals.ref_log_prob)
            kl_term = cfg.init_kl_coef * kl_loss
            loss = policy_loss + kl_term
            self._last_kl_term_tensor = kl_term
        else:
            kl_loss = torch.tensor(0.0, device=signals.log_prob.device)
            loss = policy_loss
            self._last_kl_term_tensor = None

        self._last_policy_loss_tensor = policy_loss

        # Metrics
        clip_fraction = torch.mean((torch.abs(ratio - 1.0) > cfg.clip_eps).float()).item()
        approx_kl = 0.5 * torch.mean((signals.log_prob - old_log_probs) ** 2).item()

        metrics = TrainStepMetrics(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            kl_penalty=kl_loss.item(),
            clip_fraction=clip_fraction,
            approx_kl=approx_kl,
        )

        return loss, metrics
