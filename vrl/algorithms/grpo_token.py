"""Token-level GRPO for autoregressive image / text generation.

Re-uses everything in ``vrl.algorithms.grpo.GRPO`` *except* the loss
function — a per-token PPO clipped surrogate that properly broadcasts
sequence-level advantages across the token dimension and supports a
``[B, L]`` mask for "score-only-image-token" semantics.

This is the algorithm half of the Janus-Pro RL closed loop. Pair with
``vrl.rollouts.collectors.janus_pro.JanusProCollector`` and
``vrl.rollouts.evaluators.ar.token_logprob.TokenLogProbEvaluator``.

Why subclass instead of fork?
    Advantage normalisation (``compute_advantages_from_tensors``) is
    100% identical — group-relative mean/std over per-prompt clusters.
    The only AR-specific deviation is per-token loss reduction. By
    inheriting we guarantee both algorithms stay in lock-step on the
    advantage side, which is the half that's been load-bearing in
    every flow_grpo experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vrl.algorithms.grpo import GRPO, GRPOConfig
from vrl.algorithms.types import TrainStepMetrics
from vrl.rollouts.evaluators.types import SignalBatch


@dataclass(slots=True)
class TokenGRPOConfig(GRPOConfig):
    """Token-level GRPO hyper-parameters.

    Extends ``GRPOConfig`` so anything that already accepts a ``GRPOConfig``
    keeps working unchanged.

    Extra knobs:
      ``mask_key``        — ``SignalBatch.aux`` key holding a ``[B, L]``
                            float mask (1 = score this token, 0 = ignore).
                            Used to score only image tokens, not the
                            text-prompt tokens, in Janus rollouts.
      ``kl_estimator``    — ``"k1"`` (logp - logp_ref, signed) or
                            ``"k3"`` (Schulman 2020 unbiased KL,
                            ``ratio * (logr - 1) + 1``). k3 is the trl
                            default for token RLHF — strictly non-negative,
                            tracks PPO behaviour better.
    """

    mask_key: str = "token_mask"
    kl_estimator: str = "k3"


class TokenGRPO(GRPO):
    """GRPO with per-token PPO loss for autoregressive policies.

    Forward contract (CEA path):
      * ``signals.log_prob``      shape ``[B, L]`` — fresh log-probs of
                                  *sampled* tokens under the current policy.
      * ``signals.ref_log_prob``  shape ``[B, L]`` — same under the LoRA-off
                                  reference policy. Optional (only when
                                  ``init_kl_coef > 0``).
      * ``old_log_probs``         shape ``[B, L]`` — log-probs captured at
                                  collection time (from
                                  ``JanusProPolicy.sample_image_tokens``).
      * ``advantages``            shape ``[B]``  — sequence-level advantages
                                  from ``compute_advantages_from_tensors``.
      * ``signals.aux[mask_key]`` shape ``[B, L]`` — float mask over tokens
                                  that count toward the loss.
    """

    def __init__(self, config: TokenGRPOConfig | None = None) -> None:
        # Hand the parent a slice of our config that it understands. Using
        # the same instance is fine because TokenGRPOConfig is a subclass.
        cfg = config or TokenGRPOConfig()
        super().__init__(cfg)
        self.config: TokenGRPOConfig = cfg  # narrow type for mypy

    # ------------------------------------------------------------------
    # CEA pipeline: per-token clipped surrogate
    # ------------------------------------------------------------------

    def compute_signal_loss(
        self,
        signals: SignalBatch,
        advantages: Any,        # [B]
        old_log_probs: Any,     # [B, L]
    ) -> tuple[Any, TrainStepMetrics]:
        cfg = self.config

        new_lp: torch.Tensor = signals.log_prob          # [B, L]
        old_lp: torch.Tensor = old_log_probs              # [B, L]
        if new_lp.shape != old_lp.shape:
            raise ValueError(
                f"log_prob shape mismatch: new={tuple(new_lp.shape)} "
                f"old={tuple(old_lp.shape)}"
            )

        mask = signals.aux.get(cfg.mask_key) if signals.aux else None
        if mask is None:
            mask = torch.ones_like(new_lp)
        mask = mask.to(dtype=new_lp.dtype, device=new_lp.device)

        # Broadcast sequence-level advantage across the token dim.
        if advantages.dim() == 1:
            adv_bL = advantages.unsqueeze(-1).expand_as(new_lp)
        else:
            adv_bL = advantages

        # PPO clipped surrogate, per token.
        ratio = torch.exp(new_lp - old_lp)
        clipped_ratio = torch.clamp(ratio, 1.0 - cfg.eps_clip, 1.0 + cfg.eps_clip)
        unclipped_loss = -adv_bL * ratio
        clipped_loss = -adv_bL * clipped_ratio
        per_token_loss = torch.maximum(unclipped_loss, clipped_loss)

        # Mask + sequence-mean → batch-mean. Using sum-and-divide to keep
        # an empty mask from producing NaN.
        denom = mask.sum().clamp_min(1.0)
        policy_loss = (per_token_loss * mask).sum() / denom

        # KL penalty
        if cfg.init_kl_coef > 0 and signals.ref_log_prob is not None:
            ref_lp: torch.Tensor = signals.ref_log_prob
            log_ratio = new_lp - ref_lp
            if cfg.kl_estimator == "k3":
                # Schulman's positive, low-variance estimator
                kl_per_tok = torch.exp(log_ratio) * (log_ratio - 1.0) + 1.0
            elif cfg.kl_estimator == "k1":
                kl_per_tok = log_ratio
            else:
                raise ValueError(f"unknown kl_estimator: {cfg.kl_estimator}")
            kl_loss = (kl_per_tok * mask).sum() / denom
            loss = policy_loss + cfg.init_kl_coef * kl_loss
        else:
            kl_loss = torch.zeros((), device=new_lp.device)
            loss = policy_loss

        # Diagnostics — restrict to scored tokens for accuracy.
        with torch.no_grad():
            valid = mask > 0
            if valid.any():
                ratio_valid = ratio[valid]
                clip_fraction = (
                    (torch.abs(ratio_valid - 1.0) > cfg.eps_clip).float().mean().item()
                )
                approx_kl = 0.5 * ((new_lp - old_lp) ** 2)[valid].mean().item()
            else:
                clip_fraction = 0.0
                approx_kl = 0.0

        metrics = TrainStepMetrics(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            kl_penalty=kl_loss.item(),
            clip_fraction=clip_fraction,
            approx_kl=approx_kl,
        )
        return loss, metrics
