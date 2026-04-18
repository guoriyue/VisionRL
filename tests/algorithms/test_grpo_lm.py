"""Tests for vrl.algorithms.grpo_lm.TokenGRPO.

Validates:
  * Inherits group-relative advantage from GRPO unchanged.
  * Per-token PPO clipped surrogate broadcasts [B] advantages → [B, L].
  * Mask zero → token excluded from loss.
  * KL k1 vs k3 estimators.
  * Loss == 0 when ratio==1 and KL==0.
"""

from __future__ import annotations

import math

import pytest
import torch

from vrl.algorithms.grpo_lm import TokenGRPO, TokenGRPOConfig
from vrl.rollouts.evaluators.types import SignalBatch


# ---------------------------------------------------------------------------
# Advantage path — should be byte-identical to GRPO
# ---------------------------------------------------------------------------


class TestAdvantageInheritance:
    def test_matches_diffusion_grpo(self) -> None:
        from vrl.algorithms.grpo import GRPO, GRPOConfig

        torch.manual_seed(0)
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        groups = torch.tensor([0, 0, 0, 1, 1, 1])

        diff_adv = GRPO(GRPOConfig(global_std=False)).compute_advantages_from_tensors(
            rewards, groups,
        )
        tok_adv = TokenGRPO(TokenGRPOConfig(global_std=False)).compute_advantages_from_tensors(
            rewards, groups,
        )
        assert torch.allclose(diff_adv, tok_adv)


# ---------------------------------------------------------------------------
# Per-token loss
# ---------------------------------------------------------------------------


def _signals(new_lp: torch.Tensor, ref_lp: torch.Tensor | None = None,
             mask: torch.Tensor | None = None) -> SignalBatch:
    aux = {}
    if mask is not None:
        aux["token_mask"] = mask
    return SignalBatch(
        log_prob=new_lp,
        ref_log_prob=ref_lp,
        dist_family="categorical",
        aux=aux,
    )


class TestPerTokenLoss:
    def test_zero_loss_when_ratio_one_no_kl(self) -> None:
        """new_lp == old_lp → ratio == 1; positive advantage → loss = -adv."""
        old_lp = torch.zeros(2, 4)
        new_lp = old_lp.clone()
        adv = torch.zeros(2)
        algo = TokenGRPO(TokenGRPOConfig(kl_coeff=0.0))
        loss, _ = algo.compute_signal_loss(_signals(new_lp), adv, old_lp)
        assert loss.abs().item() < 1e-6

    def test_positive_advantage_decreases_loss_when_logprob_up(self) -> None:
        old_lp = torch.zeros(2, 4)
        new_lp = old_lp + 0.1     # log-prob went up
        adv = torch.ones(2)        # positive advantage
        algo = TokenGRPO(TokenGRPOConfig(kl_coeff=0.0, clip_eps=1.0))  # disable clip
        loss, _ = algo.compute_signal_loss(_signals(new_lp), adv, old_lp)
        # ratio = exp(0.1) > 1; -adv * ratio < -1
        assert loss.item() < -1.0

    def test_clip_activates(self) -> None:
        old_lp = torch.zeros(1, 4)
        new_lp = old_lp + 5.0     # huge ratio
        adv = torch.ones(1)
        algo = TokenGRPO(TokenGRPOConfig(kl_coeff=0.0, clip_eps=0.2))
        loss, metrics = algo.compute_signal_loss(_signals(new_lp), adv, old_lp)
        assert metrics.clip_fraction == 1.0   # all 4 tokens clipped


# ---------------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------------


class TestMask:
    def test_zero_mask_zeros_loss(self) -> None:
        old_lp = torch.zeros(2, 4)
        new_lp = old_lp + 1.0
        adv = torch.ones(2)
        mask = torch.zeros_like(new_lp)        # mask everything out
        algo = TokenGRPO(TokenGRPOConfig(kl_coeff=0.0))
        loss, _ = algo.compute_signal_loss(_signals(new_lp, mask=mask), adv, old_lp)
        # mask sum is clamped to 1.0 to avoid NaN, but per_token_loss * 0 = 0
        assert loss.abs().item() < 1e-6

    def test_partial_mask_changes_loss(self) -> None:
        old_lp = torch.zeros(1, 4)
        new_lp = torch.tensor([[0.0, 0.5, 0.5, 0.0]])
        adv = torch.ones(1)
        mask_full = torch.ones_like(new_lp)
        mask_half = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
        algo = TokenGRPO(TokenGRPOConfig(kl_coeff=0.0, clip_eps=10.0))
        l_full, _ = algo.compute_signal_loss(_signals(new_lp, mask=mask_full), adv, old_lp)
        l_half, _ = algo.compute_signal_loss(_signals(new_lp, mask=mask_half), adv, old_lp)
        # half mask only counts the +0.5 tokens → mean is more negative
        assert l_half.item() < l_full.item()


# ---------------------------------------------------------------------------
# KL estimators
# ---------------------------------------------------------------------------


class TestKL:
    def test_k1_signed(self) -> None:
        new_lp = torch.zeros(1, 4) + 0.5
        ref_lp = torch.zeros(1, 4)
        old_lp = torch.zeros(1, 4)
        adv = torch.zeros(1)
        algo = TokenGRPO(TokenGRPOConfig(kl_coeff=1.0, kl_estimator="k1"))
        _, m = algo.compute_signal_loss(
            _signals(new_lp, ref_lp=ref_lp), adv, old_lp,
        )
        # k1 = log_ratio = 0.5
        assert m.kl_penalty == pytest.approx(0.5, abs=1e-5)

    def test_k3_nonnegative(self) -> None:
        torch.manual_seed(1)
        new_lp = torch.randn(2, 4)
        ref_lp = torch.randn(2, 4)
        old_lp = new_lp.clone()
        adv = torch.zeros(2)
        algo = TokenGRPO(TokenGRPOConfig(kl_coeff=1.0, kl_estimator="k3"))
        _, m = algo.compute_signal_loss(
            _signals(new_lp, ref_lp=ref_lp), adv, old_lp,
        )
        assert m.kl_penalty >= 0.0

    def test_unknown_estimator_raises(self) -> None:
        new_lp = torch.zeros(1, 2)
        ref_lp = torch.zeros(1, 2)
        algo = TokenGRPO(TokenGRPOConfig(kl_coeff=1.0, kl_estimator="bogus"))
        with pytest.raises(ValueError, match="kl_estimator"):
            algo.compute_signal_loss(
                _signals(new_lp, ref_lp=ref_lp), torch.zeros(1), new_lp,
            )


# ---------------------------------------------------------------------------
# Shape errors
# ---------------------------------------------------------------------------


class TestShapeValidation:
    def test_mismatched_logprob_shape(self) -> None:
        new_lp = torch.zeros(2, 4)
        old_lp = torch.zeros(2, 5)
        adv = torch.zeros(2)
        algo = TokenGRPO(TokenGRPOConfig(kl_coeff=0.0))
        with pytest.raises(ValueError, match="log_prob shape"):
            algo.compute_signal_loss(_signals(new_lp), adv, old_lp)
