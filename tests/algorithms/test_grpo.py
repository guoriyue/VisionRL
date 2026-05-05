"""Tests for vrl.algorithms.grpo (GRPO advantages + loss, CEA tensor path)."""

from __future__ import annotations

import pytest

from vrl.algorithms import GRPO
from vrl.algorithms.grpo import GRPOConfig
from vrl.rollouts.evaluators.types import SignalBatch

# ---------------------------------------------------------------------------
# Regression: single-sample GRPO advantage must NOT be NaN
# ---------------------------------------------------------------------------

class TestGRPOSingleSampleNaN:
    def test_single_sample_returns_zero_not_nan(self) -> None:
        """Single sample per group → advantage = 0.0, NOT NaN."""
        import torch

        grpo = GRPO()
        rewards = torch.tensor([5.0])
        group_ids = torch.tensor([0])
        advantages = grpo.compute_advantages_from_tensors(rewards, group_ids)
        assert not torch.isnan(advantages).any(), \
            f"Got NaN advantages: {advantages}"
        assert advantages[0].item() == pytest.approx(0.0)

    def test_multiple_single_sample_groups(self) -> None:
        """Multiple groups each with 1 sample → all advantages = 0."""
        import torch

        grpo = GRPO()
        rewards = torch.tensor([1.0, 5.0, 10.0])
        group_ids = torch.tensor([0, 1, 2])  # each prompt has 1 sample
        advantages = grpo.compute_advantages_from_tensors(rewards, group_ids)
        assert not torch.isnan(advantages).any()
        assert torch.allclose(advantages, torch.zeros(3))

    def test_group_with_multiple_samples_works(self) -> None:
        """Group with multiple samples → proper normalization, no NaN."""
        import torch

        grpo = GRPO()
        rewards = torch.tensor([1.0, 3.0, 5.0, 7.0])
        group_ids = torch.tensor([0, 0, 0, 0])  # all same prompt
        advantages = grpo.compute_advantages_from_tensors(rewards, group_ids)
        assert not torch.isnan(advantages).any()
        # Mean=4, should be negative for 1,3 and positive for 5,7
        assert advantages[0] < 0
        assert advantages[3] > 0


class TestGRPOFlowMatchingKL:
    def test_flow_kl_ignores_dt_by_default_to_match_flow_grpo(self) -> None:
        import torch

        grpo = GRPO(GRPOConfig(init_kl_coef=1.0))
        signals = SignalBatch(
            log_prob=torch.zeros(2),
            ref_log_prob=torch.zeros(2),
            prev_sample_mean=torch.zeros(2, 1, 1, 1),
            ref_prev_sample_mean=torch.ones(2, 1, 1, 1),
            std_dev_t=torch.ones(2, 1, 1, 1),
            dt=torch.full((2, 1, 1, 1), 0.1),
            dist_family="flow_matching",
        )

        loss, metrics = grpo.compute_signal_loss(
            signals,
            advantages=torch.zeros(2),
            old_log_probs=torch.zeros(2),
        )

        assert loss.item() == pytest.approx(0.5)
        assert metrics.kl_penalty == pytest.approx(0.5)

    def test_flow_kl_can_use_dt_when_explicitly_configured(self) -> None:
        import torch

        grpo = GRPO(GRPOConfig(init_kl_coef=1.0, flow_kl_use_dt=True))
        signals = SignalBatch(
            log_prob=torch.zeros(2),
            ref_log_prob=torch.zeros(2),
            prev_sample_mean=torch.zeros(2, 1, 1, 1),
            ref_prev_sample_mean=torch.ones(2, 1, 1, 1),
            std_dev_t=torch.ones(2, 1, 1, 1),
            dt=torch.full((2, 1, 1, 1), 0.1),
            dist_family="flow_matching",
        )

        loss, metrics = grpo.compute_signal_loss(
            signals,
            advantages=torch.zeros(2),
            old_log_probs=torch.zeros(2),
        )

        assert loss.item() == pytest.approx(50.0)
        assert metrics.kl_penalty == pytest.approx(50.0)
