"""Tests for vrl.algorithms.grpo (GRPO advantages + loss)."""

from __future__ import annotations

import pytest

from vrl.algorithms import GRPO, GRPOConfig, Rollout, RolloutBatch, RolloutGroup, Trajectory, TrajectoryStep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rollout(prompt: str, reward: float, log_probs: list[float] | None = None) -> Rollout:
    log_probs = log_probs or [-1.0]
    steps = [TrajectoryStep(timestep=i, log_prob=lp, noise_pred=None) for i, lp in enumerate(log_probs)]
    traj = Trajectory(prompt=prompt, seed=42, steps=steps, output=None)
    return Rollout(request=None, trajectory=traj, reward=reward)


# ---------------------------------------------------------------------------
# GRPO advantage tests
# ---------------------------------------------------------------------------

class TestGRPOAdvantages:
    def test_zero_variance(self) -> None:
        grpo = GRPO()
        group = RolloutGroup(
            prompt="test",
            rollouts=[_make_rollout("test", 1.0), _make_rollout("test", 1.0)],
        )
        adv = grpo.compute_advantages(group)
        assert adv.method == "grpo"
        assert all(v == pytest.approx(0.0) for v in adv.values)

    def test_normalised_values(self) -> None:
        grpo = GRPO()
        group = RolloutGroup(
            prompt="test",
            rollouts=[
                _make_rollout("test", 1.0),
                _make_rollout("test", 3.0),
                _make_rollout("test", 5.0),
            ],
        )
        adv = grpo.compute_advantages(group)
        # mean=3, std≈1.633
        assert adv.values[0] < 0
        assert adv.values[1] == pytest.approx(0.0)
        assert adv.values[2] > 0
        # Should be symmetric around 0
        assert adv.values[0] == pytest.approx(-adv.values[2])

    def test_empty_group(self) -> None:
        grpo = GRPO()
        group = RolloutGroup(prompt="test", rollouts=[])
        adv = grpo.compute_advantages(group)
        assert adv.values == []

    def test_single_rollout(self) -> None:
        grpo = GRPO()
        group = RolloutGroup(prompt="test", rollouts=[_make_rollout("test", 5.0)])
        adv = grpo.compute_advantages(group)
        # std=0 → denom=eps, advantage = (5-5)/eps = 0
        assert adv.values[0] == pytest.approx(0.0)

    def test_stats_populated(self) -> None:
        grpo = GRPO()
        group = RolloutGroup(
            prompt="test",
            rollouts=[_make_rollout("test", 2.0), _make_rollout("test", 4.0)],
        )
        adv = grpo.compute_advantages(group)
        assert adv.stats["reward_mean"] == pytest.approx(3.0)
        assert adv.stats["reward_std"] == pytest.approx(1.0)

    def test_global_std(self) -> None:
        """With global_std=True, std comes from all rewards, not just the group."""
        cfg = GRPOConfig(global_std=True)
        grpo = GRPO(cfg)

        group = RolloutGroup(
            prompt="a",
            rollouts=[_make_rollout("a", 1.0), _make_rollout("a", 3.0)],
        )
        # Global rewards include a wider range
        global_rewards = [1.0, 3.0, 10.0, 20.0]
        adv = grpo.compute_advantages(group, global_rewards=global_rewards)

        # With global std >> group std, advantages should be smaller
        grpo_local = GRPO(GRPOConfig(global_std=False))
        adv_local = grpo_local.compute_advantages(group)

        assert abs(adv.values[0]) < abs(adv_local.values[0])

    def test_adv_clip(self) -> None:
        """Advantages should be clipped to [-adv_clip_max, adv_clip_max]."""
        cfg = GRPOConfig(adv_clip_max=1.0)
        grpo = GRPO(cfg)
        group = RolloutGroup(
            prompt="test",
            rollouts=[
                _make_rollout("test", 0.0),
                _make_rollout("test", 100.0),
            ],
        )
        adv = grpo.compute_advantages(group)
        assert all(-1.0 <= v <= 1.0 for v in adv.values)


# ---------------------------------------------------------------------------
# GRPO loss tests
# ---------------------------------------------------------------------------

class TestGRPOLoss:
    def test_no_clip_at_ratio_one(self) -> None:
        """When old_lp == new_lp, ratio=1, no clipping."""
        grpo = GRPO()
        rollouts = [_make_rollout("test", 1.0, [-0.5]), _make_rollout("test", 3.0, [-0.5])]
        group = RolloutGroup(prompt="test", rollouts=rollouts)
        group.advantages = grpo.compute_advantages(group)

        batch = RolloutBatch(groups=[group])
        loss, metrics = grpo.compute_loss(batch, policy=None)

        assert metrics.clip_fraction == pytest.approx(0.0)
        assert isinstance(loss, float)

    def test_kl_penalty(self) -> None:
        """KL penalty should increase the loss."""
        cfg_no_kl = GRPOConfig(kl_coeff=0.0)
        cfg_kl = GRPOConfig(kl_coeff=0.1)

        rollouts = [_make_rollout("test", 1.0, [-0.5]), _make_rollout("test", 3.0, [-1.0])]
        for r in rollouts:
            for s in r.trajectory.steps:
                s.ref_log_prob = s.log_prob - 0.5  # ref is different

        group = RolloutGroup(prompt="test", rollouts=rollouts)
        grpo_no_kl = GRPO(cfg_no_kl)
        grpo_kl = GRPO(cfg_kl)

        group.advantages = grpo_no_kl.compute_advantages(group)
        batch = RolloutBatch(groups=[group])

        _, m_no_kl = grpo_no_kl.compute_loss(batch, policy=None, ref_policy=None)
        _, m_kl = grpo_kl.compute_loss(batch, policy=None, ref_policy="dummy")

        assert m_no_kl.kl_penalty == pytest.approx(0.0)
        assert m_kl.kl_penalty > 0

    def test_metrics_fields(self) -> None:
        grpo = GRPO()
        rollouts = [_make_rollout("test", 2.0), _make_rollout("test", 4.0)]
        group = RolloutGroup(prompt="test", rollouts=rollouts)
        group.advantages = grpo.compute_advantages(group)

        batch = RolloutBatch(groups=[group])
        _, metrics = grpo.compute_loss(batch, policy=None)

        assert metrics.reward_mean == pytest.approx(3.0)
        assert metrics.reward_std > 0


# ---------------------------------------------------------------------------
# Regression: Bug 2 — single-sample GRPO advantage must NOT be NaN
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
