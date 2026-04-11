"""Tests for vrl.algorithms (GRPO advantages + loss) and vrl.rewards (composite)."""

from __future__ import annotations

import math

import pytest

from vrl.algorithms import GRPO, GRPOConfig, Rollout, RolloutBatch, RolloutGroup, Trajectory, TrajectoryStep
from vrl.rewards.base import RewardFunction
from vrl.rewards.composite import CompositeReward


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
# Composite reward tests
# ---------------------------------------------------------------------------

class _ConstantReward(RewardFunction):
    def __init__(self, value: float) -> None:
        self.value = value

    async def score(self, rollout: Rollout) -> float:
        return self.value


@pytest.mark.asyncio
class TestCompositeReward:
    async def test_weighted_sum(self) -> None:
        comp = CompositeReward([
            (0.5, _ConstantReward(2.0)),
            (0.3, _ConstantReward(10.0)),
        ])
        r = _make_rollout("test", 0.0)
        score = await comp.score(r)
        assert score == pytest.approx(0.5 * 2.0 + 0.3 * 10.0)

    async def test_batch(self) -> None:
        comp = CompositeReward([
            (1.0, _ConstantReward(3.0)),
            (2.0, _ConstantReward(1.0)),
        ])
        rollouts = [_make_rollout("a", 0.0), _make_rollout("b", 0.0)]
        scores = await comp.score_batch(rollouts)
        assert len(scores) == 2
        assert all(s == pytest.approx(5.0) for s in scores)
