"""Tests for vrl.algorithms (GRPO advantages + loss) and vrl.rewards (composite)."""

from __future__ import annotations

import math

import pytest

from vrl.algorithms import GRPO, GRPOConfig, Rollout, RolloutBatch, RolloutGroup, Trajectory, TrajectoryStep
from vrl.algorithms.stat_tracking import PerPromptStatTracker
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
# PerPromptStatTracker tests
# ---------------------------------------------------------------------------

class TestPerPromptStatTracker:
    def test_basic_grpo(self) -> None:
        tracker = PerPromptStatTracker(global_std=False)
        prompts = ["a", "a", "b", "b"]
        rewards = [1.0, 3.0, 10.0, 20.0]
        advantages = tracker.update(prompts, rewards)

        # For prompt "a": mean=2, std=1 → advantages = [-1, 1]
        assert advantages[0] == pytest.approx(-1.0, abs=0.01)
        assert advantages[1] == pytest.approx(1.0, abs=0.01)

        # For prompt "b": mean=15, std=5 → advantages = [-1, 1]
        assert advantages[2] == pytest.approx(-1.0, abs=0.01)
        assert advantages[3] == pytest.approx(1.0, abs=0.01)

    def test_global_std(self) -> None:
        tracker = PerPromptStatTracker(global_std=True)
        prompts = ["a", "a", "b", "b"]
        rewards = [1.0, 3.0, 10.0, 20.0]
        advantages = tracker.update(prompts, rewards)

        # With global_std, std is computed over all 4 rewards
        # So advantages for "a" should be smaller than with per-group std
        assert abs(advantages[0]) < 1.0

    def test_stats(self) -> None:
        tracker = PerPromptStatTracker()
        tracker.update(["a", "b", "a"], [1.0, 2.0, 3.0])
        avg_group, n_prompts = tracker.get_stats()
        assert n_prompts == 2
        assert avg_group > 0

    def test_clear(self) -> None:
        tracker = PerPromptStatTracker()
        tracker.update(["a"], [1.0])
        tracker.clear()
        assert tracker.stats == {}


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


# ---------------------------------------------------------------------------
# DistributedKRepeatSampler tests
# ---------------------------------------------------------------------------

class TestDistributedKRepeatSampler:
    def test_k_repeat_distribution(self) -> None:
        from torch.utils.data import TensorDataset
        import torch
        from vrl.trainers.data import DistributedKRepeatSampler

        dataset = TensorDataset(torch.arange(100))
        sampler = DistributedKRepeatSampler(
            dataset=dataset, batch_size=6, k=3, num_replicas=2, rank=0, seed=42
        )
        it = iter(sampler)
        batch = next(it)
        assert len(batch) == 6

    def test_rank_sync(self) -> None:
        """Both ranks should see the same unique prompts."""
        from torch.utils.data import TensorDataset
        import torch
        from vrl.trainers.data import DistributedKRepeatSampler

        dataset = TensorDataset(torch.arange(100))
        s0 = DistributedKRepeatSampler(dataset=dataset, batch_size=4, k=2, num_replicas=2, rank=0, seed=0)
        s1 = DistributedKRepeatSampler(dataset=dataset, batch_size=4, k=2, num_replicas=2, rank=1, seed=0)
        b0 = next(iter(s0))
        b1 = next(iter(s1))
        # Together they should have 8 items from 4 unique indices, each repeated 2x
        all_indices = b0 + b1
        assert len(all_indices) == 8
        unique = set(all_indices)
        assert len(unique) == 4
