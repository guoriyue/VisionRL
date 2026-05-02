"""Tests for vrl.rewards.composite (CompositeReward)."""

from __future__ import annotations

import pytest

from vrl.algorithms import Rollout, Trajectory, TrajectoryStep
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


class _ConstantReward(RewardFunction):
    def __init__(self, value: float) -> None:
        self.value = value

    async def score(self, rollout: Rollout) -> float:
        return self.value


# ---------------------------------------------------------------------------
# Composite reward tests
# ---------------------------------------------------------------------------

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
