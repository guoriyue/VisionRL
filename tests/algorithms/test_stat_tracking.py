"""Tests for vrl.algorithms.stat_tracking (PerPromptStatTracker, sft/dpo/rwr methods)."""

from __future__ import annotations

import pytest

from vrl.algorithms.stat_tracking import PerPromptStatTracker


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
# PerPromptStatTracker — sft/dpo methods (Gap 6)
# ---------------------------------------------------------------------------

class TestStatTrackerSftDpo:
    def test_sft_picks_best(self) -> None:
        """SFT method: 1.0 for best sample, 0.0 for others."""
        tracker = PerPromptStatTracker()
        prompts = ["a", "a", "a"]
        rewards = [1.0, 5.0, 3.0]
        adv = tracker.update(prompts, rewards, method="sft")
        # Only index 1 (reward=5.0, the max) should be 1.0
        assert adv[0] == pytest.approx(0.0)
        assert adv[1] == pytest.approx(1.0)
        assert adv[2] == pytest.approx(0.0)

    def test_sft_ties(self) -> None:
        """SFT with tied best: all tied winners get 1.0."""
        tracker = PerPromptStatTracker()
        prompts = ["a", "a", "a"]
        rewards = [5.0, 5.0, 3.0]
        adv = tracker.update(prompts, rewards, method="sft")
        assert adv[0] == pytest.approx(1.0)
        assert adv[1] == pytest.approx(1.0)
        assert adv[2] == pytest.approx(0.0)

    def test_dpo_best_worst(self) -> None:
        """DPO method: +1 for best, -1 for worst, 0 for middle."""
        tracker = PerPromptStatTracker()
        prompts = ["a", "a", "a"]
        rewards = [1.0, 5.0, 3.0]
        adv = tracker.update(prompts, rewards, method="dpo")
        assert adv[0] == pytest.approx(-1.0)  # worst
        assert adv[1] == pytest.approx(1.0)   # best
        assert adv[2] == pytest.approx(0.0)   # middle

    def test_dpo_two_samples(self) -> None:
        """DPO with exactly 2 samples: one gets +1, other gets -1."""
        tracker = PerPromptStatTracker()
        prompts = ["a", "a"]
        rewards = [2.0, 8.0]
        adv = tracker.update(prompts, rewards, method="dpo")
        assert adv[0] == pytest.approx(-1.0)
        assert adv[1] == pytest.approx(1.0)

    def test_rwr_returns_raw(self) -> None:
        """RWR method returns raw rewards as advantages."""
        tracker = PerPromptStatTracker()
        prompts = ["a", "a"]
        rewards = [3.0, 7.0]
        adv = tracker.update(prompts, rewards, method="rwr")
        assert adv[0] == pytest.approx(3.0)
        assert adv[1] == pytest.approx(7.0)

    def test_multi_prompt_sft(self) -> None:
        """SFT method across multiple prompts."""
        tracker = PerPromptStatTracker()
        prompts = ["a", "a", "b", "b"]
        rewards = [1.0, 3.0, 10.0, 5.0]
        adv = tracker.update(prompts, rewards, method="sft")
        # For "a": best=3.0 → index 1 gets 1.0
        assert adv[0] == pytest.approx(0.0)
        assert adv[1] == pytest.approx(1.0)
        # For "b": best=10.0 → index 2 gets 1.0
        assert adv[2] == pytest.approx(1.0)
        assert adv[3] == pytest.approx(0.0)
