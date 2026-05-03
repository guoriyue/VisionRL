"""Tests for vrl.rollouts.evaluators (SDE step, FlowMatchingEvaluator, KL divergence)."""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# sde_step_with_logprob — CPS and noise_level (Gap 1)
# ---------------------------------------------------------------------------

class TestSDEStepWithLogprob:
    """Test sde_step_with_logprob with CPS and noise_level variants."""

    def _make_mock_scheduler(self):
        """Create a minimal mock scheduler for testing."""
        import torch

        class MockScheduler:
            def __init__(self):
                # 5 timesteps: [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
                self.sigmas = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])

            def index_for_timestep(self, t):
                # Simple mapping: t=0.8 → index 1, etc.
                diffs = (self.sigmas - t).abs()
                return diffs.argmin().item()

        return MockScheduler()

    def test_sde_type_default(self) -> None:
        """Standard SDE mode returns valid result."""
        import torch

        from vrl.algorithms.flow_matching import sde_step_with_logprob

        scheduler = self._make_mock_scheduler()
        B = 2
        model_output = torch.randn(B, 4, 8, 8)
        timestep = torch.tensor([0.8, 0.8])
        sample = torch.randn(B, 4, 8, 8)

        result = sde_step_with_logprob(
            scheduler, model_output, timestep, sample,
            sde_type="sde",
        )
        assert result.prev_sample.shape == sample.shape
        assert result.log_prob.shape == (B,)
        assert result.dt is None  # return_dt=False by default

    def test_cps_type(self) -> None:
        """CPS SDE type returns valid result with different math."""
        import torch

        from vrl.algorithms.flow_matching import sde_step_with_logprob

        scheduler = self._make_mock_scheduler()
        B = 2
        model_output = torch.randn(B, 4, 8, 8)
        timestep = torch.tensor([0.8, 0.8])
        sample = torch.randn(B, 4, 8, 8)

        result = sde_step_with_logprob(
            scheduler, model_output, timestep, sample,
            sde_type="cps",
            noise_level=1.0,
        )
        assert result.prev_sample.shape == sample.shape
        assert result.log_prob.shape == (B,)

    def test_noise_level_scales_output(self) -> None:
        """Different noise_level values produce different results in CPS."""
        import torch

        from vrl.algorithms.flow_matching import sde_step_with_logprob

        scheduler = self._make_mock_scheduler()
        B = 1
        model_output = torch.randn(B, 4, 8, 8)
        timestep = torch.tensor([0.8])
        sample = torch.randn(B, 4, 8, 8)

        # Fix generator for reproducibility
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)

        r1 = sde_step_with_logprob(
            scheduler, model_output, timestep, sample,
            sde_type="cps", noise_level=0.7, generator=gen1,
        )
        r2 = sde_step_with_logprob(
            scheduler, model_output, timestep, sample,
            sde_type="cps", noise_level=1.5, generator=gen2,
        )
        # Different noise_level → different prev_sample
        assert not torch.allclose(r1.prev_sample, r2.prev_sample)

    def test_deterministic_mode(self) -> None:
        """Deterministic mode: same input → same output, zero noise."""
        import torch

        from vrl.algorithms.flow_matching import sde_step_with_logprob

        scheduler = self._make_mock_scheduler()
        B = 1
        model_output = torch.randn(B, 4, 8, 8)
        timestep = torch.tensor([0.8])
        sample = torch.randn(B, 4, 8, 8)

        r1 = sde_step_with_logprob(
            scheduler, model_output, timestep, sample,
            deterministic=True, sde_type="sde",
        )
        r2 = sde_step_with_logprob(
            scheduler, model_output, timestep, sample,
            deterministic=True, sde_type="sde",
        )
        assert torch.allclose(r1.prev_sample, r2.prev_sample)

    def test_return_dt(self) -> None:
        """return_dt=True should populate the dt field."""
        import torch

        from vrl.algorithms.flow_matching import sde_step_with_logprob

        scheduler = self._make_mock_scheduler()
        B = 1
        model_output = torch.randn(B, 4, 8, 8)
        timestep = torch.tensor([0.8])
        sample = torch.randn(B, 4, 8, 8)

        result = sde_step_with_logprob(
            scheduler, model_output, timestep, sample,
            return_dt=True,
        )
        assert result.dt is not None

    def test_prev_sample_passthrough(self) -> None:
        """When prev_sample is given, the result should use it for log_prob calc."""
        import torch

        from vrl.algorithms.flow_matching import sde_step_with_logprob

        scheduler = self._make_mock_scheduler()
        B = 1
        model_output = torch.randn(B, 4, 8, 8)
        timestep = torch.tensor([0.8])
        sample = torch.randn(B, 4, 8, 8)
        prev_sample = torch.randn(B, 4, 8, 8)

        result = sde_step_with_logprob(
            scheduler, model_output, timestep, sample,
            prev_sample=prev_sample, sde_type="sde",
        )
        assert result.log_prob.shape == (B,)


# ---------------------------------------------------------------------------
# FlowMatchingEvaluator — init params (Gap 1, 7)
# ---------------------------------------------------------------------------

class TestFlowMatchingEvaluatorInit:
    def test_default_params(self) -> None:
        """FlowMatchingEvaluator accepts noise_level and sde_type."""
        from vrl.rollouts.evaluators.diffusion.flow_matching import FlowMatchingEvaluator

        evaluator = FlowMatchingEvaluator(scheduler=None)
        assert evaluator.noise_level == 1.0
        assert evaluator.sde_type == "sde"

    def test_custom_params(self) -> None:
        """FlowMatchingEvaluator can be configured for CPS."""
        from vrl.rollouts.evaluators.diffusion.flow_matching import FlowMatchingEvaluator

        evaluator = FlowMatchingEvaluator(
            scheduler=None, noise_level=0.7, sde_type="cps",
        )
        assert evaluator.noise_level == 0.7
        assert evaluator.sde_type == "cps"

    def test_evaluate_replays_through_policy_model(self, monkeypatch) -> None:
        """Diffusion evaluators require the model to own replay."""
        import types

        import torch

        import vrl.rollouts.evaluators.diffusion.flow_matching as fm
        from vrl.algorithms.flow_matching import SDEStepResult
        from vrl.rollouts.evaluators.types import SignalRequest

        class ReplayPolicy:
            def __init__(self) -> None:
                self.calls = 0

            def replay_forward(self, batch, timestep_idx):
                self.calls += 1
                return {
                    "noise_pred": torch.zeros_like(
                        batch.observations[:, timestep_idx],
                    ),
                }

        def fake_sde_step_with_logprob(
            scheduler, model_output, timestep, sample, **kwargs,
        ):
            del scheduler, timestep, kwargs
            return SDEStepResult(
                prev_sample=sample,
                log_prob=torch.zeros(sample.shape[0]),
                prev_sample_mean=model_output,
                std_dev_t=torch.ones(sample.shape[0]),
            )

        monkeypatch.setattr(
            fm.flow_matching_math,
            "sde_step_with_logprob",
            fake_sde_step_with_logprob,
        )

        policy = ReplayPolicy()
        batch = types.SimpleNamespace(
            observations=torch.randn(2, 1, 4),
            actions=torch.randn(2, 1, 4),
            extras={"timesteps": torch.tensor([[0.8], [0.8]])},
        )

        evaluator = fm.FlowMatchingEvaluator(scheduler=None)
        evaluator.evaluate(
            model=policy,
            batch=batch,
            timestep_idx=0,
            signal_request=SignalRequest(need_ref=False),
        )

        assert policy.calls == 1

    def test_evaluate_fails_when_model_does_not_own_replay(
        self, monkeypatch,
    ) -> None:
        """There is no collector.model or transformer-only compatibility path."""
        import types

        import torch

        import vrl.rollouts.evaluators.diffusion.flow_matching as fm
        from vrl.algorithms.flow_matching import SDEStepResult
        from vrl.rollouts.evaluators.types import SignalRequest

        def fake_sde_step_with_logprob(
            scheduler, model_output, timestep, sample, **kwargs,
        ):
            del scheduler, timestep, kwargs
            return SDEStepResult(
                prev_sample=sample,
                log_prob=torch.zeros(sample.shape[0]),
                prev_sample_mean=model_output,
                std_dev_t=torch.ones(sample.shape[0]),
            )

        monkeypatch.setattr(
            fm.flow_matching_math,
            "sde_step_with_logprob",
            fake_sde_step_with_logprob,
        )

        batch = types.SimpleNamespace(
            observations=torch.randn(2, 1, 4),
            actions=torch.randn(2, 1, 4),
            extras={"timesteps": torch.tensor([[0.8], [0.8]])},
        )

        evaluator = fm.FlowMatchingEvaluator(scheduler=None)
        with pytest.raises(AttributeError, match="replay_forward"):
            evaluator.evaluate(
                model=object(),
                batch=batch,
                timestep_idx=0,
                signal_request=SignalRequest(need_ref=False),
            )

    def test_ref_replay_uses_trainable_disable_adapter(self, monkeypatch) -> None:
        """LoRA KL replay must disable the adapter on the trainable module."""
        import contextlib
        import types

        import torch

        import vrl.rollouts.evaluators.diffusion.flow_matching as fm
        from vrl.algorithms.flow_matching import SDEStepResult
        from vrl.rollouts.evaluators.types import SignalRequest

        class ReplayPolicy:
            def __init__(self) -> None:
                self.disabled = False
                self.disabled_states: list[bool] = []

            @contextlib.contextmanager
            def disable_adapter(self):
                self.disabled = True
                try:
                    yield
                finally:
                    self.disabled = False

            def replay_forward(self, batch, timestep_idx):
                del timestep_idx
                self.disabled_states.append(self.disabled)
                return {"noise_pred": torch.zeros_like(batch.observations[:, 0])}

        def fake_sde_step_with_logprob(
            scheduler, model_output, timestep, sample, **kwargs,
        ):
            del scheduler, timestep
            return SDEStepResult(
                prev_sample=sample,
                log_prob=torch.zeros(sample.shape[0]),
                prev_sample_mean=model_output,
                std_dev_t=torch.ones(sample.shape[0]),
                dt=torch.ones(sample.shape[0])
                if kwargs.get("return_dt")
                else None,
            )

        monkeypatch.setattr(
            fm.flow_matching_math,
            "sde_step_with_logprob",
            fake_sde_step_with_logprob,
        )

        policy = ReplayPolicy()
        batch = types.SimpleNamespace(
            observations=torch.randn(2, 1, 4),
            actions=torch.randn(2, 1, 4),
            extras={"timesteps": torch.tensor([[0.8], [0.8]])},
        )

        evaluator = fm.FlowMatchingEvaluator(scheduler=None)
        signals = evaluator.evaluate(
            model=policy,
            batch=batch,
            timestep_idx=0,
            ref_model=policy,
            signal_request=SignalRequest(
                need_ref=True,
                need_kl_intermediates=True,
            ),
        )

        assert policy.disabled_states == [False, True]
        assert signals.ref_log_prob is not None

    def test_ref_replay_uses_ref_policy_directly(
        self, monkeypatch,
    ) -> None:
        """Explicit reference policies must own their own replay."""
        import contextlib
        import types

        import torch

        import vrl.rollouts.evaluators.diffusion.flow_matching as fm
        from vrl.algorithms.flow_matching import SDEStepResult
        from vrl.rollouts.evaluators.types import SignalRequest

        class ReplayPolicy:
            def __init__(self, name: str) -> None:
                self.name = name
                self.disabled = False
                self.calls: list[tuple[str, bool]] = []

            @contextlib.contextmanager
            def disable_adapter(self):
                self.disabled = True
                try:
                    yield
                finally:
                    self.disabled = False

            def replay_forward(self, batch, timestep_idx):
                self.calls.append((self.name, self.disabled))
                return {
                    "noise_pred": torch.zeros_like(
                        batch.observations[:, timestep_idx],
                    ),
                }

        def fake_sde_step_with_logprob(
            scheduler, model_output, timestep, sample, **kwargs,
        ):
            del scheduler, timestep
            return SDEStepResult(
                prev_sample=sample,
                log_prob=torch.zeros(sample.shape[0]),
                prev_sample_mean=model_output,
                std_dev_t=torch.ones(sample.shape[0]),
                dt=torch.ones(sample.shape[0])
                if kwargs.get("return_dt")
                else None,
            )

        monkeypatch.setattr(
            fm.flow_matching_math,
            "sde_step_with_logprob",
            fake_sde_step_with_logprob,
        )

        policy = ReplayPolicy("policy")
        ref_policy = ReplayPolicy("ref")
        batch = types.SimpleNamespace(
            observations=torch.randn(2, 1, 4),
            actions=torch.randn(2, 1, 4),
            extras={"timesteps": torch.tensor([[0.8], [0.8]])},
        )

        evaluator = fm.FlowMatchingEvaluator(scheduler=None)
        signals = evaluator.evaluate(
            model=policy,
            batch=batch,
            timestep_idx=0,
            ref_model=ref_policy,
            signal_request=SignalRequest(
                need_ref=True,
                need_kl_intermediates=True,
            ),
        )

        assert policy.calls == [("policy", False)]
        assert ref_policy.calls == [("ref", False)]
        assert signals.ref_log_prob is not None


# ---------------------------------------------------------------------------
# KL divergence helper
# ---------------------------------------------------------------------------

class TestComputeKLDivergence:
    def test_zero_when_same(self) -> None:
        """KL divergence is 0 when means are identical."""
        import torch

        from vrl.algorithms.flow_matching import compute_kl_divergence

        mean = torch.randn(2, 4, 8, 8)
        std = torch.ones(2, 1, 1, 1) * 0.5
        kl = compute_kl_divergence(mean, mean, std)
        assert torch.allclose(kl, torch.zeros(2))

    def test_positive_when_different(self) -> None:
        """KL divergence is positive when means differ."""
        import torch

        from vrl.algorithms.flow_matching import compute_kl_divergence

        mean1 = torch.randn(2, 4, 8, 8)
        mean2 = mean1 + 1.0  # shifted
        std = torch.ones(2, 1, 1, 1) * 0.5
        kl = compute_kl_divergence(mean1, mean2, std)
        assert (kl > 0).all()

    def test_with_dt(self) -> None:
        """KL with dt parameter scales the denominator."""
        import torch

        from vrl.algorithms.flow_matching import compute_kl_divergence

        mean1 = torch.randn(2, 4, 8, 8)
        mean2 = mean1 + 0.5
        std = torch.ones(2, 1, 1, 1) * 0.5
        dt = torch.ones(2, 1, 1, 1) * 0.1

        kl_no_dt = compute_kl_divergence(mean1, mean2, std)
        kl_dt = compute_kl_divergence(mean1, mean2, std, dt=dt)
        # dt < 1 should increase KL (smaller denominator)
        assert (kl_dt > kl_no_dt).all()
