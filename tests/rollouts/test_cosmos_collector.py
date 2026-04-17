"""Tests for vrl.rollouts.collectors.cosmos (CosmosDiffusersCollector)."""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# CosmosDiffusersCollectorConfig — config defaults
# ---------------------------------------------------------------------------

class TestCosmosDiffusersCollectorConfig:
    def test_defaults(self) -> None:
        """Verify config fields have correct Cosmos Predict2 defaults."""
        from vrl.rollouts.collectors.cosmos import CosmosDiffusersCollectorConfig

        cfg = CosmosDiffusersCollectorConfig()
        assert cfg.num_steps == 35
        assert cfg.guidance_scale == 7.0
        assert cfg.height == 704
        assert cfg.width == 1280
        assert cfg.num_frames == 93
        assert cfg.fps == 16
        assert cfg.max_sequence_length == 512
        assert cfg.cfg is True
        assert cfg.kl_reward == 0.0
        assert cfg.sde_window_size == 0
        assert cfg.sde_window_range == (0, 10)
        assert cfg.same_latent is False

    def test_custom_values(self) -> None:
        """Can set custom values for all config fields."""
        from vrl.rollouts.collectors.cosmos import CosmosDiffusersCollectorConfig

        cfg = CosmosDiffusersCollectorConfig(
            num_steps=50,
            guidance_scale=5.0,
            height=480,
            width=640,
            num_frames=81,
            fps=24,
            kl_reward=0.1,
            sde_window_size=10,
            sde_window_range=(5, 30),
            same_latent=True,
        )
        assert cfg.num_steps == 50
        assert cfg.guidance_scale == 5.0
        assert cfg.height == 480
        assert cfg.width == 640
        assert cfg.num_frames == 81
        assert cfg.fps == 24
        assert cfg.kl_reward == 0.1
        assert cfg.sde_window_size == 10
        assert cfg.sde_window_range == (5, 30)
        assert cfg.same_latent is True


# ---------------------------------------------------------------------------
# CosmosDiffusersCollector — sde_window
# ---------------------------------------------------------------------------

class TestCosmosDiffusersCollectorSdeWindow:
    def test_sde_window_disabled(self) -> None:
        """sde_window_size=0 → _get_sde_window returns None."""
        from vrl.rollouts.collectors.cosmos import (
            CosmosDiffusersCollector,
            CosmosDiffusersCollectorConfig,
        )

        cfg = CosmosDiffusersCollectorConfig(sde_window_size=0)
        collector = CosmosDiffusersCollector(
            model=None, reward_fn=None, config=cfg,
        )
        assert collector._get_sde_window() is None

    def test_sde_window_range(self) -> None:
        """sde_window_size>0 → returns a window within the range."""
        from vrl.rollouts.collectors.cosmos import (
            CosmosDiffusersCollector,
            CosmosDiffusersCollectorConfig,
        )

        cfg = CosmosDiffusersCollectorConfig(
            sde_window_size=5, sde_window_range=(0, 20),
        )
        collector = CosmosDiffusersCollector(
            model=None, reward_fn=None, config=cfg,
        )
        for _ in range(20):
            window = collector._get_sde_window()
            assert window is not None
            start, end = window
            assert start >= 0
            assert end == start + 5
            assert start <= 15  # max start = 20 - 5

    def test_sde_window_fixed_size(self) -> None:
        """Window always has exactly sde_window_size steps."""
        from vrl.rollouts.collectors.cosmos import (
            CosmosDiffusersCollector,
            CosmosDiffusersCollectorConfig,
        )

        cfg = CosmosDiffusersCollectorConfig(
            sde_window_size=3, sde_window_range=(2, 10),
        )
        collector = CosmosDiffusersCollector(
            model=None, reward_fn=None, config=cfg,
        )
        for _ in range(10):
            window = collector._get_sde_window()
            assert window[1] - window[0] == 3


# ---------------------------------------------------------------------------
# CosmosDiffusersCollector — init
# ---------------------------------------------------------------------------

class TestCosmosDiffusersCollectorInit:
    def test_accepts_model_and_reward(self) -> None:
        """Collector initializes with model and reward function."""
        from vrl.rollouts.collectors.cosmos import CosmosDiffusersCollector

        collector = CosmosDiffusersCollector(
            model="mock_model",
            reward_fn="mock_reward",
        )
        assert collector.model == "mock_model"
        assert collector.reward_fn == "mock_reward"
        assert collector.reference_image is None

    def test_accepts_reference_image(self) -> None:
        """Collector stores reference image for Video2World conditioning."""
        from vrl.rollouts.collectors.cosmos import CosmosDiffusersCollector

        collector = CosmosDiffusersCollector(
            model="mock_model",
            reward_fn="mock_reward",
            reference_image="mock_image",
        )
        assert collector.reference_image == "mock_image"

    def test_default_config(self) -> None:
        """Without explicit config, uses CosmosDiffusersCollectorConfig defaults."""
        from vrl.rollouts.collectors.cosmos import (
            CosmosDiffusersCollector,
            CosmosDiffusersCollectorConfig,
        )

        collector = CosmosDiffusersCollector(
            model=None, reward_fn=None,
        )
        assert isinstance(collector.config, CosmosDiffusersCollectorConfig)
        assert collector.config.num_steps == 35


# ---------------------------------------------------------------------------
# CosmosDiffusersCollector — forward_step (mock)
# ---------------------------------------------------------------------------

class TestCosmosDiffusersCollectorForwardStep:
    def test_forward_step_returns_noise_pred(self) -> None:
        """forward_step returns dict with noise_pred, noise_pred_cond, noise_pred_uncond."""
        import torch

        from vrl.rollouts.collectors.cosmos import (
            CosmosDiffusersCollector,
            CosmosDiffusersCollectorConfig,
        )
        from vrl.rollouts.types import ExperienceBatch

        B, T, C, D, H, W = 2, 5, 16, 4, 22, 40
        cfg = CosmosDiffusersCollectorConfig(cfg=False)

        # The collector needs a model with _predict_noise_with_model
        # (or _executor._predict_noise_with_model)
        class MockExecutor:
            def _predict_noise_with_model(self, model, denoise_state, step_idx):
                ms = denoise_state.model_state
                return {
                    "noise_pred": ms.latents,
                    "noise_pred_cond": ms.latents,
                    "noise_pred_uncond": torch.zeros_like(ms.latents),
                }

        mock_model = MockExecutor()

        collector = CosmosDiffusersCollector(
            model=mock_model, reward_fn=None, config=cfg,
        )

        # Mock transformer model: return input as noise_pred
        class MockTransformerModel:
            def __call__(self, **kwargs):
                return (kwargs["hidden_states"],)

        transformer_model = MockTransformerModel()

        # Build a mock ExperienceBatch with context
        observations = torch.randn(B, T, C, D, H, W)
        actions = torch.randn(B, T, C, D, H, W)
        timesteps = torch.rand(B, T) * 1000
        prompt_embeds = torch.randn(B, 10, 64)
        neg_embeds = torch.randn(B, 10, 64)
        cond_indicator = torch.zeros(1, C, D, 1, 1)
        uncond_indicator = torch.zeros(1, C, D, 1, 1)
        cond_mask = torch.zeros(1, 1, D, 1, 1)
        uncond_mask = torch.zeros(1, 1, D, 1, 1)
        padding_mask = torch.zeros(1, 1, H * 8, W * 8)
        init_latents = torch.randn(B, C, D, H, W)

        # Mock scheduler with sigmas for forward_step
        class MockScheduler:
            sigmas = torch.ones(T)

        batch = ExperienceBatch(
            observations=observations,
            actions=actions,
            rewards=torch.zeros(B),
            dones=torch.ones(B, dtype=torch.bool),
            group_ids=torch.zeros(B, dtype=torch.long),
            extras={
                "timesteps": timesteps,
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": neg_embeds,
                "init_latents": init_latents,
            },
            context={
                "guidance_scale": 7.0,
                "cfg": False,
                "fps": 16,
                "cond_mask": cond_mask,
                "uncond_mask": uncond_mask,
                "padding_mask": padding_mask,
                "cond_indicator": cond_indicator,
                "uncond_indicator": uncond_indicator,
                "scheduler": MockScheduler(),
            },
        )

        result = collector.forward_step(transformer_model, batch, timestep_idx=0)

        assert "noise_pred" in result
        assert "noise_pred_cond" in result
        assert "noise_pred_uncond" in result
        assert result["noise_pred"].shape[0] == B


# ---------------------------------------------------------------------------
# CosmosDiffusersCollector — seed generator selection
# ---------------------------------------------------------------------------

class TestCosmosDiffusersCollectorSeedGenerator:
    def test_seed_kwarg_creates_deterministic_generator(self) -> None:
        """When seed is provided, collect() should create a deterministic SDE generator."""
        import torch
        from unittest.mock import AsyncMock, MagicMock, patch

        from vrl.rollouts.collectors.cosmos import (
            CosmosDiffusersCollector,
            CosmosDiffusersCollectorConfig,
        )

        cfg = CosmosDiffusersCollectorConfig(num_steps=2, cfg=False)

        # Verify the code path: we just need to check that the collector
        # doesn't crash when seed is provided and that the variable name
        # change from latent_generator to sde_generator is correct.
        model = MagicMock()
        model.device = torch.device("cpu")

        ms = MagicMock()
        ms.latents = torch.randn(1, 16, 3, 8, 14)
        ms.timesteps = torch.tensor([999, 500])
        ms.prompt_embeds = torch.zeros(1, 4, 8)
        ms.negative_prompt_embeds = torch.zeros(1, 4, 8)
        ms.guidance_scale = 7.0
        ms.do_cfg = False
        ms.scheduler = MagicMock()

        loop = MagicMock()
        loop.total_steps = 2
        loop.current_step = 0
        loop.model_state = ms

        encode_result = MagicMock()
        encode_result.state_updates = {
            "prompt_embeds": torch.zeros(1, 4, 8),
            "negative_prompt_embeds": torch.zeros(1, 4, 8),
        }
        model.encode_text = AsyncMock(return_value=encode_result)
        model.denoise_init = AsyncMock(return_value=loop)
        model.predict_noise = AsyncMock(
            return_value={"noise_pred": ms.latents * 0.1}
        )
        decode_result = MagicMock()
        decode_result.state_updates = {"video": torch.zeros(1, 3, 8, 64, 64)}
        model.decode_vae = AsyncMock(return_value=decode_result)

        reward = MagicMock()
        reward.score = AsyncMock(return_value=1.0)

        collector = CosmosDiffusersCollector(model=model, reward_fn=reward, config=cfg)

        def _fake_sde(*args, **kwargs):
            sample = args[3] if len(args) > 3 else kwargs.get("sample")
            gen = kwargs.get("generator", None)
            result = MagicMock()
            if gen is not None:
                result.prev_sample = torch.randn(sample.shape, generator=gen)
            else:
                result.prev_sample = torch.randn_like(sample)
            result.log_prob = torch.zeros(sample.shape[0])
            return result

        import asyncio
        with patch(
            "vrl.rollouts.evaluators.diffusion.flow_matching.sde_step_with_logprob",
            side_effect=_fake_sde,
        ):
            # Should not crash — validates sde_generator variable name is correct
            batch = asyncio.run(collector.collect(["test prompt"], seed=42))
            assert batch.rewards.shape[0] == 1
