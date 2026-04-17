"""Tests for WanDiffusersCollector — metadata passthrough for OCR/structured rewards."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from vrl.rollouts.collectors.wan_diffusers import (
    WanDiffusersCollector,
    WanDiffusersCollectorConfig,
)


# ---------------------------------------------------------------------------
# Stub model + reward that capture what they receive
# ---------------------------------------------------------------------------

class _CaptureReward:
    """Reward function that captures rollout metadata for inspection."""

    def __init__(self) -> None:
        self.captured_rollouts: list = []

    async def score(self, rollout):
        self.captured_rollouts.append(rollout)
        return 1.0


class _StubWanModel:
    """Minimal stub implementing the DiffusersWanT2VModel interface."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")

    async def encode_text(self, request, state):
        result = MagicMock()
        result.state_updates = {
            "prompt_embeds": torch.zeros(1, 4, 8),
            "negative_prompt_embeds": torch.zeros(1, 4, 8),
        }
        return result

    async def denoise_init(self, request, state):
        ms = MagicMock()
        # Use request seed for deterministic initial latents if available
        if hasattr(request, 'seed') and request.seed is not None:
            g = torch.Generator()
            g.manual_seed(request.seed)
            ms.latents = torch.randn(1, 16, 3, 8, 14, generator=g)
        else:
            ms.latents = torch.randn(1, 16, 3, 8, 14)
        ms.timesteps = torch.tensor([999, 500])
        ms.prompt_embeds = torch.zeros(1, 4, 8)
        ms.negative_prompt_embeds = torch.zeros(1, 4, 8)
        ms.guidance_scale = 4.5
        ms.do_cfg = True
        ms.scheduler = MagicMock()

        loop = MagicMock()
        loop.total_steps = 2
        loop.current_step = 0
        loop.model_state = ms
        return loop

    async def predict_noise(self, denoise_loop, step_idx):
        # Deterministic: return a function of latents (not random)
        latents = denoise_loop.model_state.latents
        return {"noise_pred": latents * 0.1 + step_idx * 0.01}

    async def decode_vae(self, request, state):
        result = MagicMock()
        result.state_updates = {"video": torch.zeros(1, 3, 8, 64, 64)}
        return result


def _fake_sde_step(*args, **kwargs):
    """Fake sde_step_with_logprob returning plausible shapes.

    When a generator is provided, uses it for deterministic noise.
    """
    sample = args[3] if len(args) > 3 else kwargs.get("sample")
    generator = kwargs.get("generator", None)
    result = MagicMock()
    if generator is not None:
        result.prev_sample = torch.randn(sample.shape, generator=generator)
    else:
        result.prev_sample = torch.randn_like(sample)
    result.log_prob = torch.zeros(sample.shape[0])
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestWanDiffusersCollectorMetadata:
    @patch(
        "vrl.rollouts.evaluators.diffusion.flow_matching.sde_step_with_logprob",
        side_effect=_fake_sde_step,
    )
    async def test_target_text_propagated_to_reward(self, _mock_sde) -> None:
        """OCR target_text passed via collect() kwargs reaches the reward rollout."""
        reward = _CaptureReward()
        model = _StubWanModel()
        cfg = WanDiffusersCollectorConfig(num_steps=2, cfg=False)
        collector = WanDiffusersCollector(model, reward, cfg)

        await collector.collect(
            ["A sign that says HELLO"],
            target_text="HELLO",
        )

        assert len(reward.captured_rollouts) == 1
        rollout = reward.captured_rollouts[0]
        assert rollout.metadata["target_text"] == "HELLO"

    @patch(
        "vrl.rollouts.evaluators.diffusion.flow_matching.sde_step_with_logprob",
        side_effect=_fake_sde_step,
    )
    async def test_task_type_defaults_to_text_to_video(self, _mock_sde) -> None:
        reward = _CaptureReward()
        model = _StubWanModel()
        cfg = WanDiffusersCollectorConfig(num_steps=2, cfg=False)
        collector = WanDiffusersCollector(model, reward, cfg)

        await collector.collect(["test prompt"])

        rollout = reward.captured_rollouts[0]
        assert rollout.metadata["task_type"] == "text_to_video"

    @patch(
        "vrl.rollouts.evaluators.diffusion.flow_matching.sde_step_with_logprob",
        side_effect=_fake_sde_step,
    )
    async def test_sample_metadata_merged(self, _mock_sde) -> None:
        """Extra sample_metadata fields should appear in rollout.metadata."""
        reward = _CaptureReward()
        model = _StubWanModel()
        cfg = WanDiffusersCollectorConfig(num_steps=2, cfg=False)
        collector = WanDiffusersCollector(model, reward, cfg)

        await collector.collect(
            ["test prompt"],
            target_text="ABC",
            sample_metadata={"difficulty": "hard", "source": "synthetic"},
        )

        rollout = reward.captured_rollouts[0]
        assert rollout.metadata["difficulty"] == "hard"
        assert rollout.metadata["source"] == "synthetic"
        assert rollout.metadata["target_text"] == "ABC"

    @patch(
        "vrl.rollouts.evaluators.diffusion.flow_matching.sde_step_with_logprob",
        side_effect=_fake_sde_step,
    )
    async def test_references_propagated(self, _mock_sde) -> None:
        reward = _CaptureReward()
        model = _StubWanModel()
        cfg = WanDiffusersCollectorConfig(num_steps=2, cfg=False)
        collector = WanDiffusersCollector(model, reward, cfg)

        await collector.collect(
            ["test prompt"],
            references=["ref1.png", "ref2.png"],
        )

        rollout = reward.captured_rollouts[0]
        assert rollout.metadata["references"] == ["ref1.png", "ref2.png"]

    @patch(
        "vrl.rollouts.evaluators.diffusion.flow_matching.sde_step_with_logprob",
        side_effect=_fake_sde_step,
    )
    async def test_seeded_collect_is_deterministic(self, _mock_sde) -> None:
        """Two collect() calls with the same seed must produce identical rollouts."""
        model = _StubWanModel()
        cfg = WanDiffusersCollectorConfig(num_steps=2, cfg=False)

        reward1 = _CaptureReward()
        collector1 = WanDiffusersCollector(model, reward1, cfg)
        batch1 = await collector1.collect(["test prompt"], seed=123)

        reward2 = _CaptureReward()
        collector2 = WanDiffusersCollector(model, reward2, cfg)
        batch2 = await collector2.collect(["test prompt"], seed=123)

        assert torch.equal(batch1.observations, batch2.observations)
        assert torch.equal(batch1.actions, batch2.actions)
        assert torch.equal(batch1.extras["log_probs"], batch2.extras["log_probs"])
