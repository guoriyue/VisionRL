"""Tests for the world model engine end-to-end."""

import pytest
import torch

from wm_infra.config import EngineConfig, DynamicsConfig, TokenizerConfig, StateCacheConfig
from wm_infra.core.engine import WorldModelEngine, RolloutJob
from wm_infra.core.state import LatentStateManager
from wm_infra.core.scheduler import RolloutScheduler, RolloutRequest
from wm_infra.models.dynamics import LatentDynamicsModel
from wm_infra.models.base import RolloutInput
from wm_infra.tokenizer.video_tokenizer import VideoTokenizer, FSQQuantizer


# ─── Fixtures ───


def _small_config() -> EngineConfig:
    """Minimal config for fast CPU testing."""
    return EngineConfig(
        device="cpu" if not torch.cuda.is_available() else "cuda",
        dtype="float32",
        dynamics=DynamicsConfig(
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            action_dim=8,
            latent_token_dim=6,  # matches FSQ dim
            max_rollout_steps=16,
        ),
        tokenizer=TokenizerConfig(
            spatial_downsample=2,  # small for testing
            temporal_downsample=1,
            latent_channels=16,
            fsq_levels=[4, 4, 4, 3, 3, 3],
        ),
        state_cache=StateCacheConfig(
            max_batch_size=4,
            max_rollout_steps=16,
            latent_dim=6,
            num_latent_tokens=16,
            pool_size_gb=0.1,
        ),
    )


# ─── Unit Tests ───


class TestFSQQuantizer:
    def test_roundtrip(self):
        levels = [4, 4, 4, 3, 3, 3]
        fsq = FSQQuantizer(levels)
        z = torch.randn(2, 10, len(levels))
        z_q, indices = fsq(z)
        assert z_q.shape == z.shape
        assert indices.shape == (2, 10)

        # Decode indices should match quantized values
        z_decoded = fsq.decode_indices(indices)
        # After quantization + decode, values should be close (within grid)
        assert z_decoded.shape == z_q.shape

    def test_codebook_size(self):
        levels = [8, 8, 8, 5, 5, 5]
        fsq = FSQQuantizer(levels)
        assert fsq.codebook_size == 8 * 8 * 8 * 5 * 5 * 5

    def test_gradient_flows(self):
        fsq = FSQQuantizer([4, 4, 4])
        z = torch.randn(1, 5, 3, requires_grad=True)
        z_q, _ = fsq(z)
        loss = z_q.sum()
        loss.backward()
        assert z.grad is not None
        assert z.grad.shape == z.shape


class TestVideoTokenizer:
    def test_encode_decode_shape(self):
        config = TokenizerConfig(
            spatial_downsample=2,
            temporal_downsample=1,
            latent_channels=16,
            fsq_levels=[4, 4, 4, 3, 3, 3],
        )
        tok = VideoTokenizer(config)

        video = torch.randn(1, 2, 3, 8, 8)  # [B, T, C, H, W]
        z_q, indices = tok.encode(video)

        assert z_q.ndim == 4  # [B, T', N, D]
        assert indices.ndim == 3  # [B, T', N]

    def test_single_frame(self):
        config = TokenizerConfig(
            spatial_downsample=2,
            temporal_downsample=1,
            latent_channels=8,
            fsq_levels=[4, 4, 4],
        )
        tok = VideoTokenizer(config)
        frame = torch.randn(1, 3, 8, 8)
        z_q, indices = tok.encode_frame(frame)
        assert z_q.ndim == 4


class TestLatentDynamicsModel:
    def test_predict_next_shape(self):
        config = DynamicsConfig(
            hidden_dim=64, num_heads=4, num_layers=2,
            action_dim=8, latent_token_dim=6,
        )
        model = LatentDynamicsModel(config)
        state = torch.randn(1, 16, 6)  # [B, N, D]
        action = torch.randn(1, 8)  # [B, A]

        next_state = model.predict_next(state, action)
        assert next_state.shape == state.shape

    def test_rollout(self):
        config = DynamicsConfig(
            hidden_dim=64, num_heads=4, num_layers=2,
            action_dim=8, latent_token_dim=6,
        )
        model = LatentDynamicsModel(config)
        model.eval()

        inp = RolloutInput(
            latent_state=torch.randn(1, 16, 6),
            actions=torch.randn(1, 4, 8),
            num_steps=4,
        )
        with torch.inference_mode():
            out = model.rollout(inp)

        assert out.predicted_states.shape == (1, 4, 16, 6)


class TestLatentStateManager:
    def test_create_and_append(self):
        mgr = LatentStateManager(max_concurrent=4, max_memory_gb=0.01, device="cpu")
        state = mgr.create("r1", torch.randn(16, 6), max_steps=5)
        assert state.current_step == 0
        assert mgr.num_active == 1

        mgr.append_step("r1", torch.randn(8), torch.randn(16, 6))
        state = mgr.get("r1")
        assert state.current_step == 1
        assert len(state.latent_states) == 2

    def test_fork(self):
        mgr = LatentStateManager(max_concurrent=4, device="cpu")
        mgr.create("r1", torch.randn(16, 6), max_steps=10)
        mgr.append_step("r1", torch.randn(8), torch.randn(16, 6))

        forked = mgr.fork("r1", "r1_fork")
        assert forked.current_step == 1
        assert len(forked.latent_states) == 2
        assert mgr.num_active == 2

    def test_eviction(self):
        mgr = LatentStateManager(max_concurrent=2, device="cpu")
        mgr.create("r1", torch.randn(4, 4), max_steps=5)
        mgr.create("r2", torch.randn(4, 4), max_steps=5)
        # Creating a 3rd should evict r1 (oldest)
        mgr.create("r3", torch.randn(4, 4), max_steps=5)
        assert mgr.num_active == 2
        assert "r1" not in mgr._states


class TestRolloutScheduler:
    def test_submit_and_schedule(self):
        from wm_infra.config import SchedulerConfig
        scheduler = RolloutScheduler(SchedulerConfig(max_batch_size=2))

        scheduler.submit(RolloutRequest(request_id="r1", num_steps=3))
        scheduler.submit(RolloutRequest(request_id="r2", num_steps=5))

        batch = scheduler.schedule_batch()
        assert batch.size == 2
        assert "r1" in batch.request_ids
        assert "r2" in batch.request_ids

    def test_sjf_ordering(self):
        from wm_infra.config import SchedulerConfig, SchedulerPolicy
        scheduler = RolloutScheduler(SchedulerConfig(
            max_batch_size=1, policy=SchedulerPolicy.SJF,
        ))

        scheduler.submit(RolloutRequest(request_id="long", num_steps=100))
        scheduler.submit(RolloutRequest(request_id="short", num_steps=2))

        batch = scheduler.schedule_batch()
        assert batch.request_ids[0] == "short"  # shorter job first


class TestWorldModelEngine:
    def test_end_to_end_with_latent(self):
        config = _small_config()
        config.device = "cpu"
        dynamics = LatentDynamicsModel(config.dynamics)

        engine = WorldModelEngine(config, dynamics, tokenizer=None)

        job = RolloutJob(
            job_id="test1",
            initial_latent=torch.randn(16, 6),
            actions=torch.randn(3, 8),
            num_steps=3,
            return_frames=False,
            return_latents=True,
        )

        engine.submit_job(job)
        results = engine.run_until_done()

        assert len(results) == 1
        result = results[0]
        assert result.steps_completed == 3
        assert result.predicted_latents is not None
        assert result.elapsed_ms > 0

    def test_multiple_concurrent_jobs(self):
        config = _small_config()
        config.device = "cpu"
        dynamics = LatentDynamicsModel(config.dynamics)
        engine = WorldModelEngine(config, dynamics, tokenizer=None)

        for i in range(3):
            engine.submit_job(RolloutJob(
                job_id=f"job{i}",
                initial_latent=torch.randn(16, 6),
                actions=torch.randn(2, 8),
                num_steps=2,
                return_frames=False,
                return_latents=True,
            ))

        results = engine.run_until_done()
        assert len(results) == 3
        for r in results:
            assert r.steps_completed == 2
