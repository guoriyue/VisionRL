"""Parity tests for the SD3.5 generation executor migration.

These tests pin two equivalences:

1. ``SD3_5PipelineExecutor.forward`` called directly produces the same
   ``OutputBatch`` as routing the same request through the engine
   ``GenerationRuntime``/``GenerationWorker`` adapter chain. The runtime
   path must not reorder or mutate trajectory tensors.

2. ``SD3_5Collector.collect`` invoked twice with the same prompts and
   the same seed produces bitwise-identical ``ExperienceBatch`` tensors.
   This is the parity contract for Phase 2 — same prompts + same seed ⇒
   same trajectory ⇒ same advantages ⇒ same gradient.

The stubs below are bare-minimum: no diffusers / SD3.5 weights are
loaded. They exercise the executor + collector wiring, the SDE math, and
the OutputBatch → ExperienceBatch translation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
import torch

from vrl.engine import ContinuousBatchPlanner, EngineLoop, Scheduler
from vrl.engine.generation import (
    FamilyPipelineRegistry,
    GenerationIdFactory,
    GenerationModelRunner,
    GenerationRequest,
    GenerationRuntime,
    GenerationWorker,
)
from vrl.models.families.sd3_5.executor import SD3_5PipelineExecutor

# ---------------------------------------------------------------------------
# Stubs (mirrors test_pipeline_sd3_5.py — kept local so the parity test can
# evolve independently of the unit test stubs).
# ---------------------------------------------------------------------------


class _StubScheduler:
    def __init__(self, num_steps: int) -> None:
        self._num_steps = num_steps
        self.sigmas = torch.linspace(1.0, 0.0, num_steps + 2)
        self.timesteps = torch.tensor(
            [self.sigmas[i].item() for i in range(num_steps)],
            dtype=torch.float32,
        )

    def index_for_timestep(self, t: torch.Tensor) -> int:
        diffs = (self.sigmas - t).abs()
        idx = int(diffs.argmin().item())
        return min(idx, len(self.sigmas) - 2)

    def set_timesteps(self, n: int, device: Any = None) -> None:
        del device
        self.sigmas = torch.linspace(1.0, 0.0, n + 2)
        self.timesteps = torch.tensor(
            [self.sigmas[i].item() for i in range(n)],
            dtype=torch.float32,
        )


@dataclass
class _StubSamplingState:
    latents: torch.Tensor
    timesteps: torch.Tensor
    scheduler: _StubScheduler
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    negative_pooled_prompt_embeds: torch.Tensor | None
    guidance_scale: float
    do_cfg: bool
    seed: int


@dataclass
class _StubPolicy:
    embed_dim: int = 8
    pooled_dim: int = 4
    latent_channels: int = 4
    family: str = "sd3_5-stub"
    encode_calls: list[str] = field(default_factory=list)

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del negative_prompt, kwargs
        if isinstance(prompt, list):
            prompt = prompt[0]
        self.encode_calls.append(prompt)
        h = abs(hash(prompt)) % (2**31)
        gen = torch.Generator().manual_seed(h)
        prompt_embeds = torch.randn(1, 4, self.embed_dim, generator=gen)
        pooled = torch.randn(1, self.pooled_dim, generator=gen)
        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled,
            "negative_prompt_embeds": torch.zeros_like(prompt_embeds),
            "negative_pooled_prompt_embeds": torch.zeros_like(pooled),
        }

    def prepare_sampling(
        self,
        request: Any,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> _StubSamplingState:
        del kwargs
        scheduler = _StubScheduler(request.num_steps)
        bsz = encoded["prompt_embeds"].shape[0]
        H = max(1, request.height // 32)
        W = max(1, request.width // 32)
        seed = request.seed if request.seed is not None else 0
        gen = torch.Generator().manual_seed(int(seed))
        latents = torch.randn(
            bsz, self.latent_channels, H, W, generator=gen,
        )
        return _StubSamplingState(
            latents=latents,
            timesteps=scheduler.timesteps,
            scheduler=scheduler,
            prompt_embeds=encoded["prompt_embeds"],
            pooled_prompt_embeds=encoded["pooled_prompt_embeds"],
            negative_prompt_embeds=encoded.get("negative_prompt_embeds"),
            negative_pooled_prompt_embeds=encoded.get("negative_pooled_prompt_embeds"),
            guidance_scale=request.guidance_scale,
            do_cfg=request.guidance_scale > 1.0,
            seed=int(seed),
        )

    def forward_step(
        self,
        state: _StubSamplingState,
        step_idx: int,
    ) -> dict[str, torch.Tensor]:
        noise_pred = torch.sin(state.latents) * (step_idx + 1) * 0.01
        return {
            "noise_pred": noise_pred,
            "noise_pred_cond": noise_pred,
            "noise_pred_uncond": torch.zeros_like(noise_pred),
        }

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        _B, C, _H, _W = latents.shape
        rgb = latents[:, :3] if C >= 3 else latents.repeat(1, 3, 1, 1)[:, :3]
        return torch.nn.functional.interpolate(
            rgb, scale_factor=8, mode="nearest",
        )

    def export_batch_context(self, state: _StubSamplingState) -> dict[str, Any]:
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
            "model_family": self.family,
        }

    def export_training_extras(self, state: _StubSamplingState) -> dict[str, Any]:
        return {
            "prompt_embeds": state.prompt_embeds,
            "pooled_prompt_embeds": state.pooled_prompt_embeds,
            "negative_prompt_embeds": state.negative_prompt_embeds,
            "negative_pooled_prompt_embeds": state.negative_pooled_prompt_embeds,
        }


class _ConstantReward:
    """Reward that returns a deterministic float keyed off the prompt."""

    async def score(self, rollout: Any) -> float:
        # deterministic — independent of the video tensor
        return float(abs(hash(rollout.trajectory.prompt)) % 1000) / 1000.0


# ---------------------------------------------------------------------------
# Test 1: direct executor vs runtime-routed executor
# ---------------------------------------------------------------------------


def _build_request(seed: int = 99, *, num_steps: int = 3) -> GenerationRequest:
    return GenerationRequest(
        request_id="parity-test",
        family="sd3_5",
        task="t2i",
        prompts=["a red cube"],
        samples_per_prompt=2,
        sampling={
            "num_steps": num_steps,
            "guidance_scale": 4.5,
            "height": 32,
            "width": 32,
            "noise_level": 0.7,
            "cfg": True,
            "sample_batch_size": 8,
            "sde_window_size": 0,
            "sde_window_range": [0, num_steps],
            "same_latent": False,
            "max_sequence_length": 256,
            "seed": seed,
        },
        return_artifacts={"output", "rollout_trajectory_data", "denoising_env"},
    )


def test_executor_direct_matches_runtime_routed_bitwise() -> None:
    """Same request through executor.forward(...) vs Worker.execute(...) → identical tensors.

    Bitwise-equal: both code paths run the SDE math with identical inputs in
    the same order; the runtime path adds no extra ops.
    """
    policy_a = _StubPolicy()
    executor_a = SD3_5PipelineExecutor(policy_a)
    request_a = _build_request()
    specs_a = GenerationIdFactory().build_sample_specs(request_a)
    out_direct = executor_a.forward(request_a, specs_a)

    # Route through worker (same code path the runtime would take, minus
    # the asyncio engine loop — which doesn't touch tensors).
    policy_b = _StubPolicy()
    executor_b = SD3_5PipelineExecutor(policy_b)
    registry = FamilyPipelineRegistry()
    registry.register(executor_b)
    worker = GenerationWorker(registry)
    request_b = _build_request()
    out_via_worker_dict = worker.execute([request_b])
    out_via_worker = out_via_worker_dict[request_b.request_id]

    assert out_via_worker.error is None
    assert torch.equal(out_direct.output, out_via_worker.output)
    assert torch.equal(
        out_direct.rollout_trajectory_data.rollout_log_probs,
        out_via_worker.rollout_trajectory_data.rollout_log_probs,
    )
    assert torch.equal(
        out_direct.rollout_trajectory_data.dit_trajectory.latents,
        out_via_worker.rollout_trajectory_data.dit_trajectory.latents,
    )
    assert torch.equal(
        out_direct.rollout_trajectory_data.dit_trajectory.timesteps,
        out_via_worker.rollout_trajectory_data.dit_trajectory.timesteps,
    )


@pytest.mark.asyncio
async def test_executor_direct_matches_runtime_engine_loop_bitwise() -> None:
    """Routing through the full GenerationRuntime + EngineLoop preserves tensors."""
    policy_a = _StubPolicy()
    executor_a = SD3_5PipelineExecutor(policy_a)
    request_a = _build_request()
    specs_a = GenerationIdFactory().build_sample_specs(request_a)
    out_direct = executor_a.forward(request_a, specs_a)

    policy_b = _StubPolicy()
    executor_b = SD3_5PipelineExecutor(policy_b)
    registry = FamilyPipelineRegistry()
    registry.register(executor_b)
    worker = GenerationWorker(registry)
    runner = GenerationModelRunner(worker, execute_in_thread=False)
    engine_loop = EngineLoop(
        scheduler=Scheduler(
            batch_planner=ContinuousBatchPlanner(max_batch_size=1),
        ),
        model_runner=runner,
    )
    runtime = GenerationRuntime(engine_loop)
    try:
        out_runtime = await asyncio.wait_for(
            runtime.generate(_build_request()), timeout=5.0,
        )
    finally:
        await runtime.shutdown()

    assert out_runtime.error is None
    assert torch.equal(out_direct.output, out_runtime.output)
    assert torch.equal(
        out_direct.rollout_trajectory_data.rollout_log_probs,
        out_runtime.rollout_trajectory_data.rollout_log_probs,
    )
    assert torch.equal(
        out_direct.rollout_trajectory_data.dit_trajectory.latents,
        out_runtime.rollout_trajectory_data.dit_trajectory.latents,
    )


# ---------------------------------------------------------------------------
# Test 2: collector.collect() called twice with same seed → identical batch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collector_collect_twice_same_seed_bitwise() -> None:
    """Two collect() calls with the same prompts + seed → identical ExperienceBatch.

    Parity is bitwise on every tensor field. The reward fn is also
    deterministic so the kl-adjusted rewards match.
    """
    from vrl.rollouts.collectors.sd3_5 import (
        SD3_5Collector,
        SD3_5CollectorConfig,
    )

    cfg = SD3_5CollectorConfig(
        num_steps=3,
        guidance_scale=4.5,
        height=32,
        width=32,
        noise_level=0.7,
        cfg=True,
        sample_batch_size=8,
        kl_reward=0.0,
    )

    collector_a = SD3_5Collector(_StubPolicy(), _ConstantReward(), cfg)
    collector_b = SD3_5Collector(_StubPolicy(), _ConstantReward(), cfg)

    try:
        batch_a = await collector_a.collect(["a red cube"], group_size=4, seed=42)
        batch_b = await collector_b.collect(["a red cube"], group_size=4, seed=42)
    finally:
        await collector_a.shutdown()
        await collector_b.shutdown()

    assert torch.equal(batch_a.observations, batch_b.observations)
    assert torch.equal(batch_a.actions, batch_b.actions)
    assert torch.equal(batch_a.rewards, batch_b.rewards)
    assert torch.equal(batch_a.dones, batch_b.dones)
    assert torch.equal(batch_a.group_ids, batch_b.group_ids)
    assert torch.equal(batch_a.extras["log_probs"], batch_b.extras["log_probs"])
    assert torch.equal(batch_a.extras["timesteps"], batch_b.extras["timesteps"])
    assert torch.equal(batch_a.extras["kl"], batch_b.extras["kl"])
    assert torch.equal(batch_a.videos, batch_b.videos)
    assert batch_a.context == batch_b.context
    assert batch_a.prompts == batch_b.prompts


@pytest.mark.asyncio
async def test_collector_experience_batch_shape() -> None:
    """ExperienceBatch from collector has the trainer-expected fields and shapes."""
    from vrl.rollouts.collectors.sd3_5 import (
        SD3_5Collector,
        SD3_5CollectorConfig,
    )

    cfg = SD3_5CollectorConfig(
        num_steps=3,
        guidance_scale=4.5,
        height=32,
        width=32,
        sample_batch_size=8,
    )
    collector = SD3_5Collector(_StubPolicy(), _ConstantReward(), cfg)
    try:
        batch = await collector.collect(["a red cube"], group_size=4, seed=11)
    finally:
        await collector.shutdown()

    # Trainer-required fields
    assert batch.observations.shape[0] == 4
    assert batch.observations.shape[1] == 3  # num_steps
    assert batch.actions.shape == batch.observations.shape
    assert batch.rewards.shape == (4,)
    assert batch.dones.shape == (4,)
    assert batch.group_ids.shape == (4,)
    assert batch.extras["log_probs"].shape == (4, 3)
    assert batch.extras["timesteps"].shape == (4, 3)
    assert batch.extras["kl"].shape == (4, 3)
    assert "reward_before_kl" in batch.extras
    # Replay-time embeds (from policy.export_training_extras)
    assert "prompt_embeds" in batch.extras
    assert "pooled_prompt_embeds" in batch.extras
    # Context for restore_eval_state
    assert batch.context["guidance_scale"] == 4.5
    assert batch.context["cfg"] is True
