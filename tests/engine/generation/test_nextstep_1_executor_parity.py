"""Parity tests for the NextStep-1 generation executor migration.

These tests pin three equivalences:

1. ``NextStep1PipelineExecutor.forward`` called directly produces the same
   ``OutputBatch`` as routing the same request through the engine
   ``GenerationWorker`` adapter chain. The runtime path must not reorder
   or mutate trajectory tensors.

2. The same equivalence holds when the request is routed through the
   full async ``GenerationRuntime`` (which adds the asyncio engine loop
   on top of the worker).

3. ``NextStep1Collector.collect`` invoked twice with the same prompts and
   the same seed produces bitwise-identical ``ExperienceBatch`` tensors —
   especially ``actions`` (continuous tokens), ``extras["saved_noise"]``,
   and ``extras["log_probs"]``. ``saved_noise`` round-tripping bitwise IS
   the determinism contract that ``NextStep1Policy.replay_forward`` relies
   on.

The stubs below are bare-minimum: no upstream NextStep-1 weights are
loaded. They exercise the executor + collector wiring and the
OutputBatch → ExperienceBatch translation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
import torch
import torch.nn as nn

from vrl.engine import ContinuousBatchPlanner, EngineLoop, Scheduler
from vrl.engine.generation import (
    FamilyPipelineRegistry,
    GenerationIdFactory,
    GenerationModelRunner,
    GenerationRequest,
    GenerationRuntime,
    GenerationWorker,
)
from vrl.models.families.nextstep_1.executor import NextStep1PipelineExecutor

# ---------------------------------------------------------------------------
# Stubs (mirrors test_pipeline_nextstep_1.py — kept local so the parity test
# can evolve independently of the unit test stubs).
# ---------------------------------------------------------------------------


class _StubTokenizer:
    pad_token_id: int = 0

    def __call__(
        self,
        prompts: list[str],
        *,
        return_tensors: str = "pt",
        padding: str = "max_length",
        truncation: bool = True,
        max_length: int = 16,
    ) -> dict[str, torch.Tensor]:
        del return_tensors, padding, truncation
        B = len(prompts)
        ids = torch.zeros(B, max_length, dtype=torch.long)
        mask = torch.zeros(B, max_length, dtype=torch.long)
        for i, p in enumerate(prompts):
            if len(p) == 0:
                ids[i, 0] = 1
                mask[i, 0] = 1
                continue
            n = min(max_length, len(p))
            for j in range(n):
                ids[i, j] = (ord(p[j]) % 250) + 1
            mask[i, :n] = 1
        return {"input_ids": ids, "attention_mask": mask}


@dataclass
class _StubLanguageModel:
    vocab_size: int = 256
    hidden_size: int = 8

    def __post_init__(self) -> None:
        gen = torch.Generator().manual_seed(0)
        weight = torch.randn(self.vocab_size, self.hidden_size, generator=gen)
        self._embed = nn.Embedding.from_pretrained(weight, freeze=True)

    def get_input_embeddings(self) -> nn.Module:
        return self._embed


@dataclass
class _StubPolicy:
    image_token_num: int = 16
    token_dim: int = 4
    hidden_dim: int = 8
    image_size_default: int = 32
    sample_calls: int = 0
    decode_calls: int = 0
    family: str = "nextstep_1-stub"
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def __post_init__(self) -> None:
        self.processor = _StubTokenizer()
        self.language_model = _StubLanguageModel(hidden_size=self.hidden_dim)

    @torch.no_grad()
    def sample_image_tokens(
        self,
        prompt_embeds: torch.Tensor,
        uncond_embeds: torch.Tensor | None,
        prompt_mask: torch.Tensor,
        uncond_mask: torch.Tensor | None,
        *,
        cfg_scale: float | None = None,
        num_flow_steps: int | None = None,
        noise_level: float | None = None,
        image_token_num: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del uncond_embeds, prompt_mask, uncond_mask
        del cfg_scale, num_flow_steps, noise_level
        self.sample_calls += 1
        B = prompt_embeds.shape[0]
        L = int(image_token_num or self.image_token_num)
        D = self.token_dim

        gen = generator if generator is not None else torch.Generator().manual_seed(0)
        tokens = torch.randn(B, L, D, generator=gen)
        saved_noise = torch.randn(B, L, D, generator=gen)
        log_probs = -(tokens.float() ** 2).mean(dim=-1)
        return tokens, saved_noise, log_probs

    @torch.no_grad()
    def decode_image_tokens(
        self,
        tokens: torch.Tensor,
        image_size: int | None = None,
    ) -> torch.Tensor:
        self.decode_calls += 1
        H = W = int(image_size or self.image_size_default)
        B = tokens.shape[0]
        scalar = tokens.float().mean(dim=(1, 2)).view(B, 1, 1, 1)
        base = torch.linspace(-1.0, 1.0, H * W).view(1, 1, H, W)
        rgb = (scalar * base).expand(B, 3, H, W).contiguous()
        return rgb.clamp(-1.0, 1.0)


class _ConstantReward:
    """Reward fn that returns a deterministic float keyed off the prompt."""

    async def score(self, rollout: Any) -> float:
        return float(abs(hash(rollout.trajectory.prompt)) % 1000) / 1000.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_request(seed: int = 99, *, image_token_num: int = 8) -> GenerationRequest:
    return GenerationRequest(
        request_id="parity-test",
        family="nextstep_1",
        task="ar_t2i",
        prompts=["a red cube"],
        samples_per_prompt=2,
        sampling={
            "cfg_scale": 4.5,
            "num_flow_steps": 4,
            "noise_level": 1.0,
            "image_token_num": image_token_num,
            "image_size": 32,
            "max_text_length": 16,
            "rescale_to_unit": True,
            "seed": seed,
        },
        return_artifacts={"output", "rollout_trajectory_data"},
    )


# ---------------------------------------------------------------------------
# Test 1: direct executor vs Worker-routed executor (sync path)
# ---------------------------------------------------------------------------


def test_executor_direct_matches_worker_routed_bitwise() -> None:
    """Same request through executor.forward(...) vs Worker.execute(...) → identical tensors.

    Bitwise-equal: both code paths call the stub policy with identical
    inputs in the same order; the runtime path adds no extra ops.
    """
    policy_a = _StubPolicy()
    executor_a = NextStep1PipelineExecutor(policy_a)
    request_a = _build_request()
    specs_a = GenerationIdFactory().build_sample_specs(request_a)
    out_direct = executor_a.forward(request_a, specs_a)

    policy_b = _StubPolicy()
    executor_b = NextStep1PipelineExecutor(policy_b)
    registry = FamilyPipelineRegistry()
    registry.register(executor_b)
    worker = GenerationWorker(registry)
    request_b = _build_request()
    out_via_worker_dict = worker.execute([request_b])
    out_via_worker = out_via_worker_dict[request_b.request_id]

    assert out_via_worker.error is None
    assert torch.equal(out_direct.output, out_via_worker.output)
    assert torch.equal(
        out_direct.extra["tokens"],
        out_via_worker.extra["tokens"],
    )
    # saved_noise round-tripping bitwise IS the determinism contract.
    assert torch.equal(
        out_direct.extra["saved_noise"],
        out_via_worker.extra["saved_noise"],
    )
    assert torch.equal(
        out_direct.extra["log_probs"],
        out_via_worker.extra["log_probs"],
    )
    assert torch.equal(
        out_direct.rollout_trajectory_data.rollout_log_probs,
        out_via_worker.rollout_trajectory_data.rollout_log_probs,
    )


# ---------------------------------------------------------------------------
# Test 2: direct executor vs full GenerationRuntime (async path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_executor_direct_matches_runtime_engine_loop_bitwise() -> None:
    """Routing through the full GenerationRuntime + EngineLoop preserves tensors."""
    policy_a = _StubPolicy()
    executor_a = NextStep1PipelineExecutor(policy_a)
    request_a = _build_request()
    specs_a = GenerationIdFactory().build_sample_specs(request_a)
    out_direct = executor_a.forward(request_a, specs_a)

    policy_b = _StubPolicy()
    executor_b = NextStep1PipelineExecutor(policy_b)
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
            runtime.generate(_build_request()),
            timeout=5.0,
        )
    finally:
        await runtime.shutdown()

    assert out_runtime.error is None
    assert torch.equal(out_direct.output, out_runtime.output)
    assert torch.equal(
        out_direct.extra["tokens"],
        out_runtime.extra["tokens"],
    )
    assert torch.equal(
        out_direct.extra["saved_noise"],
        out_runtime.extra["saved_noise"],
    )
    assert torch.equal(
        out_direct.extra["log_probs"],
        out_runtime.extra["log_probs"],
    )


# ---------------------------------------------------------------------------
# Test 3: collector.collect() called twice with same seed → identical batch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collector_collect_twice_same_seed_bitwise() -> None:
    """Two collect() calls with the same prompts + seed → identical ExperienceBatch.

    Parity is bitwise on every tensor field. The reward fn is
    deterministic, so the rewards also match.
    """
    from vrl.rollouts.collectors import (
        NextStep1CollectorConfig,
        build_rollout_collector,
    )

    cfg = NextStep1CollectorConfig(
        n_samples_per_prompt=4,
        cfg_scale=4.5,
        num_flow_steps=4,
        noise_level=1.0,
        image_token_num=8,
        image_size=32,
        rescale_to_unit=True,
        max_text_length=16,
    )

    collector_a = build_rollout_collector(
        "nextstep_1",
        model=_StubPolicy(),
        reward_fn=_ConstantReward(),
        config=cfg,
    )
    collector_b = build_rollout_collector(
        "nextstep_1",
        model=_StubPolicy(),
        reward_fn=_ConstantReward(),
        config=cfg,
    )

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
    # saved_noise determinism — the contract NextStep1Policy.replay_forward
    # depends on. If this regresses, RL training silently uses fresh noise.
    assert torch.equal(
        batch_a.extras["saved_noise"],
        batch_b.extras["saved_noise"],
    )
    assert torch.equal(
        batch_a.extras["log_probs"],
        batch_b.extras["log_probs"],
    )
    assert torch.equal(
        batch_a.extras["token_mask"],
        batch_b.extras["token_mask"],
    )
    assert torch.equal(
        batch_a.extras["prompt_attention_mask"],
        batch_b.extras["prompt_attention_mask"],
    )
    assert torch.equal(
        batch_a.extras["uncond_input_ids"],
        batch_b.extras["uncond_input_ids"],
    )
    assert torch.equal(
        batch_a.extras["uncond_attention_mask"],
        batch_b.extras["uncond_attention_mask"],
    )
    assert torch.equal(batch_a.videos, batch_b.videos)
    assert batch_a.context == batch_b.context
    assert batch_a.prompts == batch_b.prompts


# ---------------------------------------------------------------------------
# Test 4: ExperienceBatch shape sanity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collector_experience_batch_shape() -> None:
    """ExperienceBatch from collector has the trainer/replay-expected fields and shapes.

    The replay contract for NextStep1Policy.replay_forward:
      - actions.shape == (B, L, D)
      - extras["saved_noise"].shape == (B, L, D)
      - extras["log_probs"].shape == (B, 1, L)  (CEA singleton time dim)
      - observations.shape == (B, 1, L_text)    (CEA singleton time dim)
    """
    from vrl.rollouts.collectors import (
        NextStep1CollectorConfig,
        build_rollout_collector,
    )

    cfg = NextStep1CollectorConfig(
        n_samples_per_prompt=4,
        cfg_scale=4.5,
        num_flow_steps=4,
        noise_level=1.0,
        image_token_num=8,
        image_size=32,
        rescale_to_unit=True,
        max_text_length=16,
    )
    policy = _StubPolicy()
    collector = build_rollout_collector(
        "nextstep_1",
        model=policy,
        reward_fn=_ConstantReward(),
        config=cfg,
    )
    try:
        batch = await collector.collect(["a red cube"], group_size=4, seed=11)
    finally:
        await collector.shutdown()

    B = 4
    L = cfg.image_token_num  # 8 image tokens
    D = policy.token_dim  # 4
    L_text = cfg.max_text_length  # 16

    # actions: continuous tokens [B, L, D]
    assert batch.actions.shape == (B, L, D)
    # saved_noise round-trips at full [B, L, D]
    assert batch.extras["saved_noise"].shape == (B, L, D)
    # log_probs: CEA singleton time dim → [B, 1, L]
    assert batch.extras["log_probs"].shape == (B, 1, L)
    # observations: prompt ids with singleton time dim → [B, 1, L_text]
    assert batch.observations.shape == (B, 1, L_text)
    # videos: reward layer expects T dim → [B, 3, 1, H, W]
    assert batch.videos.shape == (B, 3, 1, cfg.image_size, cfg.image_size)
    # token_mask: [B, L]
    assert batch.extras["token_mask"].shape == (B, L)
    # prompt-side replay artifacts
    assert batch.extras["prompt_attention_mask"].shape == (B, L_text)
    assert batch.extras["uncond_input_ids"].shape == (B, L_text)
    assert batch.extras["uncond_attention_mask"].shape == (B, L_text)
    # rewards / dones / group_ids
    assert batch.rewards.shape == (B,)
    assert batch.dones.shape == (B,)
    assert batch.group_ids.shape == (B,)
    # context carries replay-time scalars
    assert batch.context["cfg_scale"] == 4.5
    assert batch.context["num_flow_steps"] == 4
    assert batch.context["noise_level"] == 1.0
    assert batch.context["image_token_num"] == L
