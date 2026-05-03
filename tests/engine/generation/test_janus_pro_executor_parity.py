"""Parity tests for the Janus-Pro generation executor migration.

These tests pin three equivalences:

1. ``JanusProPipelineExecutor.forward`` called directly produces the
   same ``OutputBatch`` as routing the same request through the engine
   ``GenerationWorker`` adapter and through the full async
   ``GenerationRuntime``. The runtime path adds no extra ops on tensors.

2. ``JanusProCollector.collect`` invoked twice with the same prompts +
   ``group_size`` + ``seed`` produces bitwise-identical
   ``ExperienceBatch`` tensors. This is the parity contract for
   Phase 7 — same prompts + same seed ⇒ same trajectory ⇒ same
   advantages ⇒ same gradient.

3. ``ExperienceBatch`` shape sanity — tokens / log_probs / videos /
   observations / actions all match the trainer's expected layout.

The stubs below are bare-minimum: no DeepSeek-Janus weights are loaded.
They exercise the executor + collector wiring + the
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
from vrl.models.families.janus_pro.executor import JanusProPipelineExecutor

HIDDEN = 16
TEXT_VOCAB = 64
IMG_VOCAB = 1024


# ---------------------------------------------------------------------------
# Stubs (mirror tests/engine/generation/test_pipeline_janus_pro.py — kept
# local so the parity test can evolve independently).
# ---------------------------------------------------------------------------


class _StubTokenizer:
    pad_token_id: int = 0

    def __call__(
        self,
        formatted: list[str],
        return_tensors: str = "pt",
        padding: str | bool = True,
        truncation: bool = True,
        max_length: int = 256,
    ) -> dict[str, torch.Tensor]:
        del return_tensors, truncation
        seqs = [
            torch.tensor(
                [ord(c) % TEXT_VOCAB for c in s[:max_length]],
                dtype=torch.long,
            )
            for s in formatted
        ]
        L = max_length if padding == "max_length" else max(s.numel() for s in seqs)
        ids = torch.zeros(len(seqs), L, dtype=torch.long)
        mask = torch.zeros(len(seqs), L, dtype=torch.long)
        for i, s in enumerate(seqs):
            n = min(s.numel(), L)
            ids[i, :n] = s[:n]
            mask[i, :n] = 1
        return {"input_ids": ids, "attention_mask": mask}


class _StubProcessor:
    def __init__(self) -> None:
        self.tokenizer = _StubTokenizer()


@dataclass
class _StubPolicy:
    """Stand-in for ``JanusProPolicy`` exposing only what the executor needs.

    Crucially, ``sample_image_tokens`` consumes the global torch RNG so
    parity reduces to "did both executor calls see the same seeded RNG?"
    """

    image_token_num: int = 4
    sample_calls: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._processor = _StubProcessor()
        self._embed = nn.Embedding(TEXT_VOCAB, HIDDEN)
        self._lm = type(
            "_LM",
            (),
            {
                "get_input_embeddings": lambda _self=self: self._embed,
            },
        )()

    @property
    def processor(self) -> _StubProcessor:
        return self._processor

    @property
    def language_model(self) -> Any:
        return self._lm

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def model_family(self) -> str:
        return "janus_pro-stub"

    def sample_image_tokens(
        self,
        cond_inputs_embeds: torch.Tensor,
        uncond_inputs_embeds: torch.Tensor,
        cond_attention_mask: torch.Tensor,
        uncond_attention_mask: torch.Tensor,
        *,
        cfg_weight: float | None = None,
        temperature: float | None = None,
        image_token_num: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        L_img = image_token_num or self.image_token_num
        B = cond_inputs_embeds.shape[0]
        token_ids = torch.randint(0, IMG_VOCAB, (B, L_img), dtype=torch.long)
        log_probs = -torch.rand(B, L_img, dtype=torch.float32)
        self.sample_calls.append({"B": B, "L_img": L_img})
        return token_ids, log_probs

    def decode_image_tokens(
        self,
        image_token_ids: torch.Tensor,
        *,
        image_size: int = 384,
    ) -> torch.Tensor:
        B = image_token_ids.shape[0]
        sig = image_token_ids.float().mean(dim=-1, keepdim=True).view(B, 1, 1, 1)
        H = W = max(8, image_size // 8)
        return torch.zeros(B, 3, H, W) + sig.tanh().expand(-1, 3, H, W) * 0.5


class _ConstantReward:
    """Reward that returns a deterministic float keyed off the prompt + image."""

    async def score(self, rollout: Any) -> float:
        # Use prompt only so two calls with the same prompts get equal rewards.
        return float(abs(hash(rollout.trajectory.prompt)) % 1000) / 1000.0


# ---------------------------------------------------------------------------
# Test 1: direct executor vs runtime-routed executor
# ---------------------------------------------------------------------------


def _build_request(seed: int = 99, *, image_token_num: int = 4) -> GenerationRequest:
    return GenerationRequest(
        request_id="parity-test",
        family="janus_pro",
        task="ar_t2i",
        prompts=["a red cube"],
        samples_per_prompt=2,
        sampling={
            "cfg_weight": 5.0,
            "temperature": 1.0,
            "image_token_num": image_token_num,
            "image_size": 64,
            "max_text_length": 8,
            "seed": seed,
        },
        return_artifacts={"output", "token_ids", "token_log_probs"},
    )


def test_executor_direct_matches_worker_executed_bitwise() -> None:
    """Same request through executor.forward(...) vs Worker.execute(...) → identical tensors.

    Bitwise-equal: both code paths run the AR sampling under the same
    seeded global RNG; the worker path adds no extra ops.
    """
    policy_a = _StubPolicy(image_token_num=4)
    executor_a = JanusProPipelineExecutor(policy_a)
    request_a = _build_request()
    specs_a = GenerationIdFactory().build_sample_specs(request_a)
    out_direct = executor_a.forward(request_a, specs_a)

    # Route through worker (same code path the runtime would take, minus
    # the asyncio engine loop — which doesn't touch tensors).
    policy_b = _StubPolicy(image_token_num=4)
    executor_b = JanusProPipelineExecutor(policy_b)
    registry = FamilyPipelineRegistry()
    registry.register(executor_b)
    worker = GenerationWorker(registry)
    request_b = _build_request()
    out_via_worker_dict = worker.execute([request_b])
    out_via_worker = out_via_worker_dict[request_b.request_id]

    assert out_via_worker.error is None
    assert torch.equal(out_direct.output, out_via_worker.output)
    assert torch.equal(
        out_direct.extra["token_ids"],
        out_via_worker.extra["token_ids"],
    )
    assert torch.equal(
        out_direct.extra["token_log_probs"],
        out_via_worker.extra["token_log_probs"],
    )
    assert torch.equal(
        out_direct.extra["token_mask"],
        out_via_worker.extra["token_mask"],
    )
    assert torch.equal(
        out_direct.extra["prompt_input_ids"],
        out_via_worker.extra["prompt_input_ids"],
    )


@pytest.mark.asyncio
async def test_executor_direct_matches_runtime_engine_loop_bitwise() -> None:
    """Routing through the full GenerationRuntime + EngineLoop preserves tensors."""
    policy_a = _StubPolicy(image_token_num=4)
    executor_a = JanusProPipelineExecutor(policy_a)
    request_a = _build_request()
    specs_a = GenerationIdFactory().build_sample_specs(request_a)
    out_direct = executor_a.forward(request_a, specs_a)

    policy_b = _StubPolicy(image_token_num=4)
    executor_b = JanusProPipelineExecutor(policy_b)
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
        out_direct.extra["token_ids"],
        out_runtime.extra["token_ids"],
    )
    assert torch.equal(
        out_direct.extra["token_log_probs"],
        out_runtime.extra["token_log_probs"],
    )


# ---------------------------------------------------------------------------
# Test 2: collector.collect() called twice with same seed → identical batch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collector_collect_twice_same_seed_bitwise() -> None:
    """Two collect() calls with the same prompts + seed → identical ExperienceBatch.

    Parity is bitwise on every tensor field. The reward fn is also
    deterministic (prompt-keyed) so rewards match.
    """
    from vrl.rollouts.collectors import (
        JanusProCollectorConfig,
        build_rollout_collector,
    )

    cfg = JanusProCollectorConfig(
        n_samples_per_prompt=4,
        cfg_weight=5.0,
        temperature=1.0,
        image_token_num=4,
        image_size=64,
        rescale_to_unit=True,
        max_text_length=8,
    )

    collector_a = build_rollout_collector(
        "janus_pro",
        model=_StubPolicy(image_token_num=4),
        reward_fn=_ConstantReward(),
        config=cfg,
    )
    collector_b = build_rollout_collector(
        "janus_pro",
        model=_StubPolicy(image_token_num=4),
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
    assert torch.equal(batch_a.extras["log_probs"], batch_b.extras["log_probs"])
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
# Test 3: ExperienceBatch shape sanity (replay contract)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collector_experience_batch_shape() -> None:
    """ExperienceBatch from collector has the trainer-expected shapes.

    Specifically reproduces the layout that ``JanusProPolicy.replay_forward``
    and ``TokenLogProbEvaluator`` read:

    - ``observations.shape == (B, 1, L_text)``  (T=1 for AR)
    - ``actions.shape == (B, L_img)``
    - ``extras["log_probs"].shape == (B, 1, L_img)``
    - ``extras["token_mask"].shape == (B, L_img)``
    - ``extras["prompt_attention_mask"].shape == (B, L_text)``
    - ``videos.shape == (B, 3, 1, H, W)``
    """
    from vrl.rollouts.collectors import (
        JanusProCollectorConfig,
        build_rollout_collector,
    )

    cfg = JanusProCollectorConfig(
        n_samples_per_prompt=2,
        image_token_num=4,
        image_size=64,
        max_text_length=8,
    )
    collector = build_rollout_collector(
        "janus_pro",
        model=_StubPolicy(image_token_num=4),
        reward_fn=_ConstantReward(),
        config=cfg,
    )
    try:
        batch = await collector.collect(["a red cube"], group_size=4, seed=11)
    finally:
        await collector.shutdown()

    B, L_img = 4, 4
    L_text = 8
    assert batch.observations.shape == (B, 1, L_text)
    assert batch.actions.shape == (B, L_img)
    assert batch.rewards.shape == (B,)
    assert batch.dones.shape == (B,)
    assert batch.group_ids.shape == (B,)
    assert batch.extras["log_probs"].shape == (B, 1, L_img)
    assert batch.extras["token_mask"].shape == (B, L_img)
    assert batch.extras["prompt_attention_mask"].shape == (B, L_text)
    assert batch.extras["uncond_input_ids"].shape == (B, L_text)
    assert batch.extras["uncond_attention_mask"].shape == (B, L_text)
    # Videos: [B, 3, T=1, H, W]
    assert batch.videos.dim() == 5
    assert batch.videos.shape[0] == B
    assert batch.videos.shape[1] == 3
    assert batch.videos.shape[2] == 1
    # Context carries cfg_weight
    assert batch.context.get("cfg_weight") == 5.0


@pytest.mark.asyncio
async def test_collector_multi_prompt_group_ids() -> None:
    """Multi-prompt collect: group_ids are prompt-major [0,0,...,1,1,...]."""
    from vrl.rollouts.collectors import (
        JanusProCollectorConfig,
        build_rollout_collector,
    )

    cfg = JanusProCollectorConfig(
        n_samples_per_prompt=2,
        image_token_num=4,
        image_size=64,
        max_text_length=8,
    )
    collector = build_rollout_collector(
        "janus_pro",
        model=_StubPolicy(image_token_num=4),
        reward_fn=_ConstantReward(),
        config=cfg,
    )
    try:
        batch = await collector.collect(["a", "b"])
    finally:
        await collector.shutdown()

    assert batch.group_ids.tolist() == [0, 0, 1, 1]
    assert batch.actions.shape[0] == 4
