"""Tests for vrl.rollouts.collectors.janus_pro.JanusProCollector + TokenLogProbEvaluator.

Uses the same stub MultiModalityCausalLM as ``tests/models/test_janus_wrapper.py``
so we can validate the end-to-end CEA pipeline (collect → evaluate →
TokenGRPO loss) without downloading real Janus weights.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from vrl.algorithms.grpo_token import TokenGRPO, TokenGRPOConfig
from vrl.models.families.janus_pro.policy import (
    JANUS_IMAGE_VOCAB_SIZE,
    JanusProConfig,
    JanusProPolicy,
)
from vrl.rollouts.collectors.janus_pro import JanusProCollector, JanusProCollectorConfig
from vrl.rollouts.evaluators.lm import TokenLogProbEvaluator
from vrl.rollouts.evaluators.types import SignalRequest


HIDDEN = 32
TEXT_VOCAB = 64


class _StubLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(TEXT_VOCAB, HIDDEN)

    # ``model`` is a property rather than an attribute so nn.Module does NOT
    # register the stub as a submodule of itself — that cycle would cause
    # ``train()`` / ``apply()`` to recurse forever under OnlineTrainer.
    @property
    def model(self) -> "_StubLM":
        return self

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(self, inputs_embeds=None, attention_mask=None,
                use_cache=False, past_key_values=None, output_hidden_states=False):
        return SimpleNamespace(
            last_hidden_state=inputs_embeds,
            past_key_values=past_key_values,
        )


class _StubVQ(nn.Module):
    def decode_code(self, ids: torch.Tensor, shape: list[int]) -> torch.Tensor:
        B, _, h, w = shape
        # Image-content varies with token ids, so reward fns can differentiate.
        sig = ids.float().mean(dim=-1, keepdim=True).view(B, 1, 1, 1)
        return torch.zeros(B, 3, h * 16, w * 16) + sig.expand(-1, 3, h * 16, w * 16) * 1e-3


class _StubMMGPT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _StubLM()
        self.gen_vision_model = _StubVQ()
        self.gen_head = nn.Linear(HIDDEN, JANUS_IMAGE_VOCAB_SIZE)
        self.gen_aligner = nn.Identity()
        self.gen_embed = nn.Embedding(JANUS_IMAGE_VOCAB_SIZE, HIDDEN)

    def prepare_gen_img_embeds(self, ids: torch.Tensor) -> torch.Tensor:
        return self.gen_embed(ids)


class _StubProcessor:
    """Minimal stand-in for VLChatProcessor — just exposes a tokenizer."""

    class _Tokenizer:
        def __call__(self, formatted: list[str], return_tensors: str = "pt",
                     padding: bool = True, truncation: bool = True,
                     max_length: int = 256) -> dict[str, torch.Tensor]:
            # Each character → token id (mod TEXT_VOCAB), all sequences padded
            # to the longest in the batch.
            seqs = [
                torch.tensor([ord(c) % TEXT_VOCAB for c in s[: max_length]],
                             dtype=torch.long)
                for s in formatted
            ]
            L = max(s.numel() for s in seqs)
            ids = torch.zeros(len(seqs), L, dtype=torch.long)
            mask = torch.zeros(len(seqs), L, dtype=torch.long)
            for i, s in enumerate(seqs):
                ids[i, : s.numel()] = s
                mask[i, : s.numel()] = 1
            return {"input_ids": ids, "attention_mask": mask}

    def __init__(self) -> None:
        self.tokenizer = self._Tokenizer()


class _ConstReward:
    """Reward scored per-Rollout — matches vrl.rewards.base.RewardFunction."""

    async def score(self, rollout) -> float:
        return float(rollout.trajectory.output.flatten().mean().item())


class _BatchReward:
    """Reward exposing score_batch — exercises the batched-fast-path."""

    async def score_batch(self, rollouts) -> list[float]:
        return [float(r.trajectory.output.flatten().mean().item())
                for r in rollouts]

    async def score(self, rollout) -> float:
        return float(rollout.trajectory.output.flatten().mean().item())


@pytest.fixture
def stub_collector() -> JanusProCollector:
    cfg = JanusProConfig(use_lora=False)
    model = JanusProPolicy(config=cfg, mmgpt=_StubMMGPT(), processor=_StubProcessor())
    return JanusProCollector(
        model=model,
        reward_fn=_ConstReward(),
        config=JanusProCollectorConfig(
            n_samples_per_prompt=2,
            cfg_weight=2.0,
            image_token_num=4,
            image_size=64,
        ),
    )


# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------


class TestCollect:
    def test_returns_well_formed_batch(self, stub_collector: JanusProCollector) -> None:
        prompts = ["a cat", "a dog"]
        batch = asyncio.run(stub_collector.collect(prompts))

        # 2 prompts × 2 samples = 4 rollouts
        assert batch.actions.shape == (4, 4)        # [B, L_img]
        assert batch.rewards.shape == (4,)
        assert batch.group_ids.tolist() == [0, 0, 1, 1]
        assert batch.dones.shape == (4,)
        assert batch.dones.all()
        assert len(batch.prompts) == 4

    def test_extras_present(self, stub_collector: JanusProCollector) -> None:
        batch = asyncio.run(stub_collector.collect(["x", "y"]))
        for key in (
            "log_probs", "prompt_attention_mask",
            "uncond_input_ids", "uncond_attention_mask", "token_mask",
        ):
            assert key in batch.extras, f"missing {key}"
        # log_probs is [B, T=1, L_img] so OnlineTrainer._step_cea's
        # ``old_log_probs[:, j]`` slice at j=0 yields [B, L_img].
        assert batch.extras["log_probs"].shape == (4, 1, 4)
        assert batch.extras["token_mask"].shape == (4, 4)

    def test_observations_have_timestep_dim(
        self, stub_collector: JanusProCollector,
    ) -> None:
        """OnlineTrainer reads ``num_timesteps = batch.observations.shape[1]``;
        AR has a single effective step so shape must be [B, 1, L_text]."""
        batch = asyncio.run(stub_collector.collect(["x", "y"]))
        assert batch.observations.dim() == 3
        assert batch.observations.shape[0] == 4
        assert batch.observations.shape[1] == 1

    def test_group_size_overrides_n_samples(
        self, stub_collector: JanusProCollector,
    ) -> None:
        """OnlineTrainer passes ``group_size=`` and must override
        the collector's n_samples_per_prompt default."""
        batch = asyncio.run(stub_collector.collect(["x"], group_size=3))
        assert batch.actions.shape[0] == 3

    def test_videos_have_T_dim(self, stub_collector: JanusProCollector) -> None:
        """Reward layer convention: videos = [B, 3, T, H, W] with T=1 for images."""
        batch = asyncio.run(stub_collector.collect(["x"]))
        assert batch.videos.dim() == 5
        assert batch.videos.shape[2] == 1


# ---------------------------------------------------------------------------
# Reward routing — silent-failure red lines
# ---------------------------------------------------------------------------


def _build_collector_with_reward(reward: object) -> JanusProCollector:
    cfg = JanusProConfig(use_lora=False)
    model = JanusProPolicy(config=cfg, mmgpt=_StubMMGPT(), processor=_StubProcessor())
    return JanusProCollector(
        model=model, reward_fn=reward,
        config=JanusProCollectorConfig(
            n_samples_per_prompt=2, cfg_weight=2.0,
            image_token_num=4, image_size=64,
        ),
    )


class TestRewardRouting:
    def test_score_batch_fast_path(self) -> None:
        """Rewards exposing ``score_batch`` should take the batched path."""
        collector = _build_collector_with_reward(_BatchReward())
        batch = asyncio.run(collector.collect(["x"]))
        assert batch.rewards.shape == (2,)

    def test_per_rollout_fallback(self) -> None:
        """Rewards without ``score_batch`` fall through to per-rollout score."""
        collector = _build_collector_with_reward(_ConstReward())
        batch = asyncio.run(collector.collect(["x"]))
        assert batch.rewards.shape == (2,)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_logprob_shape(self, stub_collector: JanusProCollector) -> None:
        batch = asyncio.run(stub_collector.collect(["x", "y"]))
        evaluator = TokenLogProbEvaluator()
        signals = evaluator.evaluate(
            collector=stub_collector,
            model=stub_collector.model,
            batch=batch,
            ref_model=None,
            signal_request=SignalRequest(need_ref=False),
        )
        # Evaluator returns [B, L_img] (2D); collector's stored log_probs are
        # [B, 1, L_img] to match the OnlineTrainer timestep-dim contract.
        assert signals.log_prob.shape == batch.extras["log_probs"].shape[::2]
        assert signals.dist_family == "categorical"
        assert signals.aux.get("token_mask") is not None

    def test_need_ref_without_lora_or_ref_model_raises(
        self, stub_collector: JanusProCollector,
    ) -> None:
        """Silent-failure red-line: must raise, NOT silently set ref_lp = lp.

        Background: when ``use_lora=False`` and no explicit ``ref_model``
        is given, a naive evaluator could fall back to a no-op
        ``disable_adapter`` and end up scoring the same model twice —
        ref_lp == log_prob → KL ≡ 0 — without anyone noticing.
        """
        batch = asyncio.run(stub_collector.collect(["x"]))
        evaluator = TokenLogProbEvaluator()
        with pytest.raises(RuntimeError, match="ref_model"):
            evaluator.evaluate(
                collector=stub_collector,
                model=stub_collector.model,
                batch=batch,
                ref_model=None,
                signal_request=SignalRequest(need_ref=True),
            )

    def test_need_ref_with_explicit_ref_model_works(
        self, stub_collector: JanusProCollector,
    ) -> None:
        """When the caller hands in a real ref_model, evaluate runs fine."""
        # Build a second JanusProPolicy as the frozen ref policy.
        cfg = JanusProConfig(use_lora=False)
        ref = JanusProPolicy(config=cfg, mmgpt=_StubMMGPT(), processor=_StubProcessor())

        batch = asyncio.run(stub_collector.collect(["x"]))
        evaluator = TokenLogProbEvaluator()
        signals = evaluator.evaluate(
            collector=stub_collector,
            model=stub_collector.model,
            batch=batch,
            ref_model=ref,
            signal_request=SignalRequest(need_ref=True),
        )
        assert signals.ref_log_prob is not None
        assert signals.ref_log_prob.shape == signals.log_prob.shape


# ---------------------------------------------------------------------------
# End-to-end CEA pipeline
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_loss_backprops(self, stub_collector: JanusProCollector) -> None:
        """Full collect → evaluate → loss → backward path runs without errors."""
        # Re-build with LoRA-free trainable head so we have grads.
        cfg = JanusProConfig(use_lora=False)
        mmgpt = _StubMMGPT()
        # Make gen_head trainable (default JanusProPolicy freezes everything)
        for p in mmgpt.gen_head.parameters():
            p.requires_grad_(True)
        # Bypass the freeze: call ctor first then re-enable gen_head grad.
        model = JanusProPolicy(config=cfg, mmgpt=mmgpt, processor=_StubProcessor())
        for p in model.mmgpt.gen_head.parameters():
            p.requires_grad_(True)
        stub_collector.model = model

        batch = asyncio.run(stub_collector.collect(["a"]))
        evaluator = TokenLogProbEvaluator()
        algorithm = TokenGRPO(TokenGRPOConfig(init_kl_coef=0.0, eps_clip=0.5))

        advantages = algorithm.compute_advantages_from_tensors(
            batch.rewards, batch.group_ids,
        )
        signals = evaluator.evaluate(
            collector=stub_collector, model=model, batch=batch,
            ref_model=None,
            signal_request=SignalRequest(need_ref=False),
        )
        # OnlineTrainer slices [:, 0] before passing; replicate that here.
        loss, metrics = algorithm.compute_signal_loss(
            signals, advantages, batch.extras["log_probs"][:, 0],
        )

        # Loss is a finite scalar with grad
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        loss.backward()
        # gen_head should now have a gradient
        grad = next(model.mmgpt.gen_head.parameters()).grad
        assert grad is not None
        assert torch.isfinite(grad).all()

    def test_advantages_shape_and_zero_within_group(
        self, stub_collector: JanusProCollector,
    ) -> None:
        batch = asyncio.run(stub_collector.collect(["a", "b"]))
        algo = TokenGRPO(TokenGRPOConfig(global_std=False))
        adv = algo.compute_advantages_from_tensors(batch.rewards, batch.group_ids)
        assert adv.shape == batch.rewards.shape
        # Per-group normalisation → mean within each group ≈ 0
        for gid in batch.group_ids.unique():
            mask = batch.group_ids == gid
            if mask.sum() > 1:
                assert adv[mask].mean().abs().item() < 1e-5
