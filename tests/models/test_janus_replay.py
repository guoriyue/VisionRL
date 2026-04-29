"""Tests for the Janus-Pro replay-forward refactor (SPRINT_ar_support.md §6.6).

Covers the real boundaries of the refactor:
  1. ``JanusProPolicy.replay_forward`` returns the documented dict schema.
  2. Logits remain grad-carrying when params are unfrozen.
  3. ``JanusProCollector.forward_step`` is a thin shim that delegates
     verbatim to ``model.replay_forward``.
  4. ``JanusProPolicy`` structurally satisfies ``AutoregressivePolicy``.
  5. ``TokenLogProbEvaluator.evaluate`` calls ``model.replay_forward``
     and NEVER routes through ``collector.forward_step`` any more.

Tests run on CPU in <1s — no real Janus weights are loaded.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from vrl.models.ar import AutoregressivePolicy
from vrl.models.families.janus_pro.policy import (
    JANUS_IMAGE_VOCAB_SIZE,
    JanusProConfig,
    JanusProPolicy,
)
from vrl.rollouts.collectors.janus_pro import (
    JanusProCollector,
    JanusProCollectorConfig,
)
from vrl.rollouts.evaluators.lm.token_logprob import TokenLogProbEvaluator
from vrl.rollouts.evaluators.types import SignalRequest
from vrl.rollouts.types import ExperienceBatch


HIDDEN = 32
TEXT_VOCAB = 64


# ---------------------------------------------------------------------------
# Stubs — mirror tests/models/test_janus_wrapper.py + tests/rollouts/...
# ---------------------------------------------------------------------------


class _StubLM(nn.Module):
    """Identity trunk: last_hidden_state == inputs_embeds."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(TEXT_VOCAB, HIDDEN)

    @property
    def model(self) -> "_StubLM":
        # Property — not attribute — so ``train()`` does not infinite-recurse.
        return self

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(
        self,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: object = None,
        output_hidden_states: bool = False,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            last_hidden_state=inputs_embeds,
            past_key_values=past_key_values,
        )


class _StubVQ(nn.Module):
    def decode_code(self, ids: torch.Tensor, shape: list[int]) -> torch.Tensor:
        B, _, h, w = shape
        return torch.zeros(B, 3, h * 16, w * 16)


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


def _build_stub_model(*, unfreeze_gen_head: bool = False) -> JanusProPolicy:
    cfg = JanusProConfig(use_lora=False)
    model = JanusProPolicy(config=cfg, mmgpt=_StubMMGPT(), processor=object())
    if unfreeze_gen_head:
        for p in model.mmgpt.gen_head.parameters():
            p.requires_grad_(True)
    return model


def _make_batch(B: int = 2, L_text: int = 5, L_img: int = 4) -> ExperienceBatch:
    """Fake ``ExperienceBatch`` with the keys ``replay_forward`` reads."""
    prompt_ids = torch.randint(0, TEXT_VOCAB, (B, L_text))
    prompt_mask = torch.ones(B, L_text, dtype=torch.long)
    image_token_ids = torch.randint(0, JANUS_IMAGE_VOCAB_SIZE, (B, L_img))
    return ExperienceBatch(
        observations=prompt_ids.unsqueeze(1),  # [B, 1, L_text] — OnlineTrainer shape
        actions=image_token_ids,
        rewards=torch.zeros(B),
        dones=torch.ones(B, dtype=torch.bool),
        group_ids=torch.arange(B),
        extras={
            "prompt_attention_mask": prompt_mask,
            "log_probs": torch.zeros(B, 1, L_img),
            "token_mask": torch.ones(B, L_img),
        },
        prompts=["x"] * B,
    )


# ---------------------------------------------------------------------------
# Test 1 — return-schema contract
# ---------------------------------------------------------------------------


def test_janus_replay_forward_returns_correct_keys() -> None:
    model = _build_stub_model()
    B, L_img = 2, 4
    batch = _make_batch(B=B, L_text=5, L_img=L_img)

    out = model.replay_forward(batch)

    assert set(out.keys()) == {"logits", "image_token_ids"}
    assert out["logits"].shape == (B, L_img, JANUS_IMAGE_VOCAB_SIZE)
    assert out["image_token_ids"].shape == (B, L_img)


# ---------------------------------------------------------------------------
# Test 2 — grad-carrying logits + integer target tokens
# ---------------------------------------------------------------------------


def test_janus_replay_logits_are_grad_carrying() -> None:
    model = _build_stub_model(unfreeze_gen_head=True)
    model.train()
    batch = _make_batch()

    out = model.replay_forward(batch)

    assert out["logits"].requires_grad is True
    # image_token_ids are GRPO targets — must be integer-typed, not float.
    assert out["image_token_ids"].dtype in (torch.long, torch.int32, torch.int64)


# ---------------------------------------------------------------------------
# Test 3 — collector no longer exposes a training-replay forward_step
# ---------------------------------------------------------------------------


def test_janus_collector_has_no_forward_step() -> None:
    """Sprint Task 7: the deprecated ``forward_step`` shim is gone.

    Train-time replay ownership lives on ``model.replay_forward`` and the
    evaluator calls the model directly. Collectors expose only ``collect()``.
    """
    collector = JanusProCollector(
        model=_build_stub_model(),
        reward_fn=None,
        config=JanusProCollectorConfig(image_token_num=4, image_size=64),
    )
    assert not hasattr(collector, "forward_step")


# ---------------------------------------------------------------------------
# Test 4 — structural Protocol conformance
# ---------------------------------------------------------------------------


def test_janus_protocol_structural_conformance() -> None:
    model = _build_stub_model()
    # ``AutoregressivePolicy`` is ``@runtime_checkable`` — isinstance must work.
    assert isinstance(model, AutoregressivePolicy)


# ---------------------------------------------------------------------------
# Test 5 — TokenLogProbEvaluator does NOT route through collector.forward_step
# ---------------------------------------------------------------------------


def test_evaluator_calls_model_replay() -> None:
    """Evaluator computes log_probs through ``model.replay_forward``.

    The ``collector`` parameter is retained on the evaluator signature for
    trainer-interface compatibility but its body never reads it. We pass a
    bare MagicMock (no spec) and verify (a) ``model.replay_forward`` is
    called, (b) the collector mock receives zero method calls.
    """
    model = _build_stub_model()
    batch = _make_batch()
    collector = MagicMock()  # plain mock — the evaluator must not touch it

    evaluator = TokenLogProbEvaluator()
    with patch.object(
        model, "replay_forward", wraps=model.replay_forward,
    ) as replay_spy:
        signals = evaluator.evaluate(
            collector=collector,
            model=model,
            batch=batch,
            ref_model=None,
            signal_request=SignalRequest(need_ref=False),
        )

    # Evaluator must compute log_prob via model.replay_forward.
    assert replay_spy.call_count >= 1
    # Evaluator must NOT touch the collector at all.
    assert collector.method_calls == []
    # Sanity: returned signal has the expected categorical shape.
    assert signals.dist_family == "categorical"
    assert signals.log_prob.shape == batch.actions.shape  # [B, L_img]
