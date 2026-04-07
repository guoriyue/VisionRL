"""Tests for expert cache signature and remapping."""

from __future__ import annotations

import torch
import pytest

from wm_infra.ops.expert_cache import ExpertCache


class _DummyStream:
    pass


def _make_cache(monkeypatch) -> ExpertCache:
    monkeypatch.setattr(ExpertCache, "_ensure_pinned", staticmethod(lambda tensor: tensor))
    monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: _DummyStream())

    weights = torch.zeros(2, 2, 2)
    return ExpertCache(weights, weights.clone(), weights.clone(), max_experts_in_gpu=2, device="cpu")


def test_remap_expert_ids_accepts_only_the_current_signature(monkeypatch):
    cache = _make_cache(monkeypatch)
    cache._remap_table[:] = torch.tensor([1, 0], dtype=torch.int64)

    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    remapped = cache.remap_expert_ids(topk_ids)

    assert remapped.tolist() == [[1, 0], [0, 1]]


def test_remap_expert_ids_rejects_legacy_keyword(monkeypatch):
    cache = _make_cache(monkeypatch)
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int64)

    with pytest.raises(TypeError):
        cache.remap_expert_ids(topk_ids, expert_to_slot={0: 0})
