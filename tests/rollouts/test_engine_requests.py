"""Tests for rollout-to-engine request construction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vrl.rollouts.engine_requests import RolloutEngineRequestBuilder


def test_engine_request_builder_reads_declared_sampling_fields() -> None:
    builder = RolloutEngineRequestBuilder(
        family="fake",
        task="t2i",
        request_prefix="fake",
        config=SimpleNamespace(alpha=1, window=(0, 2)),
        sampling_fields=("alpha", "window"),
        return_artifacts=("output",),
        default_task_type="text_to_image",
    )

    plan = builder.build(
        ["prompt"],
        3,
        {
            "seed": 7,
            "policy_version": 11,
            "target_text": "HELLO",
            "sample_metadata": {"difficulty": "easy"},
        },
    )

    assert plan.request.family == "fake"
    assert plan.request.task == "t2i"
    assert plan.request.samples_per_prompt == 3
    assert plan.request.policy_version == 11
    assert plan.request.sampling == {"alpha": 1, "window": [0, 2], "seed": 7}
    assert plan.request.return_artifacts == {"output"}
    assert plan.request.metadata == {
        "difficulty": "easy",
        "target_text": "HELLO",
        "task_type": "text_to_image",
    }
    assert plan.reward_metadata == plan.request.metadata
    assert plan.pack_metadata == plan.request.metadata


def test_engine_request_builder_fails_when_registry_field_is_missing() -> None:
    builder = RolloutEngineRequestBuilder(
        family="fake",
        task="t2i",
        request_prefix="fake",
        config=SimpleNamespace(alpha=1),
        sampling_fields=("alpha", "missing"),
        return_artifacts=("output",),
    )

    with pytest.raises(ValueError, match="missing sampling field 'missing'"):
        builder.build(["prompt"], 1, {})
