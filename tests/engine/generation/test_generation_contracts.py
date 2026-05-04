"""Tests for generation request contracts."""

from __future__ import annotations

import pytest

from vrl.engine import (
    GenerationIdFactory,
    GenerationRequest,
)


def _request(
    request_id: str = "req-1",
    *,
    height: int = 512,
    width: int = 512,
    num_steps: int = 10,
    seed: int | None = 7,
) -> GenerationRequest:
    sampling = {
        "height": height,
        "width": width,
        "num_steps": num_steps,
    }
    if seed is not None:
        sampling["seed"] = seed
    return GenerationRequest(
        request_id=request_id,
        family="fake",
        task="t2i",
        prompts=["a test prompt"],
        samples_per_prompt=2,
        sampling=sampling,
        return_artifacts={"output", "rollout_trajectory_data"},
        metadata={"dataset": "unit"},
    )


def test_generation_request_validation() -> None:
    with pytest.raises(ValueError, match="prompts"):
        GenerationRequest(
            request_id="req",
            family="fake",
            task="t2i",
            prompts=[],
            samples_per_prompt=1,
        )

    with pytest.raises(ValueError, match="samples_per_prompt"):
        GenerationRequest(
            request_id="req",
            family="fake",
            task="t2i",
            prompts=["x"],
            samples_per_prompt=0,
        )

    with pytest.raises(ValueError, match="policy_version"):
        GenerationRequest(
            request_id="req",
            family="fake",
            task="t2i",
            prompts=["x"],
            samples_per_prompt=1,
            policy_version=-1,
        )


def test_generation_id_factory_is_deterministic() -> None:
    request = _request()
    specs = GenerationIdFactory().build_sample_specs(request)

    assert [spec.sample_id for spec in specs] == [
        "req-1:prompt:0:sample:0",
        "req-1:prompt:0:sample:1",
    ]
    assert [spec.seed for spec in specs] == [7, 8]
    assert {spec.group_id for spec in specs} == {"req-1:prompt:0"}
