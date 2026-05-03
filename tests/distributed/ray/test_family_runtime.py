"""Tests for family-specific Ray rollout runtime specs."""

from __future__ import annotations

import pickle

import pytest
import torch

from vrl.config.loader import load_config
from vrl.distributed.ray import build_family_ray_rollout_runtime_inputs
from vrl.engine.generation import ChunkedFamilyPipelineExecutor
from vrl.engine.generation.gather import DiffusionChunkGatherer
from vrl.models.families.janus_pro.executor import JanusProChunkGatherer
from vrl.models.families.nextstep_1.executor import NextStep1ChunkGatherer


@pytest.mark.parametrize(
    ("experiment", "family", "expected_task", "expected_gatherer"),
    [
        ("sd3_5_ocr_grpo", "sd3_5", "t2i", DiffusionChunkGatherer),
        ("wan_2_1_1_3b_grpo", "wan_2_1", "t2v", DiffusionChunkGatherer),
        ("cosmos_predict2_2b_grpo", "cosmos", "v2w", DiffusionChunkGatherer),
        ("janus_pro_1b_grpo", "janus_pro", "ar_t2i", JanusProChunkGatherer),
        ("nextstep_1_ocr_grpo", "nextstep_1", "ar_t2i", NextStep1ChunkGatherer),
    ],
)
def test_family_ray_rollout_inputs_are_serializable_and_pure(
    experiment: str,
    family: str,
    expected_task: str,
    expected_gatherer: type,
) -> None:
    cfg = load_config(
        f"experiment/{experiment}",
        overrides=[
            "distributed.backend=ray",
            "distributed.rollout.num_workers=1",
            "distributed.rollout.gpus_per_worker=0",
            "distributed.rollout.cpus_per_worker=1",
        ],
    )

    inputs = build_family_ray_rollout_runtime_inputs(
        cfg,
        family,
        weight_dtype=torch.bfloat16,
        executor_kwargs={"sample_batch_size": 2},
    )

    assert inputs is not None
    assert pickle.loads(pickle.dumps(inputs.runtime_spec)) == inputs.runtime_spec
    assert inputs.runtime_spec.family == family
    assert inputs.runtime_spec.task == expected_task
    assert inputs.runtime_spec.policy_version == 0
    assert inputs.runtime_spec.executor_kwargs == {"sample_batch_size": 2}
    assert isinstance(inputs.gatherer, expected_gatherer)
    assert not isinstance(inputs.gatherer, ChunkedFamilyPipelineExecutor)


def test_family_ray_rollout_inputs_return_none_for_local_backend() -> None:
    cfg = load_config("experiment/sd3_5_ocr_grpo")

    assert (
        build_family_ray_rollout_runtime_inputs(
            cfg,
            "sd3_5",
            weight_dtype=torch.bfloat16,
        )
        is None
    )


def test_diffusion_ray_runtime_spec_uses_worker_primitive_device_and_dtype() -> None:
    cfg = load_config(
        "experiment/sd3_5_ocr_grpo",
        overrides=[
            "distributed.backend=ray",
            "distributed.rollout.num_workers=1",
            "distributed.rollout.gpus_per_worker=1",
        ],
    )

    inputs = build_family_ray_rollout_runtime_inputs(
        cfg,
        "sd3_5",
        weight_dtype=torch.float16,
    )

    assert inputs is not None
    assert inputs.runtime_spec.build_spec is not None
    assert inputs.runtime_spec.build_spec["device"] == "cuda"
    assert inputs.runtime_spec.build_spec["dtype"] == "float16"
