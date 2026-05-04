"""Tests for rollout runtime input construction."""

from __future__ import annotations

import pickle

import pytest
import torch

from vrl.config.loader import load_config
from vrl.engine import ChunkedFamilyPipelineExecutor
from vrl.engine.gather import DiffusionChunkGatherer
from vrl.models.families.janus_pro.executor import JanusProChunkGatherer
from vrl.models.families.nextstep_1.executor import NextStep1ChunkGatherer
from vrl.rollouts.families.specs import get_rollout_family_entry
from vrl.rollouts.runtime.launch_inputs import (
    RolloutRuntimeInputs,
    build_rollout_runtime_inputs,
)


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
def test_rollout_runtime_inputs_are_serializable_and_registry_backed(
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
    entry = get_rollout_family_entry(family)

    inputs = build_rollout_runtime_inputs(
        cfg,
        family,
        weight_dtype=torch.bfloat16,
        executor_kwargs={"sample_batch_size": 2},
    )

    assert isinstance(inputs, RolloutRuntimeInputs)
    assert pickle.loads(pickle.dumps(inputs.runtime_spec)) == inputs.runtime_spec
    assert inputs.runtime_spec.family == family
    assert inputs.runtime_spec.task == expected_task
    assert inputs.runtime_spec.policy_version == 0
    assert inputs.runtime_spec.runtime_builder == entry.runtime_builder
    assert inputs.runtime_spec.executor_cls == entry.executor_cls
    assert inputs.runtime_spec.executor_kwargs == {"sample_batch_size": 2}
    assert isinstance(inputs.gatherer, expected_gatherer)
    assert not isinstance(inputs.gatherer, ChunkedFamilyPipelineExecutor)


def test_diffusion_runtime_spec_uses_worker_primitive_device_and_dtype() -> None:
    cfg = load_config(
        "experiment/sd3_5_ocr_grpo",
        overrides=[
            "distributed.backend=ray",
            "distributed.rollout.num_workers=1",
            "distributed.rollout.gpus_per_worker=1",
        ],
    )

    inputs = build_rollout_runtime_inputs(
        cfg,
        "sd3_5",
        weight_dtype=torch.float16,
    )

    assert isinstance(inputs, RolloutRuntimeInputs)
    assert inputs.runtime_spec.build_spec is not None
    assert inputs.runtime_spec.build_spec["device"] == "cuda"
    assert inputs.runtime_spec.build_spec["dtype"] == "float16"


def test_cosmos_runtime_inputs_include_reference_image_from_cfg() -> None:
    cfg = load_config(
        "experiment/cosmos_predict2_2b_grpo",
        overrides=[
            "distributed.backend=ray",
            "distributed.rollout.num_workers=1",
            "distributed.rollout.gpus_per_worker=0",
            "model.reference_image=/tmp/reference.png",
        ],
    )

    inputs = build_rollout_runtime_inputs(
        cfg,
        "cosmos_predict2",
        weight_dtype=torch.bfloat16,
    )

    assert isinstance(inputs, RolloutRuntimeInputs)
    assert inputs.runtime_spec.family == "cosmos"
    assert inputs.runtime_spec.executor_kwargs["reference_image"] == "/tmp/reference.png"
    assert inputs.runtime_spec.build_spec is not None
    assert inputs.runtime_spec.build_spec["extra"]["reference_image"] == "/tmp/reference.png"


def test_explicit_executor_kwargs_override_registry_defaults() -> None:
    cfg = load_config(
        "experiment/sd3_5_ocr_grpo",
        overrides=[
            "distributed.backend=ray",
            "distributed.rollout.num_workers=1",
            "distributed.rollout.gpus_per_worker=0",
            "rollout.sample_batch_size=8",
        ],
    )

    inputs = build_rollout_runtime_inputs(
        cfg,
        "sd3_5",
        weight_dtype=torch.bfloat16,
        executor_kwargs={"sample_batch_size": 3},
    )

    assert isinstance(inputs, RolloutRuntimeInputs)
    assert inputs.runtime_spec.executor_kwargs == {"sample_batch_size": 3}
