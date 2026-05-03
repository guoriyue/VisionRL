"""Chunk contract tests for AR family pipeline executors."""

from __future__ import annotations

import asyncio

import torch

from tests.engine.generation.test_pipeline_janus_pro import (
    _request as _janus_request,
)
from tests.engine.generation.test_pipeline_janus_pro import (
    _StubPolicy as _JanusStubPolicy,
)
from tests.engine.generation.test_pipeline_nextstep_1 import (
    _request as _nextstep_request,
)
from tests.engine.generation.test_pipeline_nextstep_1 import (
    _StubPolicy as _NextStepStubPolicy,
)
from vrl.engine.generation import (
    FamilyPipelineRegistry,
    GenerationIdFactory,
    LocalRolloutWorkerPool,
    LocalWorkerSpec,
)
from vrl.executors import ChunkedFamilyPipelineExecutor
from vrl.executors.microbatching import MicroBatchPlan
from vrl.models.families.janus_pro.executor import (
    JanusProChunkGatherer,
    JanusProPipelineExecutor,
)
from vrl.models.families.nextstep_1.executor import (
    NextStep1ChunkGatherer,
    NextStep1PipelineExecutor,
)


def test_janus_pro_forward_chunk_and_gather_use_prompt_major_order() -> None:
    policy = _JanusStubPolicy(image_token_num=4)
    executor = JanusProPipelineExecutor(policy)
    request = _janus_request(
        prompts=["red cube", "blue sphere"],
        samples_per_prompt=4,
        image_token_num=4,
        image_size=64,
        max_text_length=8,
        seed=17,
    )
    sample_specs = GenerationIdFactory().build_sample_specs(request)
    plans = [
        MicroBatchPlan(0, "red cube", 0, 2),
        MicroBatchPlan(0, "red cube", 2, 2),
        MicroBatchPlan(1, "blue sphere", 0, 2),
        MicroBatchPlan(1, "blue sphere", 2, 2),
    ]

    chunks = [executor.forward_chunk(request, plan) for plan in plans]
    gatherer = JanusProChunkGatherer()
    output = gatherer.gather_chunks(
        request,
        sample_specs,
        [chunks[3], chunks[1], chunks[2], chunks[0]],
    )

    ordered_chunks = sorted(
        chunks,
        key=lambda chunk: (chunk.prompt_index, chunk.sample_start),
    )
    assert isinstance(executor, ChunkedFamilyPipelineExecutor)
    assert not isinstance(gatherer, ChunkedFamilyPipelineExecutor)
    assert [call["B"] for call in policy.sample_calls] == [2, 2, 2, 2]
    assert [spec.sample_id for spec in output.sample_specs] == [
        spec.sample_id for spec in sample_specs
    ]
    assert torch.equal(
        output.extra["token_ids"],
        torch.cat([chunk.token_ids for chunk in ordered_chunks], dim=0),
    )
    assert output.extra["token_log_probs"].shape == (8, 4)
    assert output.extra["token_mask"].shape == (8, 4)
    assert output.output.shape[0] == 8
    assert output.rollout_trajectory_data is None
    assert output.metrics is not None
    assert output.metrics.micro_batches == 4


def test_nextstep_1_forward_chunk_and_gather_use_prompt_major_order() -> None:
    policy = _NextStepStubPolicy()
    executor = NextStep1PipelineExecutor(policy)
    request = _nextstep_request(
        prompts=["red cube", "blue sphere"],
        samples_per_prompt=4,
        image_token_num=8,
        image_size=16,
        seed=19,
    )
    sample_specs = GenerationIdFactory().build_sample_specs(request)
    plans = [
        MicroBatchPlan(0, "red cube", 0, 2),
        MicroBatchPlan(0, "red cube", 2, 2),
        MicroBatchPlan(1, "blue sphere", 0, 2),
        MicroBatchPlan(1, "blue sphere", 2, 2),
    ]

    chunks = [executor.forward_chunk(request, plan) for plan in plans]
    gatherer = NextStep1ChunkGatherer()
    output = gatherer.gather_chunks(
        request,
        sample_specs,
        [chunks[2], chunks[0], chunks[3], chunks[1]],
    )

    ordered_chunks = sorted(
        chunks,
        key=lambda chunk: (chunk.prompt_index, chunk.sample_start),
    )
    expected_tokens = torch.cat(
        [chunk.tokens for chunk in ordered_chunks],
        dim=0,
    )
    expected_log_probs = torch.cat(
        [chunk.log_probs for chunk in ordered_chunks],
        dim=0,
    )
    assert isinstance(executor, ChunkedFamilyPipelineExecutor)
    assert not isinstance(gatherer, ChunkedFamilyPipelineExecutor)
    assert policy.sample_calls == 4
    assert policy.decode_calls == 4
    assert [spec.sample_id for spec in output.sample_specs] == [
        spec.sample_id for spec in sample_specs
    ]
    assert torch.equal(output.extra["tokens"], expected_tokens)
    assert torch.equal(output.extra["log_probs"], expected_log_probs)
    assert output.rollout_trajectory_data is not None
    assert torch.equal(
        output.rollout_trajectory_data.rollout_log_probs,
        expected_log_probs,
    )
    assert output.rollout_trajectory_data.denoising_env is None
    assert output.rollout_trajectory_data.dit_trajectory is None
    assert output.extra["images_for_reward"].shape == output.output.shape
    assert output.metrics is not None
    assert output.metrics.micro_batches == 4


def test_ar_executors_run_through_local_prompt_sample_chunk_pool() -> None:
    cases = [
        (
            JanusProPipelineExecutor(_JanusStubPolicy(image_token_num=4)),
            _janus_request(
                prompts=["red cube", "blue sphere"],
                samples_per_prompt=4,
                image_token_num=4,
                image_size=64,
                max_text_length=8,
                seed=23,
            ),
        ),
        (
            NextStep1PipelineExecutor(_NextStepStubPolicy()),
            _nextstep_request(
                prompts=["red cube", "blue sphere"],
                samples_per_prompt=4,
                image_token_num=8,
                image_size=16,
                seed=29,
            ),
        ),
    ]

    for executor, request in cases:
        request.sampling["sample_batch_size"] = 2
        registry = FamilyPipelineRegistry()
        registry.register(executor)
        pool = LocalRolloutWorkerPool(
            registry,
            [LocalWorkerSpec(worker_id="w0", device="cpu")],
        )

        output = asyncio.run(pool.execute(request))
        sample_specs = GenerationIdFactory().build_sample_specs(request)

        assert [spec.sample_id for spec in output.sample_specs] == [
            spec.sample_id for spec in sample_specs
        ]
        assert output.output.shape[0] == len(sample_specs)
        assert output.metrics is not None
        assert output.metrics.micro_batches == 4
