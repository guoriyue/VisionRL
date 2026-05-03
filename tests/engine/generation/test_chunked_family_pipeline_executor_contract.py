"""Tests for chunk-level family executor contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vrl.engine.generation import (
    FamilyPipelineRegistry,
    GenerationIdFactory,
    GenerationRequest,
    LocalRolloutWorkerPool,
    LocalWorkerSpec,
    OutputBatch,
    WorkloadSignature,
)
from vrl.engine.generation.gather import ChunkGatherer, require_chunk_gatherer
from vrl.executors import ChunkedFamilyPipelineExecutor, PipelineChunkResult
from vrl.executors.microbatching import MicroBatchPlan


@dataclass(slots=True)
class _ChunkResult(PipelineChunkResult):
    prompt_index: int
    sample_start: int
    sample_count: int


class _ChunkedExecutor(ChunkedFamilyPipelineExecutor):
    family = "fake"
    task = "t2i"

    def __init__(self) -> None:
        self.forward_chunk_calls: list[MicroBatchPlan] = []

    def workload_signature(self, request: GenerationRequest) -> WorkloadSignature:
        return WorkloadSignature.from_request(request)

    def forward(
        self,
        request: GenerationRequest,
        sample_specs: list[Any],
    ) -> OutputBatch:
        chunks = [
            self.forward_chunk(
                request,
                MicroBatchPlan(
                    prompt_index=spec.prompt_index,
                    prompt=spec.prompt,
                    sample_start=spec.sample_index,
                    sample_count=1,
                ),
            )
            for spec in sample_specs
        ]
        return self.gather_chunks(request, sample_specs, chunks)

    def forward_chunk(
        self,
        request: GenerationRequest,
        chunk: MicroBatchPlan,
    ) -> _ChunkResult:
        del request
        self.forward_chunk_calls.append(chunk)
        return _ChunkResult(
            prompt_index=chunk.prompt_index,
            sample_start=chunk.sample_start,
            sample_count=chunk.sample_count,
        )

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_specs: list[Any],
        chunks: list[_ChunkResult],
    ) -> OutputBatch:
        return OutputBatch(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=list(request.prompts),
            sample_specs=sample_specs,
            output=[
                (chunk.prompt_index, chunk.sample_start, chunk.sample_count) for chunk in chunks
            ],
        )


def _request() -> GenerationRequest:
    return GenerationRequest(
        request_id="req",
        family="fake",
        task="t2i",
        prompts=["p0", "p1"],
        samples_per_prompt=4,
        sampling={"sample_batch_size": 2},
    )


def test_chunked_executor_extends_family_executor_contract() -> None:
    executor = _ChunkedExecutor()
    assert isinstance(executor, ChunkedFamilyPipelineExecutor)
    assert isinstance(executor, ChunkGatherer)
    assert require_chunk_gatherer(executor) is executor

    request = _request()
    specs = GenerationIdFactory().build_sample_specs(request)
    output = executor.forward(request, specs)

    assert output.output == [
        (0, 0, 1),
        (0, 1, 1),
        (0, 2, 1),
        (0, 3, 1),
        (1, 0, 1),
        (1, 1, 1),
        (1, 2, 1),
        (1, 3, 1),
    ]


def test_local_worker_pool_executes_prompt_major_chunks() -> None:
    registry = FamilyPipelineRegistry()
    executor = _ChunkedExecutor()
    registry.register(executor)
    pool = LocalRolloutWorkerPool(
        registry,
        [LocalWorkerSpec(worker_id="w0", device="cpu")],
    )

    import asyncio

    output = asyncio.run(pool.execute(_request()))

    assert output.output == [
        (0, 0, 2),
        (0, 2, 2),
        (1, 0, 2),
        (1, 2, 2),
    ]
    assert [chunk.prompt_index for chunk in executor.forward_chunk_calls] == [
        0,
        0,
        1,
        1,
    ]


def test_local_worker_pool_rejects_multi_worker_specs() -> None:
    registry = FamilyPipelineRegistry()
    registry.register(_ChunkedExecutor())

    import pytest

    with pytest.raises(ValueError, match="Ray backend"):
        LocalRolloutWorkerPool(
            registry,
            [
                LocalWorkerSpec(worker_id="w0", device="cpu"),
                LocalWorkerSpec(worker_id="w1", device="cpu"),
            ],
        )
