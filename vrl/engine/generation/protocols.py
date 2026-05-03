"""Generation pipeline executor protocols."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vrl.engine.generation.microbatching import MicroBatchPlan
    from vrl.engine.generation.types import (
        GenerationRequest,
        GenerationSampleSpec,
        OutputBatch,
        WorkloadSignature,
    )


class PipelineChunkResult(Protocol):
    """Family-specific chunk payload returned before final OutputBatch gather."""


@runtime_checkable
class FamilyPipelineExecutor(Protocol):
    """Family-specific model execution unit."""

    family: str
    task: str

    def workload_signature(
        self,
        request: GenerationRequest,
    ) -> WorkloadSignature:
        ...

    def forward(
        self,
        request: GenerationRequest,
        sample_specs: list[GenerationSampleSpec],
    ) -> OutputBatch:
        ...


@runtime_checkable
class BatchedFamilyPipelineExecutor(FamilyPipelineExecutor, Protocol):
    """Optional extension for same-config request fusion."""

    def forward_batch(
        self,
        requests: list[GenerationRequest],
        sample_specs_by_request: dict[str, list[GenerationSampleSpec]],
    ) -> dict[str, OutputBatch]:
        ...


@runtime_checkable
class ChunkedFamilyPipelineExecutor(FamilyPipelineExecutor, Protocol):
    """Optional extension for distributed chunk execution."""

    def forward_chunk(
        self,
        request: GenerationRequest,
        chunk: MicroBatchPlan,
    ) -> PipelineChunkResult:
        ...

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_specs: Sequence[GenerationSampleSpec],
        chunks: Sequence[PipelineChunkResult],
    ) -> OutputBatch:
        ...
