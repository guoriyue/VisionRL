"""Helpers for gathering chunked generation executor outputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from vrl.engine.generation.types import (
    GenerationRequest,
    GenerationSampleSpec,
    OutputBatch,
)
from vrl.executors.base import ChunkedFamilyPipelineExecutor, PipelineChunkResult

if TYPE_CHECKING:
    from vrl.executors.diffusion import DiffusionChunkResult


@runtime_checkable
class ChunkGatherer(Protocol):
    """Pure chunk gather contract that does not require an executor/model."""

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_specs: Sequence[GenerationSampleSpec],
        chunks: Sequence[PipelineChunkResult],
    ) -> OutputBatch: ...


def require_chunked_executor(executor: Any) -> ChunkedFamilyPipelineExecutor:
    """Return executor if it exposes the distributed chunk contract."""

    forward_chunk = getattr(executor, "forward_chunk", None)
    gather_chunks = getattr(executor, "gather_chunks", None)
    if not callable(forward_chunk) or not callable(gather_chunks):
        raise TypeError(
            f"{type(executor).__name__} does not implement "
            "forward_chunk(...) and gather_chunks(...)",
        )
    return executor


def require_chunk_gatherer(gatherer: Any) -> ChunkGatherer:
    """Return gatherer if it exposes the pure chunk gather contract."""

    gather_chunks = getattr(gatherer, "gather_chunks", None)
    if not callable(gather_chunks):
        raise TypeError(
            f"{type(gatherer).__name__} does not implement gather_chunks(...)",
        )
    return gatherer


def gather_pipeline_chunks(
    gatherer: Any,
    request: GenerationRequest,
    sample_specs: Sequence[GenerationSampleSpec],
    chunks: Sequence[PipelineChunkResult],
) -> OutputBatch:
    """Gather family-specific chunk payloads into one canonical OutputBatch."""

    chunk_gatherer = require_chunk_gatherer(gatherer)
    return chunk_gatherer.gather_chunks(request, sample_specs, chunks)


def gather_diffusion_chunks(
    request: GenerationRequest,
    sample_specs: Sequence[GenerationSampleSpec],
    chunks: Sequence[PipelineChunkResult],
    *,
    model_family: str,
    respect_cfg_flag: bool = True,
) -> OutputBatch:
    """Gather diffusion chunks using only request metadata and CPU payloads."""

    from vrl.executors.diffusion import build_diffusion_output_batch

    sampling = request.sampling
    num_steps = int(sampling["num_steps"])
    guidance_scale = float(sampling["guidance_scale"])
    cfg_enabled = guidance_scale > 1.0
    if respect_cfg_flag:
        cfg_enabled = bool(sampling.get("cfg", True)) and cfg_enabled
    return build_diffusion_output_batch(
        request=request,
        sample_specs=list(sample_specs),
        prompts=list(request.prompts),
        chunks=cast("list[DiffusionChunkResult]", list(chunks)),
        num_steps=num_steps,
        fallback_context={
            "guidance_scale": guidance_scale,
            "cfg": cfg_enabled,
            "model_family": model_family,
        },
    )


@dataclass(frozen=True, slots=True)
class DiffusionChunkGatherer:
    """Pure gatherer for shared diffusion chunk payloads."""

    model_family: str
    respect_cfg_flag: bool = True

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_specs: Sequence[GenerationSampleSpec],
        chunks: Sequence[PipelineChunkResult],
    ) -> OutputBatch:
        return gather_diffusion_chunks(
            request,
            sample_specs,
            chunks,
            model_family=self.model_family,
            respect_cfg_flag=self.respect_cfg_flag,
        )


__all__ = [
    "ChunkGatherer",
    "DiffusionChunkGatherer",
    "gather_diffusion_chunks",
    "gather_pipeline_chunks",
    "require_chunk_gatherer",
    "require_chunked_executor",
]
