"""Helpers for gathering chunked generation executor outputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vrl.engine.generation.types import (
    GenerationRequest,
    GenerationSampleSpec,
    OutputBatch,
)
from vrl.executors.base import ChunkedFamilyPipelineExecutor, PipelineChunkResult


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


def gather_pipeline_chunks(
    executor: Any,
    request: GenerationRequest,
    sample_specs: Sequence[GenerationSampleSpec],
    chunks: Sequence[PipelineChunkResult],
) -> OutputBatch:
    """Gather family-specific chunk payloads into one canonical OutputBatch."""

    chunked = require_chunked_executor(executor)
    return chunked.gather_chunks(request, sample_specs, chunks)


__all__ = [
    "gather_pipeline_chunks",
    "require_chunked_executor",
]
