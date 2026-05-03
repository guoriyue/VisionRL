"""Autoregressive generation helpers."""

from vrl.engine.generation.ar.executor_base import (
    ARChunkResult,
    ARPipelineExecutorBase,
    align_pair,
    chunk_seed_offset,
    expand_prompt_major_prompts,
    max_peak_memory_mb,
    ordered_chunks,
    parse_ar_generation_spec,
    peak_memory_mb,
    require_rows,
    validate_chunk,
)
from vrl.engine.generation.ar.sequence import ActiveSequence, ARSequenceKey
from vrl.engine.generation.ar.spec import ARGenerationSpec
from vrl.engine.generation.ar.token_scheduler import ARTokenBatch, ARTokenScheduler
from vrl.models.ar import ARStepResult

__all__ = [
    "ARChunkResult",
    "ARGenerationSpec",
    "ARPipelineExecutorBase",
    "ARSequenceKey",
    "ARStepResult",
    "ARTokenBatch",
    "ARTokenScheduler",
    "ActiveSequence",
    "align_pair",
    "chunk_seed_offset",
    "expand_prompt_major_prompts",
    "max_peak_memory_mb",
    "ordered_chunks",
    "parse_ar_generation_spec",
    "peak_memory_mb",
    "require_rows",
    "validate_chunk",
]
