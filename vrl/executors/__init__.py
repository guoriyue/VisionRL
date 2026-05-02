"""Engine-facing generation executor contracts."""

from vrl.executors.base import (
    BatchedFamilyPipelineExecutor,
    ChunkedFamilyPipelineExecutor,
    FamilyPipelineExecutor,
    PipelineChunkResult,
)
from vrl.executors.batching import forward_batch_by_merging_prompts

__all__ = [
    "BatchedFamilyPipelineExecutor",
    "ChunkedFamilyPipelineExecutor",
    "FamilyPipelineExecutor",
    "PipelineChunkResult",
    "forward_batch_by_merging_prompts",
]
