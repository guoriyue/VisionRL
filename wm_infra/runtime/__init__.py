"""Runtime substrate exports.

Keep package import side effects minimal so low-level execution primitives can
be imported without pulling in env or trainer-facing modules.
"""

from wm_infra.runtime.execution import (
    BatchSignature,
    ExecutionBatchPolicy,
    ExecutionChunk,
    ExecutionEntity,
    ExecutionStats,
    ExecutionWorkItem,
    HomogeneousChunkScheduler,
    SchedulerDecision,
    build_execution_chunks,
    chunk_fill_ratio,
    summarize_execution_chunks,
)
__all__ = [
    "BatchSignature",
    "ExecutionBatchPolicy",
    "ExecutionChunk",
    "ExecutionEntity",
    "ExecutionStats",
    "ExecutionWorkItem",
    "HomogeneousChunkScheduler",
    "SchedulerDecision",
    "build_execution_chunks",
    "chunk_fill_ratio",
    "summarize_execution_chunks",
]
