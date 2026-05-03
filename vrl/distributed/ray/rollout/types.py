"""Types shared by Ray-backed rollout execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vrl.distributed.ray.rollout.spec import RolloutRuntimeSpec
from vrl.engine.generation.microbatching import MicroBatchPlan
from vrl.engine.generation.protocols import PipelineChunkResult


@dataclass(frozen=True, slots=True)
class RayWorkerHandle:
    """Scheduler-visible metadata for one rollout worker actor."""

    worker_id: str
    node_id: str
    gpu_ids: tuple[int, ...] = ()
    actor: Any | None = None


@dataclass(slots=True)
class RayChunkResult:
    """Envelope returned by a rollout worker for one generation chunk."""

    request_id: str
    worker_id: str
    chunk: MicroBatchPlan
    output: PipelineChunkResult | None
    metrics: dict[str, Any] = field(default_factory=dict)
    policy_version: int | None = None
    error: str | None = None


__all__ = [
    "RayChunkResult",
    "RayWorkerHandle",
    "RolloutRuntimeSpec",
]
