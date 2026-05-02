"""Types shared by Ray-backed rollout execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vrl.executors.base import PipelineChunkResult
from vrl.executors.planning import MicroBatchPlan
from vrl.ipc.protocol import ArtifactRef


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
    output: PipelineChunkResult | ArtifactRef | None
    metrics: dict[str, Any] = field(default_factory=dict)
    policy_version: int | None = None
    error: str | None = None


__all__ = [
    "RayChunkResult",
    "RayWorkerHandle",
]
