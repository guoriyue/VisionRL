"""Distributed execution planning for large rollout chunks."""

from __future__ import annotations

from dataclasses import dataclass

from vrl.distributed.ray.types import RayWorkerHandle
from vrl.engine.generation.types import GenerationRequest
from vrl.executors.microbatching import (
    MicroBatchPlan,
    plan_prompt_group_microbatches,
)


@dataclass(frozen=True, slots=True)
class DeviceAssignment:
    """Map one logical chunk to one rollout worker."""

    worker_id: str
    node_id: str
    gpu_ids: tuple[int, ...]
    chunk: MicroBatchPlan


class DistributedExecutionPlanner:
    """Plan chunk placement across Ray rollout workers."""

    def plan(
        self,
        request: GenerationRequest,
        workers: list[RayWorkerHandle],
    ) -> list[DeviceAssignment]:
        if not workers:
            raise ValueError("DistributedExecutionPlanner requires at least one worker")
        max_samples = int(
            request.sampling.get("sample_batch_size", request.samples_per_prompt),
        )
        execution_plan = plan_prompt_group_microbatches(
            request.prompts,
            samples_per_prompt=request.samples_per_prompt,
            max_samples_per_microbatch=max(1, max_samples),
        )
        assignments: list[DeviceAssignment] = []
        for idx, chunk in enumerate(execution_plan.micro_batches):
            worker = workers[idx % len(workers)]
            assignments.append(
                DeviceAssignment(
                    worker_id=worker.worker_id,
                    node_id=worker.node_id,
                    gpu_ids=worker.gpu_ids,
                    chunk=chunk,
                )
            )
        return assignments


__all__ = [
    "DeviceAssignment",
    "DistributedExecutionPlanner",
]
