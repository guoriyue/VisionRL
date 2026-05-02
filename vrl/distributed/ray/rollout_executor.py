"""Distributed rollout executor that gathers Ray chunk results."""

from __future__ import annotations

import asyncio
from typing import Any

from vrl.distributed.ray.planning import DistributedExecutionPlanner
from vrl.distributed.ray.types import RayChunkResult, RayWorkerHandle
from vrl.distributed.ray.utils import require_ray
from vrl.engine.generation.gather import ChunkGatherer, gather_pipeline_chunks
from vrl.engine.generation.types import GenerationRequest, OutputBatch
from vrl.engine.generation.worker import GenerationIdFactory
from vrl.executors.base import PipelineChunkResult


class DistributedRolloutExecutor:
    """Execute one GenerationRequest across rollout workers."""

    def __init__(
        self,
        planner: DistributedExecutionPlanner,
        workers: list[RayWorkerHandle],
        gatherer: ChunkGatherer,
        *,
        id_factory: GenerationIdFactory | None = None,
    ) -> None:
        if not workers:
            raise ValueError("DistributedRolloutExecutor requires at least one worker")
        self.planner = planner
        self.workers = list(workers)
        self.gatherer = gatherer
        self.id_factory = id_factory or GenerationIdFactory()

    async def execute(self, request: GenerationRequest) -> OutputBatch:
        assignments = self.planner.plan(request, self.workers)
        worker_by_id = {worker.worker_id: worker for worker in self.workers}
        refs: list[Any] = []
        direct_results: list[RayChunkResult] = []

        for assignment in assignments:
            worker = worker_by_id[assignment.worker_id]
            actor = worker.actor
            if actor is None:
                raise RuntimeError(f"worker {worker.worker_id!r} has no actor")
            execute_chunk = actor.execute_chunk
            remote = getattr(execute_chunk, "remote", None)
            if callable(remote):
                refs.append(remote(request, assignment.chunk))
            else:
                direct_results.append(execute_chunk(request, assignment.chunk))

        if refs:
            ray = require_ray()
            remote_results = await asyncio.to_thread(ray.get, refs)
            results = [*direct_results, *remote_results]
        else:
            results = direct_results

        if len(results) != len(assignments):
            raise RuntimeError(
                "distributed rollout returned wrong number of chunks: "
                f"{len(results)} != {len(assignments)}",
            )

        for result in results:
            if result.error:
                raise RuntimeError(
                    "distributed rollout chunk failed "
                    f"(worker_id={result.worker_id}, chunk={result.chunk}): "
                    f"{result.error}",
                )
            if (
                request.policy_version is not None
                and result.policy_version != request.policy_version
            ):
                raise RuntimeError(
                    "distributed rollout policy_version mismatch "
                    f"(worker_id={result.worker_id}, "
                    f"expected={request.policy_version}, "
                    f"actual={result.policy_version})",
                )

        chunk_outputs: list[PipelineChunkResult] = []
        for result in results:
            if result.output is None:
                raise RuntimeError(
                    f"distributed rollout chunk returned no output: {result}",
                )
            chunk_outputs.append(result.output)

        sample_specs = self.id_factory.build_sample_specs(request)
        return gather_pipeline_chunks(
            self.gatherer,
            request,
            sample_specs,
            chunk_outputs,
        )


__all__ = ["DistributedRolloutExecutor"]
