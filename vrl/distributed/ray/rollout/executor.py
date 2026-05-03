"""Distributed rollout executor that gathers Ray chunk results."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

from vrl.distributed.ray.dependencies import require_ray
from vrl.distributed.ray.rollout.planner import DistributedExecutionPlanner
from vrl.distributed.ray.rollout.types import RayChunkResult, RayWorkerHandle
from vrl.engine.generation.gather import ChunkGatherer, gather_pipeline_chunks
from vrl.engine.generation.protocols import PipelineChunkResult
from vrl.engine.generation.types import GenerationRequest, OutputBatch
from vrl.engine.generation.worker import GenerationIdFactory


class DistributedRolloutExecutor:
    """Execute one GenerationRequest across rollout workers."""

    def __init__(
        self,
        planner: DistributedExecutionPlanner,
        workers: list[RayWorkerHandle],
        gatherer: ChunkGatherer,
        *,
        id_factory: GenerationIdFactory | None = None,
        max_inflight_chunks_per_worker: int = 1,
    ) -> None:
        if not workers:
            raise ValueError("DistributedRolloutExecutor requires at least one worker")
        if max_inflight_chunks_per_worker < 1:
            raise ValueError("max_inflight_chunks_per_worker must be >= 1")
        self.planner = planner
        self.workers = list(workers)
        self.gatherer = gatherer
        self.id_factory = id_factory or GenerationIdFactory()
        self.max_inflight_chunks_per_worker = int(max_inflight_chunks_per_worker)

    async def execute(self, request: GenerationRequest) -> OutputBatch:
        assignments = self.planner.plan(request, self.workers)
        worker_by_id = {worker.worker_id: worker for worker in self.workers}
        remote_jobs: list[tuple[int, Any, RayWorkerHandle, Any]] = []
        result_pairs: list[tuple[int, RayChunkResult]] = []

        for job_index, assignment in enumerate(assignments):
            worker = worker_by_id[assignment.worker_id]
            actor = worker.actor
            if actor is None:
                raise RuntimeError(f"worker {worker.worker_id!r} has no actor")
            execute_chunk = actor.execute_chunk
            remote = getattr(execute_chunk, "remote", None)
            if callable(remote):
                remote_jobs.append((job_index, remote, worker, assignment.chunk))
            else:
                result_pairs.append((job_index, execute_chunk(request, assignment.chunk)))

        if remote_jobs:
            result_pairs.extend(await self._run_remote_jobs(request, remote_jobs))

        results = [result for _, result in sorted(result_pairs, key=lambda pair: pair[0])]

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

    async def _run_remote_jobs(
        self,
        request: GenerationRequest,
        jobs: list[tuple[int, Any, RayWorkerHandle, Any]],
    ) -> list[tuple[int, RayChunkResult]]:
        ray = require_ray()
        pending = deque(jobs)
        inflight_by_worker = {worker.worker_id: 0 for _, _, worker, _ in jobs}
        ref_to_job: dict[Any, tuple[int, str]] = {}
        result_pairs: list[tuple[int, RayChunkResult]] = []

        def _submit_ready() -> None:
            made_progress = True
            while pending and made_progress:
                made_progress = False
                for _ in range(len(pending)):
                    job_index, remote, worker, chunk = pending.popleft()
                    if inflight_by_worker[worker.worker_id] >= self.max_inflight_chunks_per_worker:
                        pending.append((job_index, remote, worker, chunk))
                        continue
                    ref = remote(request, chunk)
                    ref_to_job[ref] = (job_index, worker.worker_id)
                    inflight_by_worker[worker.worker_id] += 1
                    made_progress = True

        _submit_ready()
        while ref_to_job:
            ready, _ = await asyncio.to_thread(
                ray.wait,
                list(ref_to_job),
                num_returns=1,
            )
            job_index, worker_id = ref_to_job.pop(ready[0])
            inflight_by_worker[worker_id] -= 1
            result = await asyncio.to_thread(ray.get, ready[0])
            result_pairs.append((job_index, result))
            _submit_ready()

        return result_pairs


__all__ = ["DistributedRolloutExecutor"]
