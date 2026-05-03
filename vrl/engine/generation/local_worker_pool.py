"""Single-worker local chunk execution helper for rollout tests/debugging."""

from __future__ import annotations

from dataclasses import dataclass

from vrl.engine.generation.gather import (
    gather_pipeline_chunks,
    require_chunked_executor,
)
from vrl.engine.generation.registry import FamilyPipelineRegistry
from vrl.engine.generation.types import GenerationRequest, OutputBatch
from vrl.engine.generation.worker import GenerationIdFactory
from vrl.executors.base import PipelineChunkResult
from vrl.executors.microbatching import (
    MicroBatchPlan,
    plan_prompt_group_microbatches,
)


@dataclass(frozen=True, slots=True)
class LocalWorkerSpec:
    """Resource identity for one local rollout worker."""

    worker_id: str
    device: str


class LocalRolloutWorker:
    """Execute one chunk through a family executor in this process."""

    def __init__(
        self,
        registry: FamilyPipelineRegistry,
        spec: LocalWorkerSpec,
    ) -> None:
        self.registry = registry
        self.spec = spec

    def execute_chunk(
        self,
        request: GenerationRequest,
        chunk: MicroBatchPlan,
    ) -> PipelineChunkResult:
        executor = self.registry.resolve(request.family, request.task)
        chunked = require_chunked_executor(executor)
        return chunked.forward_chunk(request, chunk)


class LocalRolloutWorkerPool:
    """Split one request into chunks on one local worker.

    Local execution is intentionally not a multi-GPU backend. Multi-worker,
    multi-GPU, and multi-node rollout placement belongs to the Ray backend.
    """

    def __init__(
        self,
        registry: FamilyPipelineRegistry,
        worker_specs: list[LocalWorkerSpec],
        *,
        id_factory: GenerationIdFactory | None = None,
    ) -> None:
        if not worker_specs:
            raise ValueError("LocalRolloutWorkerPool requires at least one worker")
        if len(worker_specs) != 1:
            raise ValueError(
                "LocalRolloutWorkerPool is a single-worker debug helper; "
                "use the Ray backend for multi-worker rollout",
            )
        self.registry = registry
        self.worker_specs = list(worker_specs)
        self.id_factory = id_factory or GenerationIdFactory()
        self.workers = [LocalRolloutWorker(registry, spec) for spec in self.worker_specs]

    async def execute(self, request: GenerationRequest) -> OutputBatch:
        """Execute a full request by sequentially running its chunks locally."""

        executor = self.registry.resolve(request.family, request.task)
        sample_specs = self.id_factory.build_sample_specs(request)
        max_samples = int(
            request.sampling.get("sample_batch_size", request.samples_per_prompt),
        )
        plan = plan_prompt_group_microbatches(
            request.prompts,
            samples_per_prompt=request.samples_per_prompt,
            max_samples_per_microbatch=max(1, max_samples),
        )

        chunks: list[PipelineChunkResult] = []
        worker = self.workers[0]
        for micro_batch in plan.micro_batches:
            chunks.append(worker.execute_chunk(request, micro_batch))

        return gather_pipeline_chunks(executor, request, sample_specs, chunks)


__all__ = [
    "LocalRolloutWorker",
    "LocalRolloutWorkerPool",
    "LocalWorkerSpec",
]
