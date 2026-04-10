"""Worker, queues, and stage runners for engine execution.

Merges:
- workers/queues.py (AsyncQueue, RequestQueue, ResultQueue)
- pipeline/stage.py (StageRunner, StageSpec, EncodeStage, DynamicsStage)
- workers/worker.py (Worker)
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

import torch

from wm_infra.engine.mem_cache.paged_pool import PagedLatentPool
from wm_infra.engine.model_executor.task_graph import TaskGraph
from wm_infra.engine.types import EntityRequest, StepResult

# ---------------------------------------------------------------------------
# Async queues (from workers/queues.py)
# ---------------------------------------------------------------------------

T = TypeVar("T")


class AsyncQueue(Generic[T]):
    """Simple typed wrapper around ``asyncio.Queue``."""

    def __init__(self, maxsize: int = 0) -> None:
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=maxsize)

    async def put(self, item: T) -> None:
        await self._queue.put(item)

    def put_nowait(self, item: T) -> None:
        self._queue.put_nowait(item)

    async def get(self) -> T:
        return await self._queue.get()

    def get_nowait(self) -> T:
        return self._queue.get_nowait()

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()

    def drain(self) -> list[T]:
        """Non-blocking drain: pop all currently available items."""
        items: list[T] = []
        while not self._queue.empty():
            try:
                items.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return items


class RequestQueue(AsyncQueue[EntityRequest]):
    """Queue for incoming entity requests."""

    pass


class ResultQueue(AsyncQueue[StepResult]):
    """Queue for outgoing step results."""

    pass


# ---------------------------------------------------------------------------
# Stage runners (from pipeline/stage.py)
# ---------------------------------------------------------------------------


@runtime_checkable
class StageRunner(Protocol):
    """Protocol for a runnable engine stage."""

    name: str

    def forward(self, batch: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Run the stage on a batched tensor and return the result."""
        ...


@dataclass(frozen=True, slots=True)
class StageSpec:
    """Declarative specification for a pipeline stage."""

    name: str
    stream_id: int = 0
    device: str = "cpu"
    priority: int = 0


class EncodeStage:
    """Encode (tokenize) observations into latent space.

    For testing, uses a simple linear projection. In production this would
    wrap the video tokenizer encoder.
    """

    name: str = "encode"

    def __init__(
        self,
        latent_dim: int = 16,
        observation_dim: int | None = None,
    ) -> None:
        self.latent_dim = latent_dim
        self.observation_dim = observation_dim or latent_dim

    def forward(self, batch: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Encode observations -> latent. Identity/passthrough for shape compatibility."""
        if batch.shape[-1] == self.latent_dim:
            return batch
        # Simple truncation/padding to latent_dim
        if batch.shape[-1] > self.latent_dim:
            return batch[..., : self.latent_dim]
        pad = torch.zeros(
            *batch.shape[:-1],
            self.latent_dim - batch.shape[-1],
            device=batch.device,
            dtype=batch.dtype,
        )
        return torch.cat([batch, pad], dim=-1)


class DynamicsStage:
    """One step of the dynamics model (world model forward).

    For testing, applies a simple additive offset. In production this wraps
    the block-causal transformer.
    """

    name: str = "dynamics"

    def __init__(self, step_delta: float = 0.01) -> None:
        self.step_delta = step_delta

    def forward(self, batch: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Dynamics step: latent_t -> latent_{t+1}."""
        return batch + self.step_delta


# ---------------------------------------------------------------------------
# Worker (from workers/worker.py)
# ---------------------------------------------------------------------------


class Worker:
    """Batched execution worker for engine stages.

    Parameters
    ----------
    pool : PagedLatentPool
        The shared latent pool for gather/scatter.
    device : str
        Torch device for computation.
    """

    def __init__(
        self,
        pool: PagedLatentPool,
        device: str = "cpu",
    ) -> None:
        self.pool = pool
        self.device = device
        self._stages: dict[str, StageRunner] = {}

    def register_stage(self, stage: StageRunner) -> None:
        """Register a named stage for use in execution."""
        self._stages[stage.name] = stage

    def get_stage(self, name: str) -> StageRunner | None:
        return self._stages.get(name)

    @property
    def stage_names(self) -> list[str]:
        return list(self._stages.keys())

    def execute_step(
        self,
        entity_ids: Sequence[str],
        stage_name: str,
        step_indices: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> list[StepResult]:
        """Execute a batched stage on the given entities.

        Steps: gather from pool -> stage forward -> scatter to pool -> results.
        """
        stage = self._stages.get(stage_name)
        if stage is None:
            raise KeyError(f"Stage {stage_name!r} not registered")

        if not entity_ids:
            return []

        # Gather
        batch = self.pool.gather_batch(entity_ids)

        # Forward
        output = stage.forward(batch, **kwargs)

        # Scatter
        self.pool.scatter_results(entity_ids, output)

        # Build results
        results: list[StepResult] = []
        indices = step_indices or [0] * len(entity_ids)
        for i, eid in enumerate(entity_ids):
            results.append(
                StepResult(
                    request_id=eid,
                    step_index=indices[i],
                    output_latent=output[i] if i < len(output) else None,
                )
            )
        return results

    def execute_step_with_task_graph(
        self,
        entity_ids: Sequence[str],
        stage_names: Sequence[str],
        step_indices: Sequence[int] | None = None,
        use_cuda_streams: bool = False,
    ) -> dict[str, list[StepResult]]:
        """Execute multiple stages via a task graph for overlap.

        Builds a linear DAG of stages (each depends on the previous),
        executes via ``TaskGraph``, and returns results keyed by stage name.
        """
        if not entity_ids:
            return {}

        graph = TaskGraph(use_cuda_streams=use_cuda_streams)

        # Gather node
        gathered_batch: dict[str, torch.Tensor] = {}

        def gather_fn() -> torch.Tensor:
            t = self.pool.gather_batch(entity_ids)
            gathered_batch["data"] = t
            return t

        graph.add_node("gather", gather_fn, stream_id=0)

        # Stage nodes
        prev_name = "gather"
        stage_outputs: dict[str, torch.Tensor] = {}

        for sname in stage_names:
            stage = self._stages.get(sname)
            if stage is None:
                raise KeyError(f"Stage {sname!r} not registered")

            def make_fn(_stage: StageRunner, _prev: str) -> Any:
                def fn() -> torch.Tensor:
                    inp = stage_outputs.get(_prev, gathered_batch.get("data"))
                    if inp is None:
                        raise RuntimeError(f"No input for stage {_stage.name}")
                    out = _stage.forward(inp)
                    stage_outputs[_stage.name] = out
                    return out

                return fn

            graph.add_node(sname, make_fn(stage, prev_name), stream_id=0)
            graph.add_edge(prev_name, sname)
            prev_name = sname

        # Scatter node
        last_stage = stage_names[-1]

        def scatter_fn() -> None:
            data = stage_outputs.get(last_stage)
            if data is not None:
                self.pool.scatter_results(entity_ids, data)

        graph.add_node("scatter", scatter_fn, stream_id=0)
        graph.add_edge(last_stage, "scatter")

        # Execute
        graph.execute()

        # Build results per stage
        indices = step_indices or [0] * len(entity_ids)
        all_results: dict[str, list[StepResult]] = {}
        for sname in stage_names:
            output = stage_outputs.get(sname)
            results: list[StepResult] = []
            for i, eid in enumerate(entity_ids):
                results.append(
                    StepResult(
                        request_id=eid,
                        step_index=indices[i],
                        output_latent=output[i]
                        if output is not None and i < len(output)
                        else None,
                    )
                )
            all_results[sname] = results

        return all_results
