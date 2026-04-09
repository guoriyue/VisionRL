"""Worker — owns the pool and stages, executes batched gather/forward/scatter.

The worker is the leaf executor in the engine hierarchy: the engine loop
calls ``execute_step()`` and the worker handles gather from pool →
stage forward → scatter to pool.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from wm_infra.engine._types import StepResult
from wm_infra.engine.pipeline.stage import StageRunner
from wm_infra.engine.pipeline.task_graph import TaskGraph
from wm_infra.engine.state.paged_pool import PagedLatentPool


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

        Steps: gather from pool → stage forward → scatter to pool → results.
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
            results.append(StepResult(
                request_id=eid,
                step_index=indices[i],
                output_latent=output[i] if i < len(output) else None,
            ))
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
                results.append(StepResult(
                    request_id=eid,
                    step_index=indices[i],
                    output_latent=output[i] if output is not None and i < len(output) else None,
                ))
            all_results[sname] = results

        return all_results
