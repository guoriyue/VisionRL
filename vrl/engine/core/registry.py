"""Family pipeline executor registry."""

from __future__ import annotations

from dataclasses import dataclass

from vrl.engine.core.protocols import FamilyPipelineExecutor


@dataclass(frozen=True, slots=True)
class ExecutorKey:
    family: str
    task: str


class FamilyPipelineRegistry:
    """Resolve family/task pairs to pipeline executors."""

    def __init__(self) -> None:
        self._executors: dict[ExecutorKey, FamilyPipelineExecutor] = {}

    def register(self, executor: FamilyPipelineExecutor) -> None:
        key = ExecutorKey(executor.family, executor.task)
        if key in self._executors:
            raise ValueError(
                f"Generation executor already registered for "
                f"family={key.family!r}, task={key.task!r}"
            )
        self._executors[key] = executor

    def resolve(self, family: str, task: str) -> FamilyPipelineExecutor:
        key = ExecutorKey(family, task)
        try:
            return self._executors[key]
        except KeyError as exc:
            registered = sorted(
                (executor_key.family, executor_key.task) for executor_key in self._executors
            )
            raise NotImplementedError(
                f"No generation executor registered for family={family!r}, "
                f"task={task!r}; registered={registered}"
            ) from exc
