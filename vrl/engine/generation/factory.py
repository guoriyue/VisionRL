"""Factory helpers for local collector-facing generation runtimes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.engine.generation.runtime import RolloutBackend


def build_local_generation_runtime(
    *,
    model: Any,
    executor_cls: type,
    cfg: Any,
    executor_kwargs: Mapping[str, Any] | None = None,
) -> RolloutBackend:
    """Build the local in-process GenerationRuntime for one family executor."""
    if model is None:
        raise RuntimeError(
            "local generation runtime requires a live model; inject a runtime "
            "for distributed rollout",
        )

    from vrl.engine.generation.registry import FamilyPipelineRegistry
    from vrl.engine.generation.runtime import GenerationRuntime
    from vrl.engine.generation.worker import GenerationWorker

    registry = FamilyPipelineRegistry()
    registry.register(executor_cls(model, **dict(executor_kwargs or {})))
    worker = GenerationWorker(registry)
    return GenerationRuntime(worker, execute_in_thread=False)


__all__ = [
    "build_local_generation_runtime",
]
