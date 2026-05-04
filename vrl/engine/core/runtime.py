"""Generation runtime facade."""

from __future__ import annotations

import asyncio
from typing import Protocol

from vrl.engine.core.types import GenerationRequest, OutputBatch
from vrl.engine.core.worker import GenerationWorker


class RolloutBackend(Protocol):
    """Generation backend consumed by rollout collectors."""

    async def generate(self, request: GenerationRequest) -> OutputBatch: ...


class GenerationRuntime(RolloutBackend):
    """In-process generation runtime used by rollout collectors."""

    def __init__(
        self,
        worker: GenerationWorker,
        *,
        execute_in_thread: bool = False,
    ) -> None:
        self.worker = worker
        self.execute_in_thread = execute_in_thread

    async def generate(self, request: GenerationRequest) -> OutputBatch:
        outputs = await self.generate_many([request])
        return outputs.get(request.request_id) or _missing_output(request.request_id)

    async def generate_many(
        self,
        requests: list[GenerationRequest],
    ) -> dict[str, OutputBatch]:
        if self.execute_in_thread:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.worker.execute, requests)
        return self.worker.execute(requests)

    async def shutdown(self) -> None:
        return None


def _missing_output(request_id: str) -> OutputBatch:
    return OutputBatch(
        request_id=request_id,
        family="unknown",
        task="unknown",
        prompts=[],
        sample_specs=[],
        output=None,
        error="Generation worker did not return an output for this request",
    )
