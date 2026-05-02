"""Generation engine adapters and runtime facade."""

from __future__ import annotations

from typing import Protocol

from vrl.engine.generation.types import (
    GenerationRequest,
    OutputBatch,
    WorkloadSignature,
)
from vrl.engine.generation.worker import GenerationWorker
from vrl.engine.loop import EngineLoop
from vrl.engine.protocols import BatchPlanner
from vrl.engine.types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
)


class RolloutBackend(Protocol):
    """Generation backend consumed by rollout collectors."""

    async def generate(self, request: GenerationRequest) -> OutputBatch:
        ...


class GenerationRuntime(RolloutBackend):
    """Typed facade over the existing EngineLoop."""

    def __init__(self, engine_loop: EngineLoop) -> None:
        self.engine_loop = engine_loop
        self._started = False

    async def start(self) -> None:
        if not self._started:
            await self.engine_loop.start()
            self._started = True

    async def generate(self, request: GenerationRequest) -> OutputBatch:
        await self.submit(request)
        return await self.wait(request.request_id)

    async def submit(self, request: GenerationRequest) -> str:
        await self.start()
        await self.engine_loop.add_request(request.request_id, request)
        return request.request_id

    async def wait(self, request_id: str) -> OutputBatch:
        result = await self.engine_loop.get_result(request_id)
        if not isinstance(result, OutputBatch):
            raise TypeError(
                f"GenerationRuntime expected OutputBatch, got {type(result)!r}"
            )
        return result

    async def shutdown(self) -> None:
        if self._started:
            await self.engine_loop.stop()
            self._started = False


class GenerationModelRunner:
    """Existing EngineLoop model_runner implementation for generation."""

    def __init__(
        self,
        worker: GenerationWorker,
        *,
        execute_in_thread: bool = True,
    ) -> None:
        self.worker = worker
        self.execute_in_thread = execute_in_thread

    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        requests = _requests_from_scheduler_output(scheduler_output)
        outputs = self.worker.execute(requests)
        request_outputs: dict[str, RequestOutput] = {}
        for scheduler_request in scheduler_output.requests:
            output = outputs.get(scheduler_request.request_id)
            if output is None:
                output = _missing_output(scheduler_request.request_id)
            request_outputs[scheduler_request.request_id] = RequestOutput(
                request_id=scheduler_request.request_id,
                data=output,
                finished=True,
                finish_reason="completed",
            )
        req_ids = [request.request_id for request in scheduler_output.requests]
        return ModelRunnerOutput(
            outputs=request_outputs,
            req_ids=req_ids,
            req_id_to_index={request_id: idx for idx, request_id in enumerate(req_ids)},
        )


class GenerationBatchPlanner(BatchPlanner):
    """Select same-signature generation requests for one engine tick."""

    def __init__(self, max_batch_size: int = 1) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        self.max_batch_size = max_batch_size

    def select_requests(
        self,
        waiting: list[SchedulerRequest],
        running: list[SchedulerRequest],
    ) -> list[SchedulerRequest]:
        selected: list[SchedulerRequest] = list(running)
        budget = self.max_batch_size - len(selected)
        if budget <= 0:
            return selected

        signature = _signature_for(selected[0]) if selected else None
        for request in waiting:
            if budget <= 0:
                break
            request_signature = _signature_for(request)
            if signature is None:
                signature = request_signature
            elif request_signature != signature:
                continue
            selected.append(request)
            budget -= 1
        return selected

    def build_batch(self, requests: list[SchedulerRequest]) -> list[GenerationRequest]:
        batch: list[GenerationRequest] = []
        for request in requests:
            if not isinstance(request.data, GenerationRequest):
                raise TypeError(
                    "GenerationBatchPlanner expects SchedulerRequest.data to be "
                    f"GenerationRequest, got {type(request.data)!r}"
                )
            batch.append(request.data)
        return batch


def _requests_from_scheduler_output(
    scheduler_output: SchedulerOutput,
) -> list[GenerationRequest]:
    batch_data = scheduler_output.batch_data
    if batch_data is None:
        batch_data = [request.data for request in scheduler_output.requests]
    if isinstance(batch_data, GenerationRequest):
        batch_data = [batch_data]
    requests = list(batch_data)
    for request in requests:
        if not isinstance(request, GenerationRequest):
            raise TypeError(
                "GenerationModelRunner expects SchedulerRequest.data to be "
                f"GenerationRequest, got {type(request)!r}"
            )
    return requests


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


def _signature_for(request: SchedulerRequest) -> WorkloadSignature:
    data = request.data
    if not isinstance(data, GenerationRequest):
        raise TypeError(
            "GenerationBatchPlanner expects SchedulerRequest.data to be "
            f"GenerationRequest, got {type(data)!r}"
        )
    return WorkloadSignature.from_request(data)
