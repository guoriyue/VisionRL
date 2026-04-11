"""Pluggable protocols: BatchPlanner, IterationController, CacheManager, FeedbackMailbox."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from vrl.engine.types import RequestOutput, SchedulerRequest

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class BatchPlanner(Protocol):
    """Select requests and build batch payload."""

    def select_requests(
        self,
        waiting: list[SchedulerRequest],
        running: list[SchedulerRequest],
    ) -> list[SchedulerRequest]: ...

    def build_batch(self, requests: list[SchedulerRequest]) -> Any: ...


@runtime_checkable
class IterationController(Protocol):
    """Per-request completion logic after each pass."""

    def update_request(self, request: SchedulerRequest, output: RequestOutput) -> None: ...
    def is_finished(self, request: SchedulerRequest, output: RequestOutput) -> bool: ...


@runtime_checkable
class CacheManager(Protocol):
    """Output cache for redundant execution skipping."""

    def get(self, request: SchedulerRequest) -> RequestOutput | None: ...
    def put(self, request: SchedulerRequest, output: RequestOutput) -> None: ...
    def clear(self) -> None: ...


@runtime_checkable
class FeedbackMailbox(Protocol):
    """Non-blocking feedback mailbox keyed by request ID."""

    def has(self, request_id: str) -> bool: ...
    def pop(self, request_id: str) -> Any | None: ...


# ---------------------------------------------------------------------------
# Default implementations
# ---------------------------------------------------------------------------


class ContinuousBatchPlanner:
    """FIFO batch planner: running requests first, then waiting up to budget."""

    def __init__(self, max_batch_size: int = 32) -> None:
        self.max_batch_size = max_batch_size

    def select_requests(
        self,
        waiting: list[SchedulerRequest],
        running: list[SchedulerRequest],
    ) -> list[SchedulerRequest]:
        selected: list[SchedulerRequest] = list(running)
        budget = self.max_batch_size - len(selected)
        for req in waiting:
            if budget <= 0:
                break
            selected.append(req)
            budget -= 1
        return selected

    def build_batch(self, requests: list[SchedulerRequest]) -> Any:
        return [r.data for r in requests]


class VideoDiffusionIterationController:
    """Multi-step controller: error propagation + terminal detection."""

    def update_request(self, request: SchedulerRequest, output: RequestOutput) -> None:
        if output.finish_reason == "error" and request.error is None:
            request.error = RuntimeError("Model execution failed")

    def is_finished(self, request: SchedulerRequest, output: RequestOutput) -> bool:
        if output.finished:
            from vrl.engine.model_executor.execution_state import VideoExecutionState

            state = request.data
            if isinstance(state, VideoExecutionState):
                request.data = state.stage_results
            return True
        return False
