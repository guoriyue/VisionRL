"""Batch planner implementations."""

from __future__ import annotations

from typing import Any

from vrl.engine.protocols import BatchPlanner
from vrl.engine.scheduler_types import SchedulerRequest


class ContinuousBatchPlanner(BatchPlanner):
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
