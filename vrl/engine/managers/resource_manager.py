"""Resource manager implementations."""

from __future__ import annotations

from vrl.engine.types import SchedulerRequest


class SimpleResourceManager:
    """Count-based resource manager: admits up to max_count concurrent requests."""

    def __init__(self, max_count: int = 32) -> None:
        self.max_count = max_count
        self._count = 0

    def can_allocate(self, request: SchedulerRequest) -> bool:
        return self._count < self.max_count

    def allocate(self, request: SchedulerRequest) -> None:
        self._count += 1

    def free(self, request: SchedulerRequest) -> None:
        self._count = max(0, self._count - 1)
