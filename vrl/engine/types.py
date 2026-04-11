"""Engine types: request lifecycle, scheduler I/O, model runner output."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Request lifecycle
# ---------------------------------------------------------------------------


class SchedulerStatus(Enum):
    """Request lifecycle state."""

    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass(slots=True)
class SchedulerRequest:
    """One request tracked by the scheduler. ``data`` is opaque to the scheduler."""

    request_id: str
    data: Any
    status: SchedulerStatus = SchedulerStatus.WAITING
    arrival_time: float = field(default_factory=time.monotonic)
    finish_time: float | None = None
    error: Exception | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SchedulerOutput:
    """Batch selected for this iteration."""

    requests: list[SchedulerRequest] = field(default_factory=list)
    batch_data: Any = None
    step_id: int = 0


@dataclass(slots=True)
class RequestOutput:
    """Single-request result from one execution pass."""

    request_id: str
    data: Any = None
    finished: bool = False
    finish_reason: str | None = None
    extra: dict[str, Any] | None = None


@dataclass(slots=True)
class ModelRunnerOutput:
    """Aggregated results from one execution pass."""

    outputs: dict[str, RequestOutput] = field(default_factory=dict)
    req_ids: list[str] = field(default_factory=list)
    req_id_to_index: dict[str, int] = field(default_factory=dict)
