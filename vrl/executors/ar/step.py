"""Step-level contracts for scheduled AR executors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ARStepResult:
    """One scheduled AR token step.

    ``sequence_ids`` and every tensor-like value must follow the same order as
    the input ``ActiveSequence`` list. ``replay_extras`` is reserved for
    per-step tensors needed by training replay; for NextStep-1 it must include
    ``saved_noise``.
    """

    sequence_ids: list[str]
    positions: list[int]
    token: Any
    log_prob: Any
    replay_extras: dict[str, Any] = field(default_factory=dict)
