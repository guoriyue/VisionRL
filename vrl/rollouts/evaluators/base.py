"""Evaluator protocol — extract training signals from model forward results."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from vrl.rollouts.evaluators.types import SignalBatch, SignalRequest
from vrl.rollouts.experience import ExperienceBatch


@runtime_checkable
class Evaluator(Protocol):
    """Extract training signals from model forward results.

    Uses ``model.replay_forward`` for the train-time forward pass and
    extracts distribution-family-specific signals (log_prob, KL, etc.).

    Replay ownership lives on the policy. Evaluators must not route train-time
    replay through collectors.
    """

    def evaluate(
        self,
        model: Any,
        batch: ExperienceBatch,
        timestep_idx: int,
        ref_model: Any | None = None,
        signal_request: SignalRequest | None = None,
    ) -> SignalBatch:
        """Run model.replay_forward() -> extract log_prob, KL, etc."""
        ...
