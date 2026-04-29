"""Evaluator protocol — extract training signals from model forward results."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from vrl.rollouts.collectors.base import Collector
from vrl.rollouts.evaluators.types import SignalBatch, SignalRequest
from vrl.rollouts.types import ExperienceBatch


@runtime_checkable
class Evaluator(Protocol):
    """Extract training signals from model forward results.

    Uses ``model.replay_forward`` for the train-time forward pass and
    extracts distribution-family-specific signals (log_prob, KL, etc.).

    The ``collector`` parameter is retained for trainer-interface
    compatibility but evaluators do not use it — replay ownership lives
    on the policy.
    """

    def evaluate(
        self,
        collector: Collector,
        model: Any,
        batch: ExperienceBatch,
        timestep_idx: int,
        ref_model: Any | None = None,
        signal_request: SignalRequest | None = None,
    ) -> SignalBatch:
        """Run model.replay_forward() -> extract log_prob, KL, etc."""
        ...
