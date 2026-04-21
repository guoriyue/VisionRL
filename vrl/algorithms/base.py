"""Algorithm ABC — advantage computation and policy loss."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from vrl.algorithms.types import TrainStepMetrics
from vrl.rollouts.evaluators.types import SignalBatch


class Algorithm(ABC):
    """Base class for RL algorithms (GRPO, REINFORCE, etc.).

    CEA pipeline interface:
    - compute_advantages_from_tensors(rewards, group_ids)
    - compute_signal_loss(signals, advantages, old_log_probs)
    """

    @abstractmethod
    def compute_advantages_from_tensors(
        self,
        rewards: Any,        # [B] tensor
        group_ids: Any,      # [B] tensor — prompt group assignment
    ) -> Any:                # [B] tensor of advantages
        """Compute per-sample advantages from reward tensors."""

    @abstractmethod
    def compute_signal_loss(
        self,
        signals: SignalBatch,
        advantages: Any,          # [B] or [B, T] advantages
        old_log_probs: Any,       # [B] old log-probs from collection
    ) -> tuple[Any, TrainStepMetrics]:
        """Compute loss from evaluator signals. Returns (loss, metrics)."""
