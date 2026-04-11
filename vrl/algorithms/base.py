"""Algorithm ABC — advantage computation and policy loss."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from vrl.algorithms.types import Advantages, RolloutBatch, RolloutGroup, TrainStepMetrics


class Algorithm(ABC):
    """Base class for RL algorithms (GRPO, REINFORCE, etc.)."""

    @abstractmethod
    def compute_advantages(self, group: RolloutGroup) -> Advantages:
        """Compute per-rollout advantages for a single prompt group."""

    @abstractmethod
    def compute_loss(
        self,
        batch: RolloutBatch,
        policy: Any,
        ref_policy: Any = None,
    ) -> tuple[Any, TrainStepMetrics]:
        """Compute the policy gradient loss and metrics.

        ``policy`` / ``ref_policy`` are opaque — the algorithm uses
        pre-computed log-probs from ``TrajectoryStep.log_prob`` and only
        needs the models for optional KL computation.

        Returns (loss_tensor, metrics).
        """
