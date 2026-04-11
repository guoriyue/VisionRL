"""Trainer configuration and training state."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TrainerConfig:
    """Configuration for the training loop."""

    group_size: int = 4
    lr: float = 1e-5
    max_grad_norm: float = 1.0
    epochs_per_step: int = 1


@dataclass(slots=True)
class TrainState:
    """Mutable training state tracked across steps."""

    step: int = 0
    total_reward: float = 0.0
    total_loss: float = 0.0
