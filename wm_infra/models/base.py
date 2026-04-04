"""Base interface for world models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class RolloutInput:
    """Input for a single rollout request."""

    latent_state: torch.Tensor  # [B, N, D] current latent state
    actions: torch.Tensor  # [B, T, A] sequence of future actions
    num_steps: int  # how many steps to predict


@dataclass
class RolloutOutput:
    """Output of a rollout prediction."""

    predicted_states: torch.Tensor  # [B, T, N, D] predicted latent states
    predicted_indices: torch.Tensor | None = None  # [B, T, N] discrete token indices


class WorldModel(ABC):
    """Abstract base class for world models.

    A world model predicts future latent states given current state + actions:
        z_{t+1} = f(z_t, a_t)
    """

    @abstractmethod
    def predict_next(
        self,
        latent_state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the next latent state given current state and action.

        Args:
            latent_state: [B, N, D] current latent state tokens
            action: [B, A] action vector

        Returns:
            next_state: [B, N, D] predicted next latent state
        """
        ...

    @abstractmethod
    def rollout(self, input: RolloutInput) -> RolloutOutput:
        """Predict multiple future states autoregressively.

        Args:
            input: RolloutInput with initial state and action sequence

        Returns:
            RolloutOutput with predicted states for each step
        """
        ...

    @abstractmethod
    def get_initial_state(self, observation: torch.Tensor) -> torch.Tensor:
        """Encode an observation into the latent state.

        Args:
            observation: raw observation (e.g., video frame)

        Returns:
            latent_state: [B, N, D] initial latent state
        """
        ...
