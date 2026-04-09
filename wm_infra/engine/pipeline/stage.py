"""Stage runner protocol and concrete encode/dynamics stages.

Each stage implements a ``forward(batch, **kwargs)`` protocol that
the worker calls during execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class StageRunner(Protocol):
    """Protocol for a runnable engine stage."""

    name: str

    def forward(self, batch: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Run the stage on a batched tensor and return the result."""
        ...


@dataclass(frozen=True, slots=True)
class StageSpec:
    """Declarative specification for a pipeline stage."""

    name: str
    stream_id: int = 0
    device: str = "cpu"
    priority: int = 0


class EncodeStage:
    """Encode (tokenize) observations into latent space.

    For testing, uses a simple linear projection. In production this would
    wrap the video tokenizer encoder.
    """

    name: str = "encode"

    def __init__(
        self,
        latent_dim: int = 16,
        observation_dim: int | None = None,
    ) -> None:
        self.latent_dim = latent_dim
        self.observation_dim = observation_dim or latent_dim

    def forward(self, batch: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Encode observations → latent. Identity/passthrough for shape compatibility."""
        if batch.shape[-1] == self.latent_dim:
            return batch
        # Simple truncation/padding to latent_dim
        if batch.shape[-1] > self.latent_dim:
            return batch[..., : self.latent_dim]
        pad = torch.zeros(
            *batch.shape[:-1], self.latent_dim - batch.shape[-1],
            device=batch.device, dtype=batch.dtype,
        )
        return torch.cat([batch, pad], dim=-1)


class DynamicsStage:
    """One step of the dynamics model (world model forward).

    For testing, applies a simple additive offset. In production this wraps
    the block-causal transformer.
    """

    name: str = "dynamics"

    def __init__(self, step_delta: float = 0.01) -> None:
        self.step_delta = step_delta

    def forward(self, batch: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Dynamics step: latent_t → latent_{t+1}."""
        return batch + self.step_delta
