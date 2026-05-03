"""Packer protocol for turning engine outputs into trainer experiences."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from vrl.engine.generation import OutputBatch
from vrl.rollouts.experience import ExperienceBatch


@dataclass(slots=True)
class RolloutPackContext:
    """Non-engine metadata required by rollout packers."""

    metadata: dict[str, Any]
    device: Any | None = None
    kl_reward: float = 0.0
    rescale_to_unit: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class RolloutPacker(Protocol):
    """Translate generation outputs into RL training batches."""

    def reward_outputs(
        self,
        output: OutputBatch,
        context: RolloutPackContext,
    ) -> Any: ...

    def reward_prompts(
        self,
        output: OutputBatch,
        context: RolloutPackContext,
    ) -> list[str]: ...

    async def pack(
        self,
        output: OutputBatch,
        rewards_raw: Any,
        context: RolloutPackContext,
    ) -> ExperienceBatch: ...


__all__ = ["RolloutPackContext", "RolloutPacker"]
