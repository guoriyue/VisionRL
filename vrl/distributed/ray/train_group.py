"""Ray train actor group scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RayTrainRankSpec:
    """Rank placement metadata for a future Ray-managed FSDP group."""

    rank: int
    local_rank: int
    world_size: int
    node_id: str
    gpu_ids: tuple[int, ...] = ()


class RayTrainGroup:
    """Thin owner for train actors.

    P0 supports a single train actor. Future FSDP support should expand this
    into N rank actors that initialize torch.distributed inside each actor.
    """

    def __init__(self, actors: list[Any]) -> None:
        if not actors:
            raise ValueError("RayTrainGroup requires at least one actor")
        self.actors = list(actors)

    def primary(self) -> Any:
        return self.actors[0]

    def train_step(self, prompt_batch: list[Any]) -> Any:
        actor = self.primary()
        train_step = actor.train_step
        remote = getattr(train_step, "remote", None)
        if callable(remote):
            return remote(prompt_batch)
        return train_step(prompt_batch)


__all__ = [
    "RayTrainGroup",
    "RayTrainRankSpec",
]
