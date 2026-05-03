"""Shared reward scoring for rollout collectors."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from vrl.algorithms.types import Rollout, Trajectory


class RewardScorer:
    """Score decoded rollout outputs without knowing ExperienceBatch layout."""

    def __init__(self, reward_fn: Any | None) -> None:
        self.reward_fn = reward_fn

    async def score(
        self,
        outputs: Any,
        prompts: Sequence[str],
        metadata: Mapping[str, Any],
        device: Any,
    ) -> torch.Tensor:
        if self.reward_fn is None:
            return torch.zeros(_batch_size(outputs), device=device)

        rollouts = [
            Rollout(
                request=None,
                trajectory=Trajectory(
                    prompt=prompts[i],
                    seed=0,
                    steps=[],
                    output=outputs[i],
                ),
                metadata=dict(metadata),
            )
            for i in range(_batch_size(outputs))
        ]

        batch_fn = getattr(self.reward_fn, "score_batch", None)
        if batch_fn is not None:
            raw = batch_fn(rollouts)
            if inspect.isawaitable(raw):
                raw = await raw
        else:
            raw = []
            for rollout in rollouts:
                value = self.reward_fn.score(rollout)
                if inspect.isawaitable(value):
                    value = await value
                raw.append(value)

        return torch.tensor(
            [float(score) for score in raw],
            device=device,
            dtype=torch.float32,
        )


def _batch_size(outputs: Any) -> int:
    shape = getattr(outputs, "shape", None)
    if shape is not None:
        return int(shape[0])
    return len(outputs)


__all__ = ["RewardScorer"]
