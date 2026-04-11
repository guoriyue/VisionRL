"""Remote reward function — scores rollouts via an async HTTP endpoint."""

from __future__ import annotations

from typing import Any

import aiohttp

from vrl.algorithms.types import Rollout
from vrl.rewards.base import RewardFunction


class RemoteRewardFunction(RewardFunction):
    """Calls a remote reward service over HTTP.

    The service receives a JSON payload with the rollout's prompt, seed,
    and output metadata.  It returns ``{"score": <float>}`` (single) or
    ``{"scores": [<float>, ...]}`` (batch).
    """

    def __init__(self, url: str, timeout: float = 30.0) -> None:
        self.url = url
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    def _serialize_rollout(self, rollout: Rollout) -> dict[str, Any]:
        return {
            "prompt": rollout.trajectory.prompt,
            "seed": rollout.trajectory.seed,
            "reward": rollout.reward,
            "metadata": rollout.metadata,
        }

    async def score(self, rollout: Rollout) -> float:
        payload = self._serialize_rollout(rollout)
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(self.url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return float(data["score"])

    async def score_batch(self, rollouts: list[Rollout]) -> list[float]:
        payload = {"rollouts": [self._serialize_rollout(r) for r in rollouts]}
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(self.url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return [float(s) for s in data["scores"]]
