"""Collector protocol — collects training experience from model rollouts."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from vrl.rollouts.types import ExperienceBatch


@runtime_checkable
class Collector(Protocol):
    """Collects training experience from model rollouts.

    A collector owns rollout semantics only:

      - prompt expansion / group sampling
      - calling the policy's sampling path
      - reward scoring
      - ``ExperienceBatch`` packing
      - prompt / reference / metadata forwarding

    Train-time replay forward used to live here as ``forward_step``; that
    ownership has moved to the policy (``model.replay_forward``). Evaluators
    call the policy directly. New collectors must NOT add training-replay
    math — keep this protocol to ``collect()`` only.
    """

    async def collect(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> ExperienceBatch:
        ...
