"""Online RL trainer — collect → reward → advantage → loss → update → sync."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from vrl.algorithms.base import Algorithm
from vrl.algorithms.types import RolloutBatch, RolloutGroup, TrainStepMetrics
from vrl.rewards.base import RewardFunction
from vrl.trainers.base import Trainer
from vrl.trainers.types import TrainerConfig, TrainState
from vrl.trainers.weight_sync import WeightSyncer


@runtime_checkable
class RolloutSource(Protocol):
    """Protocol for rollout collectors (implemented in ``rollout/`` later)."""

    async def collect(self, prompts: list[str], **kwargs: Any) -> list[RolloutGroup]:
        ...


class OnlineTrainer(Trainer):
    """Orchestrates the online RL loop.

    1. Collect rollouts via ``rollout_source``
    2. Score with ``reward_fn``
    3. Compute advantages via ``algorithm``
    4. Compute loss via ``algorithm``
    5. (Optional) sync weights via ``weight_syncer``
    """

    def __init__(
        self,
        algorithm: Algorithm,
        reward_fn: RewardFunction,
        rollout_source: RolloutSource,
        weight_syncer: WeightSyncer | None = None,
        config: TrainerConfig | None = None,
        policy: Any = None,
        ref_policy: Any = None,
        prompts: list[str] | None = None,
    ) -> None:
        self.algorithm = algorithm
        self.reward_fn = reward_fn
        self.rollout_source = rollout_source
        self.weight_syncer = weight_syncer
        self.config = config or TrainerConfig()
        self.policy = policy
        self.ref_policy = ref_policy
        self.prompts = prompts or []
        self.state = TrainState()

    async def step(self) -> TrainStepMetrics:
        # 1. Collect rollouts
        groups = await self.rollout_source.collect(self.prompts)

        # 2. Score each rollout
        for group in groups:
            scores = await self.reward_fn.score_batch(group.rollouts)
            for rollout, score in zip(group.rollouts, scores):
                rollout.reward = score

        # 3. Compute advantages
        for group in groups:
            group.advantages = self.algorithm.compute_advantages(group)

        # 4. Compute loss
        batch = RolloutBatch(groups=groups)
        loss, metrics = self.algorithm.compute_loss(
            batch, self.policy, self.ref_policy
        )

        # 5. Update state
        self.state.step += 1
        self.state.total_reward += metrics.reward_mean
        self.state.total_loss += metrics.loss

        # 6. Sync weights (if configured)
        if self.weight_syncer is not None and self.policy is not None:
            state_dict = (
                self.policy.state_dict()
                if hasattr(self.policy, "state_dict")
                else {}
            )
            await self.weight_syncer.push(state_dict)

        return metrics

    def state_dict(self) -> dict:
        return {
            "step": self.state.step,
            "total_reward": self.state.total_reward,
            "total_loss": self.state.total_loss,
        }

    def load_state_dict(self, state: dict) -> None:
        self.state.step = state.get("step", 0)
        self.state.total_reward = state.get("total_reward", 0.0)
        self.state.total_loss = state.get("total_loss", 0.0)
