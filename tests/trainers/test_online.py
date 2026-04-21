"""Tests for vrl.trainers.online (OnlineTrainer CEA pipeline)."""

from __future__ import annotations

import pytest


class TestOnlineTrainerCeaRegressions:
    def _make_cea_trainer(self, rewards: list[float]):
        import torch
        import torch.nn as nn

        from vrl.algorithms.types import TrainStepMetrics
        from vrl.rollouts.evaluators.types import SignalBatch
        from vrl.rollouts.types import ExperienceBatch
        from vrl.trainers.online import OnlineTrainer
        from vrl.trainers.types import TrainerConfig

        class _Algorithm:
            class _Config:
                global_std = False
                eps = 1e-8
                adv_clip_max = 5.0

            config = _Config()

            def compute_advantages_from_tensors(self, rewards, group_ids):
                advantages = torch.zeros_like(rewards)
                for gid in torch.unique(group_ids):
                    mask = group_ids == gid
                    gr = rewards[mask]
                    if gr.numel() <= 1:
                        continue
                    mean = gr.mean()
                    std = gr.std().clamp(min=1e-8)
                    advantages[mask] = (gr - mean) / std
                return advantages

            def compute_signal_loss(self, signals, advantages, old_log_probs):
                loss = signals.log_prob.mean()
                metrics = TrainStepMetrics(
                    loss=loss.item(),
                    policy_loss=loss.item(),
                    approx_kl=float(old_log_probs.mean().item()),
                )
                return loss, metrics

        class _Collector:
            def __init__(self, reward_values: list[float]) -> None:
                self._reward_values = reward_values
                self._cursor = 0

            async def collect(self, prompts, **kwargs):
                group_size = int(kwargs.get("group_size", 1))
                rewards = []
                for _ in range(group_size):
                    rewards.append(self._reward_values[self._cursor])
                    self._cursor += 1
                return ExperienceBatch(
                    observations=torch.zeros(group_size, 2, 1),
                    actions=torch.zeros(group_size, 2, 1),
                    rewards=torch.tensor(rewards, dtype=torch.float32),
                    dones=torch.ones(group_size, dtype=torch.bool),
                    group_ids=torch.zeros(group_size, dtype=torch.long),
                    extras={
                        "log_probs": torch.tensor(
                            [[0.0, 1.0]] * group_size, dtype=torch.float32,
                        ),
                    },
                    prompts=list(prompts) * group_size,
                )

        class _Evaluator:
            def evaluate(
                self,
                collector,
                model,
                batch,
                timestep_idx,
                ref_model=None,
                signal_request=None,
            ):
                batch_size = batch.rewards.shape[0]
                log_prob = model.weight.view(1).expand(batch_size)
                return SignalBatch(log_prob=log_prob)

        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)

        trainer = OnlineTrainer(
            algorithm=_Algorithm(),
            collector=_Collector(rewards),
            evaluator=_Evaluator(),
            model=model,
            config=TrainerConfig(
                lr=0.01,
                group_size=2,
                mixed_precision="no",
            ),
            device="cpu",
        )
        return trainer

    def test_cea_step_advantages_independent_across_steps(self) -> None:
        """Second-step advantages should be normalized against the current group only,
        with no state leaking from previous steps."""
        import asyncio

        trainer = self._make_cea_trainer([0.0, 0.0, 0.0, 1.0])

        asyncio.run(trainer.step(["prompt-a"]))
        second_step = asyncio.run(trainer.step(["prompt-a"]))

        # Advantages are computed purely from current group — no stale history
        assert second_step.advantage_mean == pytest.approx(0.0, abs=1e-3)

    def test_cea_metrics_propagate_approx_kl(self) -> None:
        """CEA aggregation should not silently drop approx_kl."""
        import asyncio

        trainer = self._make_cea_trainer([0.0, 1.0])
        metrics = asyncio.run(trainer.step(["prompt-a"]))

        assert metrics.approx_kl == pytest.approx(0.5)

    def test_cea_step_forwards_prompt_example_kwargs(self) -> None:
        """PromptExample fields should be forwarded as kwargs to collector.collect()."""
        import asyncio

        import torch

        from vrl.algorithms.types import TrainStepMetrics
        from vrl.rollouts.evaluators.types import SignalBatch
        from vrl.rollouts.types import ExperienceBatch
        from vrl.trainers.data import PromptExample
        from vrl.trainers.online import OnlineTrainer
        from vrl.trainers.types import TrainerConfig

        captured_kwargs: list[dict] = []

        class _Algorithm:
            class _Config:
                global_std = False
                eps = 1e-8
                adv_clip_max = 5.0

            config = _Config()

            def compute_advantages_from_tensors(self, rewards, group_ids):
                advantages = torch.zeros_like(rewards)
                for gid in torch.unique(group_ids):
                    mask = group_ids == gid
                    gr = rewards[mask]
                    if gr.numel() <= 1:
                        continue
                    mean = gr.mean()
                    std = gr.std().clamp(min=1e-8)
                    advantages[mask] = (gr - mean) / std
                return advantages

            def compute_signal_loss(self, signals, advantages, old_log_probs):
                loss = signals.log_prob.mean()
                return loss, TrainStepMetrics(
                    loss=loss.item(), policy_loss=loss.item(), approx_kl=0.0,
                )

        class _CapturingCollector:
            async def collect(self, prompts, **kwargs):
                captured_kwargs.append(dict(kwargs))
                group_size = int(kwargs.get("group_size", 1))
                return ExperienceBatch(
                    observations=torch.zeros(group_size, 2, 1),
                    actions=torch.zeros(group_size, 2, 1),
                    rewards=torch.ones(group_size, dtype=torch.float32),
                    dones=torch.ones(group_size, dtype=torch.bool),
                    group_ids=torch.zeros(group_size, dtype=torch.long),
                    extras={
                        "log_probs": torch.zeros(group_size, 2, dtype=torch.float32),
                    },
                    prompts=list(prompts) * group_size,
                )

        class _Evaluator:
            def evaluate(self, collector, model, batch, timestep_idx, **kw):
                batch_size = batch.rewards.shape[0]
                return SignalBatch(log_prob=model.weight.view(1).expand(batch_size))

        import torch.nn as nn

        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)

        trainer = OnlineTrainer(
            algorithm=_Algorithm(),
            collector=_CapturingCollector(),
            evaluator=_Evaluator(),
            model=model,
            config=TrainerConfig(lr=0.01, group_size=2, mixed_precision="no"),
            device="cpu",
        )

        example = PromptExample(
            prompt="sign says HELLO",
            target_text="HELLO",
            task_type="text_to_video",
            metadata={"difficulty": "easy"},
        )
        asyncio.run(trainer.step([example]))

        # Group-batched collect: one call per prompt, carrying group_size=2.
        assert len(captured_kwargs) == 1
        kw = captured_kwargs[0]
        assert kw["group_size"] == 2
        assert kw["target_text"] == "HELLO"
        assert kw["task_type"] == "text_to_video"
        assert kw["sample_metadata"]["difficulty"] == "easy"
