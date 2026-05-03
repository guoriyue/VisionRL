"""Tests for vrl.trainers.online (OnlineTrainer CEA pipeline)."""

from __future__ import annotations

import pytest

from vrl.rollouts.collectors.base import Collector
from vrl.rollouts.evaluators.base import Evaluator


class TestOnlineTrainerCeaRegressions:
    def _make_cea_trainer(self, rewards: list[float]):
        import torch
        import torch.nn as nn

        from vrl.algorithms.types import TrainStepMetrics
        from vrl.rollouts.evaluators.types import SignalBatch
        from vrl.rollouts.experience import ExperienceBatch
        from vrl.trainers.online import OnlineTrainer
        from vrl.trainers.types import DebugConfig, EMAConfig, OptimConfig, TrainerConfig

        class _Algorithm:
            class _Config:
                global_std = False
                eps = 1e-8
                adv_clip_max = 5.0
                init_kl_coef = 0.0

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

        class _Collector(Collector):
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

        class _Evaluator(Evaluator):
            def evaluate(
                self,
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
                optim=OptimConfig(lr=0.01),
                ema=EMAConfig(),
                debug=DebugConfig(),
                n=2,
                bf16=False,
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
        from vrl.rollouts.experience import ExperienceBatch
        from vrl.trainers.data import PromptExample
        from vrl.trainers.online import OnlineTrainer
        from vrl.trainers.types import DebugConfig, EMAConfig, OptimConfig, TrainerConfig

        captured_kwargs: list[dict] = []

        class _Algorithm:
            class _Config:
                global_std = False
                eps = 1e-8
                adv_clip_max = 5.0
                init_kl_coef = 0.0

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

        class _CapturingCollector(Collector):
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

        class _Evaluator(Evaluator):
            def evaluate(self, model, batch, timestep_idx, **kw):
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
            config=TrainerConfig(
                optim=OptimConfig(lr=0.01),
                ema=EMAConfig(),
                debug=DebugConfig(),
                n=2,
                bf16=False,
            ),
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

    def test_cea_batches_plain_prompts_for_rollout_but_splits_training(self) -> None:
        """Plain prompts should collect together, then train as group-local batches."""
        import asyncio

        import torch
        import torch.nn as nn

        from vrl.algorithms.types import TrainStepMetrics
        from vrl.rollouts.evaluators.types import SignalBatch
        from vrl.rollouts.experience import ExperienceBatch
        from vrl.trainers.online import OnlineTrainer
        from vrl.trainers.types import DebugConfig, EMAConfig, OptimConfig, TrainerConfig

        collect_calls: list[list[str]] = []
        evaluate_batch_sizes: list[int] = []
        evaluate_group_ids: list[list[int]] = []

        class _Algorithm:
            class _Config:
                global_std = False
                eps = 1e-8
                adv_clip_max = 5.0
                init_kl_coef = 0.0

            config = _Config()

            def compute_advantages_from_tensors(self, rewards, group_ids):
                advantages = torch.zeros_like(rewards)
                for gid in torch.unique(group_ids):
                    mask = group_ids == gid
                    gr = rewards[mask]
                    advantages[mask] = gr - gr.mean()
                return advantages

            def compute_signal_loss(self, signals, advantages, old_log_probs):
                loss = signals.log_prob.mean() + advantages.mean() * 0.0
                return loss, TrainStepMetrics(
                    loss=loss.item(),
                    policy_loss=loss.item(),
                    approx_kl=float(old_log_probs.mean().item()),
                )

        class _Collector(Collector):
            async def collect(self, prompts, **kwargs):
                prompts = list(prompts)
                collect_calls.append(prompts)
                group_size = int(kwargs.get("group_size", 1))
                batch_size = len(prompts) * group_size
                group_ids = torch.tensor(
                    [
                        prompt_idx
                        for prompt_idx in range(len(prompts))
                        for _ in range(group_size)
                    ],
                    dtype=torch.long,
                )
                rewards = torch.tensor(
                    [float(i % group_size) for i in range(batch_size)],
                    dtype=torch.float32,
                )
                return ExperienceBatch(
                    observations=torch.zeros(batch_size, 2, 1),
                    actions=torch.zeros(batch_size, 2, 1),
                    rewards=rewards,
                    dones=torch.ones(batch_size, dtype=torch.bool),
                    group_ids=group_ids,
                    extras={"log_probs": torch.zeros(batch_size, 2)},
                    prompts=[p for p in prompts for _ in range(group_size)],
                )

        class _Evaluator(Evaluator):
            def evaluate(self, model, batch, timestep_idx, **kw):
                del timestep_idx, kw
                evaluate_batch_sizes.append(int(batch.rewards.shape[0]))
                evaluate_group_ids.append(
                    [int(x) for x in batch.group_ids.detach().cpu().tolist()]
                )
                return SignalBatch(
                    log_prob=model.weight.view(1).expand(batch.rewards.shape[0])
                )

        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)

        trainer = OnlineTrainer(
            algorithm=_Algorithm(),
            collector=_Collector(),
            evaluator=_Evaluator(),
            model=model,
            config=TrainerConfig(
                optim=OptimConfig(lr=0.01),
                ema=EMAConfig(),
                debug=DebugConfig(),
                n=2,
                bf16=False,
            ),
            device="cpu",
        )

        asyncio.run(trainer.step(["prompt-a", "prompt-b"]))

        assert collect_calls == [["prompt-a", "prompt-b"]]
        assert evaluate_batch_sizes == [2, 2, 2, 2]
        assert evaluate_group_ids == [[0, 0], [0, 0], [1, 1], [1, 1]]
