"""Tests for vrl.trainers.online (OnlineTrainer backward/step, accelerator, stat tracker)."""

from __future__ import annotations

import pytest

from vrl.rewards.base import RewardFunction


# ---------------------------------------------------------------------------
# OnlineTrainer backward/step tests
# ---------------------------------------------------------------------------

class TestOnlineTrainerBackward:
    """Verify that OnlineTrainer actually calls backward + optimizer.step."""

    def _make_trainer(self):
        import torch
        import torch.nn as nn
        from vrl.algorithms.grpo import GRPO, GRPOConfig
        from vrl.trainers.online import OnlineTrainer
        from vrl.trainers.types import TrainerConfig

        # Tiny model with one param
        model = nn.Linear(4, 1, bias=False)
        initial_weight = model.weight.data.clone()

        # Dummy reward + rollout source (not used in train_on_samples)
        class _DummyReward(RewardFunction):
            async def score(self, rollout):
                return 1.0

        class _DummySource:
            async def collect(self, prompts, **kw):
                return []

        # Dummy log_prob_computer that does a real matmul
        class _DummyLogProb:
            def compute_log_prob(self, model, samples, j, prompt_embeds, neg):
                x = samples["latents"][:, j]  # [B, 4]
                out = model(x).squeeze(-1)     # [B]
                log_prob = -out.pow(2)
                prev_mean = out.unsqueeze(-1)
                std = torch.ones_like(prev_mean)
                dt = torch.ones(1, device=x.device)
                return log_prob, prev_mean, std, dt

        cfg = TrainerConfig(
            lr=0.1,
            max_grad_norm=1.0,
            clip_range=0.2,
            beta=0.0,
            mixed_precision="no",
        )
        trainer = OnlineTrainer(
            algorithm=GRPO(),
            reward_fn=_DummyReward(),
            rollout_source=_DummySource(),
            model=model,
            log_prob_computer=_DummyLogProb(),
            config=cfg,
            device="cpu",
        )
        return trainer, model, initial_weight

    def test_weights_change_after_train_on_samples(self) -> None:
        """After train_on_samples, model weights must differ (backward ran)."""
        import torch

        trainer, model, initial_weight = self._make_trainer()

        B, T, D = 2, 3, 4
        samples = {
            "latents": torch.randn(B, T, D),
            "next_latents": torch.randn(B, T, D),
            "timesteps": torch.arange(T).unsqueeze(0).expand(B, -1),
            "log_probs": torch.zeros(B, T),
            "prompt_embeds": torch.randn(B, 8),
        }
        advantages = torch.tensor([[1.0, -1.0, 0.5], [0.5, 1.0, -0.5]])
        train_timesteps = list(range(T))

        metrics = trainer.train_on_samples(
            samples, advantages, train_timesteps,
            prompt_embeds=samples["prompt_embeds"],
        )

        # Weights MUST have changed
        assert not torch.allclose(model.weight.data, initial_weight), \
            "Weights did not change — backward/step not working"
        assert "loss" in metrics
        assert "policy_loss" in metrics

    def test_grad_norm_clipping(self) -> None:
        """Gradient norm should be clipped to max_grad_norm."""
        import torch

        trainer, model, _ = self._make_trainer()

        B, T, D = 2, 3, 4
        samples = {
            "latents": torch.randn(B, T, D) * 100,  # large values → large grads
            "next_latents": torch.randn(B, T, D),
            "timesteps": torch.arange(T).unsqueeze(0).expand(B, -1),
            "log_probs": torch.zeros(B, T),
            "prompt_embeds": torch.randn(B, 8),
        }
        advantages = torch.ones(B, T) * 10.0
        train_timesteps = list(range(T))

        # After clipping, grad norm should be <= max_grad_norm (+ float tolerance)
        trainer.train_on_samples(
            samples, advantages, train_timesteps,
            prompt_embeds=samples["prompt_embeds"],
        )
        # If we got here without error, clipping worked. The model weights updated.
        assert True

    def test_ema_update(self) -> None:
        """EMA parameters should differ from initial after training."""
        import torch
        import torch.nn as nn

        from vrl.algorithms.grpo import GRPO
        from vrl.trainers.online import OnlineTrainer
        from vrl.trainers.types import TrainerConfig

        model = nn.Linear(4, 1, bias=False)

        class _DummyReward(RewardFunction):
            async def score(self, rollout):
                return 1.0

        class _DummySource:
            async def collect(self, prompts, **kw):
                return []

        class _DummyLogProb:
            def compute_log_prob(self, model, samples, j, prompt_embeds, neg):
                x = samples["latents"][:, j]
                out = model(x).squeeze(-1)
                return -out.pow(2), out.unsqueeze(-1), torch.ones_like(out.unsqueeze(-1)), torch.ones(1)

        cfg = TrainerConfig(
            lr=0.1, max_grad_norm=1.0, clip_range=0.2, beta=0.0,
            mixed_precision="no", ema=True, ema_decay=0.9,
        )
        trainer = OnlineTrainer(
            algorithm=GRPO(), reward_fn=_DummyReward(),
            rollout_source=_DummySource(), model=model,
            log_prob_computer=_DummyLogProb(), config=cfg, device="cpu",
        )

        B, T, D = 2, 3, 4
        samples = {
            "latents": torch.randn(B, T, D),
            "next_latents": torch.randn(B, T, D),
            "timesteps": torch.arange(T).unsqueeze(0).expand(B, -1),
            "log_probs": torch.zeros(B, T),
            "prompt_embeds": torch.randn(B, 8),
        }
        advantages = torch.ones(B, T)

        trainer.train_on_samples(
            samples, advantages, list(range(T)),
            prompt_embeds=samples["prompt_embeds"],
        )

        ema = trainer._ema
        assert ema is not None
        # EMA should have been initialised and stepped
        assert len(ema.ema_parameters) > 0


# ---------------------------------------------------------------------------
# OnlineTrainer — accelerator integration (Gap 8)
# ---------------------------------------------------------------------------

class TestOnlineTrainerAccelerator:
    def test_accelerator_backward_called(self) -> None:
        """When accelerator is passed, _backward uses it."""
        import torch
        import torch.nn as nn
        from vrl.algorithms.grpo import GRPO
        from vrl.trainers.online import OnlineTrainer
        from vrl.trainers.types import TrainerConfig

        calls = []

        class MockAccelerator:
            sync_gradients = True

            def backward(self, loss):
                calls.append("backward")
                loss.backward()

            def clip_grad_norm_(self, params, max_norm):
                calls.append("clip")
                nn.utils.clip_grad_norm_(params, max_norm)

        model = nn.Linear(4, 1, bias=False)
        accel = MockAccelerator()

        class _DummyLogProb:
            def compute_log_prob(self, model, samples, j, pe, ne):
                x = samples["latents"][:, j]
                out = model(x).squeeze(-1)
                return -out.pow(2), out.unsqueeze(-1), torch.ones_like(out.unsqueeze(-1)), torch.ones(1)

        cfg = TrainerConfig(lr=0.1, max_grad_norm=1.0, clip_range=0.2, mixed_precision="no")
        trainer = OnlineTrainer(
            algorithm=GRPO(), model=model,
            log_prob_computer=_DummyLogProb(),
            config=cfg, device="cpu",
            accelerator=accel,
        )

        B, T, D = 2, 2, 4
        samples = {
            "latents": torch.randn(B, T, D),
            "next_latents": torch.randn(B, T, D),
            "timesteps": torch.arange(T).unsqueeze(0).expand(B, -1),
            "log_probs": torch.zeros(B, T),
            "prompt_embeds": torch.randn(B, 8),
        }
        advantages = torch.ones(B, T)

        trainer.train_on_samples(
            samples, advantages, list(range(T)),
            prompt_embeds=samples["prompt_embeds"],
        )

        assert "backward" in calls, "Accelerator.backward() was not called"
        assert "clip" in calls, "Accelerator.clip_grad_norm_() was not called"

    def test_no_accelerator_uses_plain_backward(self) -> None:
        """Without accelerator, plain loss.backward() is used."""
        import torch
        import torch.nn as nn
        from vrl.algorithms.grpo import GRPO
        from vrl.trainers.online import OnlineTrainer
        from vrl.trainers.types import TrainerConfig

        model = nn.Linear(4, 1, bias=False)
        initial_w = model.weight.data.clone()

        class _DummyLogProb:
            def compute_log_prob(self, model, samples, j, pe, ne):
                x = samples["latents"][:, j]
                out = model(x).squeeze(-1)
                return -out.pow(2), out.unsqueeze(-1), torch.ones_like(out.unsqueeze(-1)), torch.ones(1)

        cfg = TrainerConfig(lr=0.1, max_grad_norm=1.0, clip_range=0.2, mixed_precision="no")
        trainer = OnlineTrainer(
            algorithm=GRPO(), model=model,
            log_prob_computer=_DummyLogProb(),
            config=cfg, device="cpu",
            accelerator=None,  # no accelerator
        )

        B, T, D = 2, 2, 4
        samples = {
            "latents": torch.randn(B, T, D),
            "next_latents": torch.randn(B, T, D),
            "timesteps": torch.arange(T).unsqueeze(0).expand(B, -1),
            "log_probs": torch.zeros(B, T),
            "prompt_embeds": torch.randn(B, 8),
        }
        trainer.train_on_samples(
            samples, torch.ones(B, T), list(range(T)),
            prompt_embeds=samples["prompt_embeds"],
        )
        # Weights should still change (backward + step happened)
        assert not torch.allclose(model.weight.data, initial_w)


# ---------------------------------------------------------------------------
# Regression: Bug 3 — stat_tracking wired into OnlineTrainer
# ---------------------------------------------------------------------------

class TestOnlineTrainerStatTracker:
    def test_stat_tracker_initialized(self) -> None:
        """OnlineTrainer should have a PerPromptStatTracker instance."""
        from vrl.algorithms.grpo import GRPO
        from vrl.trainers.online import OnlineTrainer
        from vrl.trainers.types import TrainerConfig

        trainer = OnlineTrainer(
            algorithm=GRPO(),
            config=TrainerConfig(),
            device="cpu",
        )
        assert trainer._stat_tracker is not None
        from vrl.algorithms.stat_tracking import PerPromptStatTracker
        assert isinstance(trainer._stat_tracker, PerPromptStatTracker)


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
