"""Online RL trainer — CEA pipeline (Collector + Evaluator + Algorithm).

collect -> evaluate -> advantage -> loss -> backward -> step.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn

from vrl.algorithms.base import Algorithm
from vrl.algorithms.types import TrainStepMetrics
from vrl.rollouts.types import ExperienceBatch, stack_batches
from vrl.trainers.base import Trainer
from vrl.trainers.ema import EMAModuleWrapper
from vrl.trainers.types import TrainerConfig, TrainState
from vrl.trainers.weight_sync import WeightSyncer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def _create_optimizer(
    parameters: Any,
    config: TrainerConfig,
) -> torch.optim.Optimizer:
    """Create an AdamW optimizer."""
    optim = config.optim
    if optim.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Install bitsandbytes for 8-bit Adam: pip install bitsandbytes"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    return optimizer_cls(
        parameters,
        lr=optim.lr,
        betas=(optim.adam_beta1, optim.adam_beta2),
        weight_decay=optim.weight_decay,
        eps=optim.eps,
    )


# ---------------------------------------------------------------------------
# Phase profiler
# ---------------------------------------------------------------------------

class PhaseTimer:
    """Accumulating phase timer with optional CUDA sync.

    Each ``time(name)`` call returns a context manager whose wall time is
    added to ``self.times[name]``. When ``sync=True`` and CUDA is available,
    ``torch.cuda.synchronize()`` is called on both ends so async GPU kernels
    are captured.
    """

    def __init__(self, enabled: bool = False, sync: bool = True) -> None:
        self.enabled = enabled
        self.sync = sync and torch.cuda.is_available()
        self.times: dict[str, float] = defaultdict(float)

    @contextlib.contextmanager
    def time(self, name: str):
        if not self.enabled:
            yield
            return
        if self.sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.sync:
                torch.cuda.synchronize()
            self.times[name] += time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Autocast helper
# ---------------------------------------------------------------------------

def _get_autocast(config: TrainerConfig, device: torch.device) -> Any:
    """Return a bf16 autocast context manager (or no-op when disabled)."""
    if config.bf16:
        return torch.amp.autocast(str(device), dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _apply_sample_mask(batch: ExperienceBatch, mask: torch.Tensor) -> ExperienceBatch:
    """Filter ExperienceBatch along sample dim by a boolean mask.

    All per-sample tensors (observations, actions, rewards, dones, group_ids,
    videos, and extras whose leading dim matches the batch) are indexed by
    ``mask``. Non-per-sample extras and context are carried through unchanged.
    """
    new_extras: dict[str, Any] = {}
    batch_size = mask.shape[0]
    for k, v in batch.extras.items():
        if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == batch_size:
            new_extras[k] = v[mask]
        else:
            new_extras[k] = v
    videos = batch.videos[mask] if batch.videos is not None else None
    if batch.prompts is not None:
        mask_list = mask.detach().cpu().tolist()
        prompts = [p for p, m in zip(batch.prompts, mask_list) if m]
    else:
        prompts = None
    return ExperienceBatch(
        observations=batch.observations[mask],
        actions=batch.actions[mask],
        rewards=batch.rewards[mask],
        dones=batch.dones[mask],
        group_ids=batch.group_ids[mask],
        extras=new_extras,
        context=batch.context,
        videos=videos,
        prompts=prompts,
    )


# ---------------------------------------------------------------------------
# OnlineTrainer
# ---------------------------------------------------------------------------

class OnlineTrainer(Trainer):
    """Orchestrates the CEA online RL loop.

    Pipeline: collect -> evaluate -> advantage -> loss -> backward -> step.
    """

    def __init__(
        self,
        algorithm: Algorithm,
        collector: Any,
        evaluator: Any,
        model: nn.Module,
        ref_model: nn.Module | None = None,
        weight_syncer: WeightSyncer | None = None,
        config: TrainerConfig | None = None,
        prompts: list[str] | None = None,
        device: torch.device | str = "cuda",
        accelerator: Any | None = None,
        stat_tracker: Any | None = None,
    ) -> None:
        self.algorithm = algorithm
        self.collector = collector
        self.evaluator = evaluator
        self.model = model
        self.ref_model = ref_model
        self.weight_syncer = weight_syncer
        self.config = config or TrainerConfig()
        self.prompts = prompts or []
        self.device = torch.device(device) if isinstance(device, str) else device
        self.state = TrainState()
        self.accelerator = accelerator
        # Optional per-prompt history stat tracker (e.g. PerPromptStatTracker).
        # When set, trainer uses tracker-derived advantages (long-horizon
        # normalization) and applies zero-advantage sample filtering.
        self.stat_tracker = stat_tracker

        self._optimizer: torch.optim.Optimizer | None = None
        self._ema: EMAModuleWrapper | None = None

        if self.config.optim.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    def _ensure_optimizer(self) -> torch.optim.Optimizer:
        if self._optimizer is None:
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            self._optimizer = _create_optimizer(trainable, self.config)
        return self._optimizer

    def _ensure_ema(self) -> EMAModuleWrapper | None:
        if not self.config.ema.enable:
            return None
        if self._ema is None:
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            self._ema = EMAModuleWrapper(
                trainable,
                decay=self.config.ema.decay,
                update_step_interval=self.config.ema.update_interval,
                device=self.device,
            )
        return self._ema

    # ------------------------------------------------------------------
    # Accelerator-aware backward/step helpers
    # ------------------------------------------------------------------

    def _backward(self, loss: Any) -> None:
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()

    def _clip_and_step(self, optimizer: Any) -> float:
        """Clip grads, step optimizer, return pre-clip total grad-norm (float)."""
        cfg = self.config
        grad_norm: Any = 0.0
        if self.accelerator is not None:
            if self.accelerator.sync_gradients and cfg.max_norm > 0:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), cfg.max_norm
                )
        else:
            if cfg.max_norm > 0:
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.max_norm
                )
            else:
                # no clip — compute norm manually for diagnostic
                sq_sum = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        sq_sum += float(p.grad.detach().pow(2).sum().item())
                grad_norm = sq_sum ** 0.5
        optimizer.step()
        optimizer.zero_grad()
        return float(grad_norm) if hasattr(grad_norm, "__float__") else float(grad_norm)

    # ------------------------------------------------------------------
    # Training step — CEA pipeline
    # ------------------------------------------------------------------

    async def step(self, prompts: list[str] | None = None) -> TrainStepMetrics:
        """Run one full training step: collect -> evaluate -> advantage -> loss -> backward -> step."""
        from vrl.rollouts.evaluators.types import SignalRequest
        from vrl.trainers.data import PromptExample

        if prompts is not None:
            self.prompts = prompts

        cfg = self.config
        optimizer = self._ensure_optimizer()
        ema = self._ensure_ema()

        timer = PhaseTimer(enabled=cfg.profile)

        # 1. Collect group_size samples per prompt
        all_batches: list[ExperienceBatch] = []
        with timer.time("collect"):
            for prompt_idx, item in enumerate(self.prompts):
                if isinstance(item, PromptExample):
                    prompt_str = item.prompt
                    collect_kwargs: dict[str, Any] = {
                        "target_text": item.target_text,
                        "references": item.references,
                        "task_type": item.task_type,
                        "request_overrides": item.request_overrides,
                        "sample_metadata": item.metadata,
                    }
                else:
                    prompt_str = str(item)
                    collect_kwargs = {}

                # Group-batched collect: one call produces cfg.n samples.
                b = await self.collector.collect(
                    [prompt_str],
                    group_size=cfg.n,
                    **collect_kwargs,
                )
                b.group_ids[:] = prompt_idx
                all_batches.append(b)

            # Gradient accumulation: keep per-prompt batches separate so each
            # forward/backward sees only `group_size` samples. Stacking 4+
            # prompts into one tensor blows past 31GB even at group_size=4;
            # by accumulating gradients across small batches we keep memory
            # at per-prompt collector scale but still get the effective
            # batch = prompts_per_step × group_size for optimizer update.

        # 2. Compute advantages (per-prompt normalization).
        # Rewards + prompts are concatenated across all collected batches so
        # the tracker sees every prompt together and groups properly. The
        # resulting advantages are then split back per-batch for the
        # per-batch training loop.
        tracker_group_size: float = 0.0
        tracker_trained_prompt_num: int = 0
        with timer.time("advantage"):
            all_rewards = torch.cat([b.rewards for b in all_batches])
            all_prompts_flat: list[str] = []
            for b in all_batches:
                assert b.prompts is not None, "batch.prompts must be populated"
                all_prompts_flat.extend(b.prompts)
            all_group_ids = torch.cat([b.group_ids for b in all_batches])

            if self.stat_tracker is not None:
                adv_np = self.stat_tracker.update(all_prompts_flat, all_rewards)
                tracker_group_size, tracker_trained_prompt_num = (
                    self.stat_tracker.get_stats()
                )
                self.stat_tracker.clear()
                advantages_all = torch.as_tensor(
                    adv_np, dtype=torch.float32, device=all_rewards.device,
                )
                adv_clip_max = getattr(self.algorithm.config, "adv_clip_max", None)
                if adv_clip_max is not None:
                    advantages_all = torch.clamp(
                        advantages_all, -adv_clip_max, adv_clip_max
                    )
            else:
                advantages_all = self.algorithm.compute_advantages_from_tensors(
                    all_rewards, all_group_ids,
                )

        # Advantage diagnostics on the full (pre-filter) advantages.
        _adv_abs = advantages_all.detach().abs()
        _total = max(advantages_all.numel(), 1)
        adv_zero_rate = float((_adv_abs < 1e-6).sum().item()) / _total
        _clip_max = getattr(self.algorithm.config, "adv_clip_max", None)
        adv_saturation = (
            float((_adv_abs >= _clip_max - 1e-6).sum().item()) / _total
            if _clip_max is not None else 0.0
        )

        pre_filter_reward_mean = all_rewards.mean().item()
        pre_filter_reward_std = (
            all_rewards.std().item() if all_rewards.numel() > 1 else 0.0
        )
        pre_filter_adv_mean = advantages_all.mean().item()

        # Split advantages back per-batch for the gradient-accumulation loop.
        split_sizes = [b.rewards.shape[0] for b in all_batches]
        adv_split = list(torch.split(advantages_all, split_sizes))

        # Zero-advantage sample filter + full-dead fallback, applied *per-batch*
        # so each filtered batch remains a standalone tensor for forward/backward.
        filtered_batches: list[ExperienceBatch] = []
        filtered_advs: list[torch.Tensor] = []
        if self.stat_tracker is not None:
            for b, adv_b in zip(all_batches, adv_split):
                mask = adv_b.detach().abs() != 0
                if not bool(mask.any()):
                    adv_b = adv_b + 1e-6
                    mask = adv_b.detach().abs() != 0
                if not bool(mask.all()):
                    b = _apply_sample_mask(b, mask)
                    adv_b = adv_b[mask]
                if b.rewards.shape[0] > 0:
                    filtered_batches.append(b)
                    filtered_advs.append(adv_b)
        else:
            filtered_batches = list(all_batches)
            filtered_advs = adv_split

        # 3. Train loop — gradient accumulation across per-prompt batches.
        self.model.train()
        autocast_ctx = _get_autocast(cfg, self.device)
        agg_metrics: dict[str, list[float]] = defaultdict(list)

        # If every batch was filtered out (all dead), skip training this step.
        if not filtered_batches:
            logger.info(
                "step %d: all batches filtered (zero advantages); skipping backward",
                self.state.step,
            )
            # Early exit — still advance state + return metrics with zeros.
            self.state.step += 1
            reward_mean = pre_filter_reward_mean
            reward_std = pre_filter_reward_std
            return TrainStepMetrics(
                loss=0.0, policy_loss=0.0, kl_penalty=0.0,
                reward_mean=reward_mean, reward_std=reward_std,
                advantage_mean=pre_filter_adv_mean,
                clip_fraction=0.0, approx_kl=0.0, grad_norm=0.0,
                adv_saturation=adv_saturation, adv_zero_rate=adv_zero_rate,
                phase_times=dict(timer.times),
            )

        # Timestep schedule — same num_timesteps across all batches (collector
        # uses the same scheduler), so pick from first filtered batch.
        num_timesteps = filtered_batches[0].observations.shape[1]
        train_timestep_count = max(1, int(num_timesteps * cfg.timestep_fraction))
        if train_timestep_count < num_timesteps:
            step_size = num_timesteps / train_timestep_count
            train_indices = [int(i * step_size) for i in range(train_timestep_count)]
        else:
            train_indices = list(range(num_timesteps))

        # Number of accumulation micro-batches (loss scaled by this so total
        # gradient magnitude equals a single forward over the stacked batch).
        num_accum = len(filtered_batches)

        # Debug first step: compare old vs fresh log-probs on first timestep
        # (using first filtered batch so memory footprint is bounded).
        if cfg.debug.first_step and self.state.step == 0:
            _dbg_batch = filtered_batches[0]
            _dbg_old_lp = _dbg_batch.extras["log_probs"]
            with autocast_ctx:
                _dbg_signals = self.evaluator.evaluate(
                    self.collector,
                    self.model,
                    _dbg_batch,
                    0,
                    ref_model=self.ref_model,
                    signal_request=SignalRequest(need_ref=False, need_kl_intermediates=False),
                )
            _old_lp_0 = _dbg_old_lp[:, 0] if _dbg_old_lp.ndim > 1 else _dbg_old_lp
            _diff = (_dbg_signals.log_prob - _old_lp_0).abs()
            logger.info(
                "DEBUG first-step log-prob diff: mean=%.6f max=%.6f | "
                "old_lp[0]=%.6f fresh_lp[0]=%.6f",
                _diff.mean().item(), _diff.max().item(),
                _old_lp_0[0].item(), _dbg_signals.log_prob[0].item(),
            )

        if cfg.debug.grad_split:
            import sys as _sys
            _msg = (
                f"\n[GRAD-SPLIT TRACER] about to enter inner loop: "
                f"step={self.state.step} ppo_epochs={cfg.ppo_epochs} "
                f"num_filtered_batches={len(filtered_batches)} "
                f"num_train_indices={len(train_indices)}\n"
            )
            print(_msg, file=_sys.stderr, flush=True)
            print(_msg, flush=True)
            logger.info(_msg.strip())
            try:
                with open("/tmp/grad_split_debug.log", "a") as _f:
                    _f.write(_msg)
            except Exception:
                pass
        for _inner_epoch in range(cfg.ppo_epochs):
            # Accumulate gradients across all per-prompt batches, then step once.
            for b, adv_b in zip(filtered_batches, filtered_advs):
                old_lp = b.extras["log_probs"]
                for j in train_indices:
                    with timer.time("evaluate"):
                        with autocast_ctx:
                            signals = self.evaluator.evaluate(
                                self.collector,
                                self.model,
                                b,
                                j,
                                ref_model=self.ref_model,
                                signal_request=SignalRequest(
                                    need_ref=self.algorithm.config.init_kl_coef > 0,
                                    need_kl_intermediates=self.algorithm.config.init_kl_coef > 0,
                                ),
                            )
                            old_lp_j = old_lp[:, j] if old_lp.ndim > 1 else old_lp
                            loss, metrics = self.algorithm.compute_signal_loss(
                                signals, adv_b, old_lp_j
                            )
                            # Scale loss by num_accum so the accumulated
                            # gradient matches a single pass over the full
                            # stacked batch in magnitude.
                            loss = loss / num_accum

                    # Grad-split diagnostic: fire ONCE per process on first
                    # backward we actually reach, to verify the KL term is
                    # not drowning the policy gradient. Stamps a class flag
                    # to ensure single-shot.
                    _grad_split_fired = getattr(
                        OnlineTrainer, "_grad_split_already_fired", False
                    )
                    if cfg.debug.grad_split and not _grad_split_fired:
                        OnlineTrainer._grad_split_already_fired = True  # type: ignore[attr-defined]
                        import sys as _sys
                        _enter = f"\n[GRAD-SPLIT] entering diagnostic block (step={self.state.step}, j={j})\n"
                        print(_enter, file=_sys.stderr, flush=True)
                        print(_enter, flush=True)
                        logger.info(_enter.strip())
                        try:
                            with open("/tmp/grad_split_debug.log", "a") as _f:
                                _f.write(_enter)
                        except Exception:
                            pass
                        try:
                            import torch as _t
                            p_t = getattr(self.algorithm, "_last_policy_loss_tensor", None)
                            k_t = getattr(self.algorithm, "_last_kl_term_tensor", None)
                            params = [p for p in self.model.parameters() if p.requires_grad]
                            p_norm = float("nan")
                            k_norm = float("nan")
                            if p_t is not None and p_t.requires_grad:
                                p_grads = _t.autograd.grad(
                                    p_t, params, retain_graph=True, allow_unused=True
                                )
                                p_norm = (
                                    sum((g.detach() ** 2).sum().item() for g in p_grads if g is not None)
                                ) ** 0.5
                            if k_t is not None and k_t.requires_grad:
                                k_grads = _t.autograd.grad(
                                    k_t, params, retain_graph=True, allow_unused=True
                                )
                                k_norm = (
                                    sum((g.detach() ** 2).sum().item() for g in k_grads if g is not None)
                                ) ** 0.5
                            ratio = p_norm / k_norm if k_norm and k_norm > 0 else float("inf")
                            _result = (
                                f"\n[GRAD-SPLIT RESULT] step={self.state.step} j={j} "
                                f"||grad(policy)||={p_norm:.4e} "
                                f"||grad(beta*kl)||={k_norm:.4e} "
                                f"policy/kl_ratio={ratio:.3f} "
                                f"policy_loss={p_t.item() if p_t is not None else float('nan'):.4e} "
                                f"kl_term={k_t.item() if k_t is not None else float('nan'):.4e}\n"
                            )
                            import sys as _sys
                            print(_result, file=_sys.stderr, flush=True)
                            print(_result, flush=True)
                            logger.info(_result.strip())
                            try:
                                with open("/tmp/grad_split_debug.log", "a") as _f:
                                    _f.write(_result)
                            except Exception:
                                pass
                        except Exception as _e:
                            logger.warning("debug_grad_split failed: %s", _e)

                    with timer.time("backward"):
                        self._backward(loss)

                    agg_metrics["loss"].append(metrics.loss)
                    agg_metrics["policy_loss"].append(metrics.policy_loss)
                    agg_metrics["kl_penalty"].append(metrics.kl_penalty)
                    agg_metrics["clip_fraction"].append(metrics.clip_fraction)
                    agg_metrics["approx_kl"].append(metrics.approx_kl)

            # One optimizer step per inner epoch after all micro-batches processed.
            with timer.time("optim_step"):
                _gn = self._clip_and_step(optimizer)
                agg_metrics["grad_norm"].append(_gn)

            if ema is not None:
                trainable = [p for p in self.model.parameters() if p.requires_grad]
                ema.step(trainable, self.state.global_step)

            self.state.global_step += 1

        # Aggregate metrics — each metric averages over its own count (loss/policy
        # appended per-timestep, grad_norm appended per-inner-epoch).
        def avg(key: str) -> float:
            vals = agg_metrics.get(key, [])
            return sum(vals) / len(vals) if vals else 0.0

        reward_mean = pre_filter_reward_mean
        reward_std = pre_filter_reward_std
        adv_mean = pre_filter_adv_mean

        phase_times = dict(timer.times)
        if cfg.profile:
            try:
                from vrl.rollouts.collectors import wan2_1 as _wdc
                phase_times.update(_wdc._LAST_COLLECT_PHASES)
            except Exception:
                pass
        if cfg.profile and phase_times:
            total = sum(
                v for k, v in phase_times.items() if not k.startswith("collect.")
            )
            parts = " | ".join(
                f"{k}={v:.3f}s ({100*v/total:.1f}%)" for k, v in phase_times.items()
            )
            logger.info("phase_times[step=%d] total=%.3fs | %s",
                        self.state.step, total, parts)

        metrics = TrainStepMetrics(
            loss=avg("loss"),
            policy_loss=avg("policy_loss"),
            kl_penalty=avg("kl_penalty"),
            reward_mean=reward_mean,
            reward_std=reward_std,
            advantage_mean=adv_mean,
            clip_fraction=avg("clip_fraction"),
            approx_kl=avg("approx_kl"),
            grad_norm=avg("grad_norm"),
            adv_saturation=adv_saturation,
            adv_zero_rate=adv_zero_rate,
            group_size=tracker_group_size,
            trained_prompt_num=tracker_trained_prompt_num,
            phase_times=phase_times,
        )

        # Update state
        self.state.step += 1
        self.state.total_reward += metrics.reward_mean
        self.state.total_loss += metrics.loss

        # Sync weights
        if self.weight_syncer is not None:
            state_dict = self.model.state_dict()
            await self.weight_syncer.push(state_dict)

        return metrics

    # ------------------------------------------------------------------
    # State dict
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        d: dict[str, Any] = {
            "step": self.state.step,
            "global_step": self.state.global_step,
            "total_reward": self.state.total_reward,
            "total_loss": self.state.total_loss,
        }
        if self._optimizer is not None:
            d["optimizer"] = self._optimizer.state_dict()
        if self._ema is not None:
            d["ema"] = self._ema.state_dict()
        return d

    def load_state_dict(self, state: dict) -> None:
        self.state.step = state.get("step", 0)
        self.state.global_step = state.get("global_step", 0)
        self.state.total_reward = state.get("total_reward", 0.0)
        self.state.total_loss = state.get("total_loss", 0.0)
        if "optimizer" in state and self._optimizer is not None:
            self._optimizer.load_state_dict(state["optimizer"])
        if "ema" in state and self._ema is not None:
            self._ema.load_state_dict(state["ema"])
