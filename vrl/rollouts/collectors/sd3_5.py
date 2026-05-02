"""SD3.5-Medium collector for Flow-GRPO image RL training.

Collector → GenerationRuntime (engine) → SD3_5PipelineExecutor → OutputBatch
            → reward / KL adjustment / ExperienceBatch

Post-Phase-2: the diffusion denoise loop, micro-batching, and decode all
live in :class:`vrl.models.families.sd3_5.executor.SD3_5PipelineExecutor`.
The collector is the RL-semantics layer:

- builds a ``GenerationRequest`` from prompts + config + per-call kwargs;
- submits it through ``GenerationRuntime``;
- scores rewards on the decoded images;
- subtracts ``kl_reward * kl`` from rewards (Flow-GRPO Eq.X);
- packs the result as a trainer-shaped :class:`ExperienceBatch`.

The construction of the runtime (engine loop + scheduler + executor
registry) happens lazily inside ``__init__`` so the public collector
signature is unchanged for callers (e.g. ``vrl/scripts/sd3_5/train.py``).
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vrl.rollouts.collectors.base import Collector
from vrl.rollouts.types import ExperienceBatch

if TYPE_CHECKING:
    from vrl.engine.generation import GenerationRuntime, OutputBatch

logger = logging.getLogger(__name__)

_LAST_COLLECT_PHASES: dict[str, float] = {}


def _sync_time() -> float:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


@dataclass(slots=True)
class SD3_5CollectorConfig:
    """Configuration for SD3_5Collector."""

    # Diffusion sampling — paper B.3: sampling T=10, eval T=40, resolution 512
    num_steps: int = 10
    guidance_scale: float = 4.5
    height: int = 512
    width: int = 512
    max_sequence_length: int = 256

    # Flow-GRPO Eq.9 noise level a (paper B.3: a=0.7)
    noise_level: float = 0.7

    # CFG during sampling
    cfg: bool = True

    # Rollout-side micro-batching: when group_size > sample_batch_size,
    # split group into ceil(group/sample_batch_size) sequential micro-rollouts.
    # Trades wall-time for VRAM so single-GPU can match paper's G=24.
    sample_batch_size: int = 8

    # KL reward — subtract kl_reward * kl from rewards before advantages.
    kl_reward: float = 0.0

    # SDE window — only inject SDE noise for steps within the window.
    sde_window_size: int = 0
    sde_window_range: tuple[int, int] = (0, 10)

    # Same latent — reuse the same noise for samples sharing a prompt.
    same_latent: bool = False

    # Engine-side same-shape request batching. Default 1 preserves the
    # current trainer behavior; raise this for concurrent collect callers.
    max_batch_requests: int = 1


class SD3_5Collector(Collector):
    """Collect rollouts from SD3.5-M with per-step log-probabilities.

    Group rollouts of size ``group_size`` are split into micro-rollouts
    of size ``sample_batch_size``; each micro-rollout runs an independent
    denoise loop and the trajectories are concatenated on the batch dim
    into a single ``ExperienceBatch`` so per-prompt advantage normalization
    sees the full group.
    """

    def __init__(
        self,
        model: Any | None,  # SD3_5Policy
        reward_fn: Any,  # RewardFunction instance
        config: SD3_5CollectorConfig | None = None,
        *,
        runtime: GenerationRuntime | None = None,
    ) -> None:
        self.model = model
        self.reward_fn = reward_fn
        self.config = config or SD3_5CollectorConfig()
        self._runtime: GenerationRuntime | None = runtime

    # ------------------------------------------------------------------
    # Runtime construction
    # ------------------------------------------------------------------

    def _build_runtime(self) -> GenerationRuntime:
        """Lazily wire engine-loop ⇒ generation runtime around ``self.model``.

        Mirrors §3.1.1 of the SGLang-style sprint: P0 reuses the existing
        ``EngineLoop`` + ``Scheduler`` with ``GenerationBatchPlanner``.
        Default config keeps one generation request per tick; increasing
        ``max_batch_requests`` enables
        same-shape request fusion. We register a single ``SD3_5PipelineExecutor`` keyed on
        ``(family="sd3_5", task="t2i")``.
        """
        if self.model is None:
            raise RuntimeError(
                "SD3_5Collector cannot build a local runtime without a model; "
                "inject a GenerationRuntime for distributed rollout.",
            )
        from vrl.engine import (
            EngineLoop,
            Scheduler,
        )
        from vrl.engine.generation import (
            FamilyPipelineRegistry,
            GenerationBatchPlanner,
            GenerationModelRunner,
            GenerationRuntime,
            GenerationWorker,
        )
        from vrl.models.families.sd3_5.executor import SD3_5PipelineExecutor

        registry = FamilyPipelineRegistry()
        registry.register(
            SD3_5PipelineExecutor(
                self.model,
                sample_batch_size=self.config.sample_batch_size,
            ),
        )
        worker = GenerationWorker(registry)
        runner = GenerationModelRunner(worker, execute_in_thread=False)
        engine_loop = EngineLoop(
            scheduler=Scheduler(
                batch_planner=GenerationBatchPlanner(
                    max_batch_size=self.config.max_batch_requests,
                ),
            ),
            model_runner=runner,
        )
        return GenerationRuntime(engine_loop)

    @property
    def runtime(self) -> GenerationRuntime:
        if self._runtime is None:
            self._runtime = self._build_runtime()
        return self._runtime

    async def shutdown(self) -> None:
        if self._runtime is not None:
            await self._runtime.shutdown()
            self._runtime = None

    # ------------------------------------------------------------------
    # collect
    # ------------------------------------------------------------------

    async def collect(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> ExperienceBatch:
        """Collect SD3 rollouts via the engine generation runtime.

        Steps:
        1. Build a ``GenerationRequest`` from config + kwargs.
        2. ``GenerationRuntime.generate(request)`` → ``OutputBatch`` (engine
           runs the denoise loop / decode in
           ``SD3_5PipelineExecutor.forward``).
        3. Score rewards on the decoded images.
        4. Subtract ``kl_reward * kl_total`` from rewards.
        5. Pack into ``ExperienceBatch`` (parity with the pre-migration
           shape — same observations/actions/log_probs/timesteps/kl).
        """
        from vrl.engine.generation import GenerationRequest

        cfg = self.config

        target_text = kwargs.get("target_text", "")
        references = kwargs.get("references", [])
        task_type = kwargs.get("task_type", "text_to_image")
        request_overrides = kwargs.get("request_overrides", {})
        sample_metadata = kwargs.get("sample_metadata", {})
        seed = kwargs.get("seed")
        policy_version = kwargs.get("policy_version")
        group_size = int(kwargs.get("group_size", 1))

        _prof = os.environ.get("VRL_PROFILE_COLLECT") == "1"
        _phases: dict[str, float] = {}
        _t = _sync_time() if _prof else 0.0

        sampling: dict[str, Any] = {
            "num_steps": cfg.num_steps,
            "guidance_scale": cfg.guidance_scale,
            "height": cfg.height,
            "width": cfg.width,
            "noise_level": cfg.noise_level,
            "cfg": cfg.cfg,
            "sample_batch_size": cfg.sample_batch_size,
            "sde_window_size": cfg.sde_window_size,
            "sde_window_range": list(cfg.sde_window_range),
            "same_latent": cfg.same_latent,
            "max_sequence_length": cfg.max_sequence_length,
            "return_kl": cfg.kl_reward > 0,
        }
        if seed is not None:
            sampling["seed"] = seed
        # request_overrides applied last so callers can pin per-prompt num_steps,
        # guidance_scale, etc. (PromptExample.request_overrides).
        sampling.update(request_overrides)

        metadata: dict[str, Any] = dict(sample_metadata)
        if target_text:
            metadata["target_text"] = target_text
        if references:
            metadata["references"] = references
        metadata["task_type"] = task_type

        request = GenerationRequest(
            request_id=f"sd3_5-{uuid.uuid4()}",
            family="sd3_5",
            task="t2i",
            prompts=prompts,
            samples_per_prompt=group_size,
            sampling=sampling,
            return_artifacts={
                "output",
                "rollout_trajectory_data",
                "trajectory_timesteps",
                "trajectory_latents",
                "denoising_env",
            },
            metadata=metadata,
            policy_version=policy_version,
        )

        output = await self.runtime.generate(request)
        if output.error:
            raise RuntimeError(
                f"SD3.5 generation failed (request_id={request.request_id}): "
                f"{output.error}",
            )

        if _prof:
            _now = _sync_time()
            _phases["collect.engine_generate"] = _now - _t
            _t = _now

        batch = await self._output_batch_to_experience_batch(
            output,
            prompts=prompts,
            metadata=metadata,
            phases=_phases if _prof else None,
            phase_t=_t if _prof else None,
        )

        if _prof:
            _LAST_COLLECT_PHASES.clear()
            _LAST_COLLECT_PHASES.update(_phases)

        return batch

    # ------------------------------------------------------------------
    # OutputBatch → ExperienceBatch
    # ------------------------------------------------------------------

    async def _output_batch_to_experience_batch(
        self,
        output: OutputBatch,
        *,
        prompts: list[str],
        metadata: dict[str, Any],
        phases: dict[str, float] | None,
        phase_t: float | None,
    ) -> ExperienceBatch:
        """Translate an engine ``OutputBatch`` into a trainer ``ExperienceBatch``.

        Inverse of the old inline construction: pull rollout trajectory
        tensors out of ``output.rollout_trajectory_data`` and the replay
        embeds out of ``denoising_env.extra``, then run reward + KL
        adjustment.
        """
        import torch

        from vrl.algorithms.types import Rollout, Trajectory

        cfg = self.config

        rt = output.rollout_trajectory_data
        if rt is None or rt.dit_trajectory is None:
            raise RuntimeError(
                "SD3.5 OutputBatch is missing rollout_trajectory_data / dit_trajectory",
            )
        observations = rt.dit_trajectory.latents
        timesteps_tensor = rt.dit_trajectory.timesteps
        log_probs = rt.rollout_log_probs
        if rt.denoising_env is None:
            raise RuntimeError("SD3.5 OutputBatch is missing denoising_env")
        env_extra = rt.denoising_env.extra
        actions = env_extra["actions"]
        kl_tensor = env_extra["kl"]
        training_extras: dict[str, Any] = env_extra["training_extras"]
        rollout_context: dict[str, Any] = env_extra["context"]
        video = env_extra["videos"]

        if observations is None or actions is None or log_probs is None:
            raise RuntimeError(
                "SD3.5 OutputBatch is missing trajectory tensors "
                "(observations/actions/log_probs)",
            )

        batch_size = observations.shape[0]
        device = observations.device

        if len(output.sample_specs) != batch_size:
            raise RuntimeError(
                "SD3.5 OutputBatch sample_specs length does not match batch size",
            )
        effective_prompts = [spec.prompt for spec in output.sample_specs]

        # Score with reward function — one rollout per sample.
        rewards_list: list[float] = []
        for i in range(batch_size):
            dummy_trajectory = Trajectory(
                prompt=effective_prompts[i],
                seed=0,
                steps=[],
                output=video[i],
            )
            dummy_rollout = Rollout(
                request=None,
                trajectory=dummy_trajectory,
                metadata=metadata,
            )
            rewards_list.append(await self.reward_fn.score(dummy_rollout))

        if phases is not None and phase_t is not None:
            _now = _sync_time()
            phases["collect.reward_score"] = _now - phase_t

        rewards_raw = torch.tensor(
            rewards_list, dtype=torch.float32, device=device,
        )
        if cfg.kl_reward > 0:
            kl_total = kl_tensor.sum(dim=1)
            rewards_adjusted = rewards_raw - cfg.kl_reward * kl_total
        else:
            rewards_adjusted = rewards_raw

        dones = torch.ones(batch_size, dtype=torch.bool, device=device)
        group_ids = torch.tensor(
            [spec.prompt_index for spec in output.sample_specs],
            dtype=torch.long,
            device=device,
        )

        extras: dict[str, Any] = {
            "log_probs": log_probs,
            "timesteps": timesteps_tensor,
            "kl": kl_tensor,
            "reward_before_kl": rewards_raw,
        }
        extras.update(training_extras)

        return ExperienceBatch(
            observations=observations,
            actions=actions,
            rewards=rewards_adjusted,
            dones=dones,
            group_ids=group_ids,
            extras=extras,
            context=rollout_context,
            videos=video,
            prompts=effective_prompts,
        )
