"""Cosmos Predict2 Video2World collector for diffusion RL training.

Collector → GenerationRuntime (engine) → CosmosPipelineExecutor → OutputBatch
            → reward / KL adjustment / ExperienceBatch

Post-Phase-4: the diffusion denoise loop, micro-batching, and decode all
live in :class:`vrl.models.families.cosmos.executor.CosmosPipelineExecutor`.
The collector is the RL-semantics layer:

- builds a ``GenerationRequest`` from prompts + config + per-call kwargs;
- submits it through ``GenerationRuntime``;
- scores rewards on the decoded videos;
- subtracts ``kl_reward * kl`` from rewards;
- packs the result as a trainer-shaped :class:`ExperienceBatch`.

The construction of the runtime (engine loop + scheduler + executor
registry) happens lazily inside ``_build_runtime`` so the public collector
``__init__`` signature is unchanged for callers (e.g.
``vrl/scripts/cosmos/train.py``):

    CosmosPredict2Collector(model, reward_fn, config, *, reference_image=None)

Reference image handling: the collector holds the per-rollout-run
``reference_image`` and forwards it to the engine executor at runtime
construction time (mirroring the executor's
``reference_image=`` constructor argument). Per-call overrides via
``collect(..., reference_image=...)`` flow through the request metadata
and take precedence inside the executor.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vrl.rollouts.collectors.base import Collector
from vrl.rollouts.types import ExperienceBatch

if TYPE_CHECKING:
    from vrl.engine.generation import GenerationRuntime, OutputBatch

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CosmosPredict2CollectorConfig:
    """Configuration for CosmosPredict2Collector."""

    num_steps: int = 35
    guidance_scale: float = 7.0
    height: int = 704
    width: int = 1280
    num_frames: int = 93  # default for Cosmos Predict2 (81 gen + 12 cond)
    max_sequence_length: int = 512
    fps: int = 16

    # CFG during sampling
    cfg: bool = True

    # Rollout-side micro-batching: when group_size > sample_batch_size,
    # split group into ceil(group/sample_batch_size) sequential micro-rollouts.
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


class CosmosPredict2Collector(Collector):
    """Collect rollouts from Cosmos Predict2 V2W with per-step log-probs.

    The denoise loop, prompt encode, micro-batching and decode all run
    inside ``CosmosPipelineExecutor.forward``; this collector only owns
    the reward scoring and ``ExperienceBatch`` assembly.
    """

    def __init__(
        self,
        model: Any | None,  # CosmosPredict2Policy
        reward_fn: Any,  # RewardFunction instance
        config: CosmosPredict2CollectorConfig | None = None,
        *,
        reference_image: Any = None,  # PIL.Image for Video2World conditioning
        runtime: GenerationRuntime | None = None,
    ) -> None:
        self.model = model
        self.reward_fn = reward_fn
        self.config = config or CosmosPredict2CollectorConfig()
        self.reference_image = reference_image
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
        same-shape request fusion. We register a single ``CosmosPipelineExecutor`` keyed on
        ``(family="cosmos", task="v2w")``.
        """
        if self.model is None:
            raise RuntimeError(
                "CosmosPredict2Collector cannot build a local runtime without a "
                "model; inject a GenerationRuntime for distributed rollout.",
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
        from vrl.models.families.cosmos.executor import CosmosPipelineExecutor

        registry = FamilyPipelineRegistry()
        registry.register(
            CosmosPipelineExecutor(
                self.model,
                reference_image=self.reference_image,
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
        """Collect Cosmos Predict2 V2W rollouts via the engine generation runtime.

        Steps:
        1. Build a ``GenerationRequest`` from config + kwargs.
        2. ``GenerationRuntime.generate(request)`` → ``OutputBatch`` (engine
           runs the denoise loop / decode in
           ``CosmosPipelineExecutor.forward``).
        3. Score rewards on the decoded videos.
        4. Subtract ``kl_reward * kl_total`` from rewards.
        5. Pack into ``ExperienceBatch``.
        """
        from vrl.engine.generation import GenerationRequest

        cfg = self.config

        target_text = kwargs.get("target_text", "")
        references = kwargs.get("references", [])
        task_type = kwargs.get("task_type", "video2world")
        request_overrides = kwargs.get("request_overrides", {})
        sample_metadata = kwargs.get("sample_metadata", {})
        seed = kwargs.get("seed")
        policy_version = kwargs.get("policy_version")
        group_size = int(kwargs.get("group_size", 1))
        # Per-call reference_image override; falls through to executor's
        # constructor-level ``reference_image`` when absent.
        per_call_reference_image = kwargs.get("reference_image")

        sampling: dict[str, Any] = {
            "num_steps": cfg.num_steps,
            "guidance_scale": cfg.guidance_scale,
            "height": cfg.height,
            "width": cfg.width,
            "num_frames": cfg.num_frames,
            "fps": cfg.fps,
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
        sampling.update(request_overrides)

        metadata: dict[str, Any] = dict(sample_metadata)
        if target_text:
            metadata["target_text"] = target_text
        if references:
            metadata["references"] = references
        metadata["task_type"] = task_type
        if per_call_reference_image is not None:
            metadata["reference_image"] = per_call_reference_image

        request = GenerationRequest(
            request_id=f"cosmos-{uuid.uuid4()}",
            family="cosmos",
            task="v2w",
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
                f"Cosmos Predict2 generation failed (request_id="
                f"{request.request_id}): {output.error}",
            )

        return await self._output_batch_to_experience_batch(
            output, prompts=prompts, metadata=metadata,
        )

    # ------------------------------------------------------------------
    # OutputBatch → ExperienceBatch
    # ------------------------------------------------------------------

    async def _output_batch_to_experience_batch(
        self,
        output: OutputBatch,
        *,
        prompts: list[str],
        metadata: dict[str, Any],
    ) -> ExperienceBatch:
        """Translate an engine ``OutputBatch`` into a trainer ``ExperienceBatch``.

        Pulls rollout trajectory tensors out of
        ``output.rollout_trajectory_data`` and the replay embeds out of
        ``denoising_env.extra``, then runs reward + KL adjustment.
        """
        import torch

        from vrl.algorithms.types import Rollout, Trajectory

        cfg = self.config

        rt = output.rollout_trajectory_data
        if rt is None or rt.dit_trajectory is None:
            raise RuntimeError(
                "Cosmos OutputBatch is missing rollout_trajectory_data / "
                "dit_trajectory",
            )
        observations = rt.dit_trajectory.latents
        timesteps_tensor = rt.dit_trajectory.timesteps
        log_probs = rt.rollout_log_probs
        if rt.denoising_env is None:
            raise RuntimeError("Cosmos OutputBatch is missing denoising_env")
        env_extra = rt.denoising_env.extra
        actions = env_extra["actions"]
        kl_tensor = env_extra["kl"]
        training_extras: dict[str, Any] = env_extra["training_extras"]
        rollout_context: dict[str, Any] = env_extra["context"]
        video = env_extra["videos"]

        if observations is None or actions is None or log_probs is None:
            raise RuntimeError(
                "Cosmos OutputBatch is missing trajectory tensors "
                "(observations/actions/log_probs)",
            )

        batch_size = observations.shape[0]
        device = observations.device

        if len(output.sample_specs) != batch_size:
            raise RuntimeError(
                "Cosmos OutputBatch sample_specs length does not match batch size",
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
