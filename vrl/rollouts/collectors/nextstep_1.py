"""NextStep-1 rollout collector for token-level GRPO.

Collector → GenerationRuntime (engine) → NextStep1PipelineExecutor → OutputBatch
            → reward / ExperienceBatch

Post-Phase-7: the AR sampling + decode live in
:class:`vrl.models.families.nextstep_1.executor.NextStep1PipelineExecutor`.
This module is the RL-semantics layer:

  1. Build a ``GenerationRequest`` from prompts + config + per-call kwargs.
  2. Submit through ``GenerationRuntime`` and unpack the resulting
     ``OutputBatch`` (engine-side decoded image + replay artifacts).
  3. Score the reward fn on the decoded images.
  4. Pack into a trainer-shaped :class:`ExperienceBatch` whose
     ``actions``/``extras["saved_noise"]``/``extras["log_probs"]`` match
     what ``NextStep1Policy.replay_forward`` reads.

The construction of the runtime (engine loop + scheduler + executor
registry) happens lazily inside ``__init__``; the public collector
signature is unchanged so ``vrl/scripts/nextstep_1/train.py`` still
instantiates ``NextStep1Collector(model, reward_fn, config)`` directly.
"""

from __future__ import annotations

import inspect
import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vrl.rollouts.collectors.base import Collector
from vrl.rollouts.types import ExperienceBatch

if TYPE_CHECKING:  # pragma: no cover
    from vrl.engine.generation import GenerationRuntime, OutputBatch
    from vrl.models.families.nextstep_1.policy import NextStep1Policy
    from vrl.rewards.base import RewardFunction

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NextStep1CollectorConfig:
    """Configuration for ``NextStep1Collector``."""

    n_samples_per_prompt: int = 4
    cfg_scale: float = 4.5
    num_flow_steps: int = 20
    noise_level: float = 1.0
    image_token_num: int = 1024     # 32 x 32 patches per 256^2 image
    image_size: int = 256
    rescale_to_unit: bool = True    # convert [-1, 1] -> [0, 1] for the reward layer
    max_text_length: int = 256
    # Engine-side same-shape request batching. Default 1 preserves the
    # current trainer behavior; raise this for concurrent collect callers.
    max_batch_requests: int = 1


class NextStep1Collector(Collector):
    """Collect on-policy rollouts from a ``NextStep1Policy`` wrapper."""

    def __init__(
        self,
        model: NextStep1Policy,
        reward_fn: RewardFunction | None = None,
        config: NextStep1CollectorConfig | None = None,
    ) -> None:
        self.model = model
        self.reward_fn = reward_fn
        self.config = config or NextStep1CollectorConfig()
        self._runtime: GenerationRuntime | None = None

    # ------------------------------------------------------------------
    # Runtime construction
    # ------------------------------------------------------------------

    def _build_runtime(self) -> GenerationRuntime:
        """Lazily wire engine-loop ⇒ generation runtime around ``self.model``.

        Mirrors §3.1.1 of the SGLang-style sprint: P0 reuses the existing
        ``EngineLoop`` + ``Scheduler`` with ``GenerationBatchPlanner``.
        Default config keeps one generation request per tick; increasing
        ``max_batch_requests`` enables
        same-shape request fusion. We register a single ``NextStep1PipelineExecutor`` keyed on
        ``(family="nextstep_1", task="ar_t2i")``.
        """
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
        from vrl.models.families.nextstep_1.executor import (
            NextStep1PipelineExecutor,
        )

        registry = FamilyPipelineRegistry()
        registry.register(NextStep1PipelineExecutor(self.model))
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
    # Public: collect
    # ------------------------------------------------------------------

    async def collect(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> ExperienceBatch:
        """Collect NextStep-1 rollouts via the engine generation runtime.

        Steps:
          1. Build a ``GenerationRequest`` from config + kwargs.
          2. ``GenerationRuntime.generate(request)`` → ``OutputBatch``
             (engine runs the AR loop + decode in
             ``NextStep1PipelineExecutor.forward``).
          3. Score rewards on the decoded images.
          4. Pack into ``ExperienceBatch`` (shape parity with the
             pre-migration path — ``actions`` = continuous tokens,
             ``extras["saved_noise"]``, ``extras["log_probs"]`` carry the
             replay-determinism artifacts that
             ``NextStep1Policy.replay_forward`` reads).
        """
        from vrl.engine.generation import GenerationRequest

        cfg = self.config
        device = self.model.device

        n_per = int(kwargs.get("group_size") or cfg.n_samples_per_prompt)
        seed = kwargs.get("seed")

        sampling: dict[str, Any] = {
            "cfg_scale": cfg.cfg_scale,
            "num_flow_steps": cfg.num_flow_steps,
            "noise_level": cfg.noise_level,
            "image_token_num": cfg.image_token_num,
            "image_size": cfg.image_size,
            "max_text_length": cfg.max_text_length,
            "rescale_to_unit": cfg.rescale_to_unit,
        }
        if seed is not None:
            sampling["seed"] = seed

        # Forward PromptExample-level metadata for OCR / reference rewards
        rollout_metadata: dict[str, Any] = {}
        target_text = kwargs.get("target_text")
        if target_text:
            rollout_metadata["target_text"] = target_text
        references = kwargs.get("references")
        if references:
            rollout_metadata["references"] = references
        sample_md = kwargs.get("sample_metadata")
        if sample_md:
            rollout_metadata.update(sample_md)

        request = GenerationRequest(
            request_id=f"nextstep_1-{uuid.uuid4()}",
            family="nextstep_1",
            task="ar_t2i",
            prompts=prompts,
            samples_per_prompt=n_per,
            sampling=sampling,
            return_artifacts={"output", "rollout_trajectory_data"},
            metadata={"rollout_metadata": rollout_metadata},
        )

        output = await self.runtime.generate(request)
        if output.error:
            raise RuntimeError(
                f"NextStep-1 generation failed (request_id={request.request_id}): "
                f"{output.error}",
            )

        return await self._output_batch_to_experience_batch(
            output,
            prompts=prompts,
            n_per=n_per,
            rollout_metadata=rollout_metadata,
            device=device,
        )

    # ------------------------------------------------------------------
    # OutputBatch → ExperienceBatch
    # ------------------------------------------------------------------

    async def _output_batch_to_experience_batch(
        self,
        output: OutputBatch,
        *,
        prompts: list[str],
        n_per: int,
        rollout_metadata: dict[str, Any],
        device: Any,
    ) -> ExperienceBatch:
        """Translate an engine ``OutputBatch`` into a trainer ``ExperienceBatch``.

        Inverse of the old inline construction: pull the AR artifacts
        (``tokens``/``saved_noise``/``log_probs``) and prompt-side embeds
        out of ``output.extra``, then run reward.
        """
        extra = output.extra
        tokens = extra["tokens"]                       # [B, L_img, D_token]
        saved_noise = extra["saved_noise"]             # [B, L_img, D_token]
        old_logprobs = extra["log_probs"]              # [B, L_img]
        prompt_ids = extra["prompt_input_ids"]         # [B, L_text]
        prompt_mask = extra["prompt_attention_mask"]   # [B, L_text]
        uncond_ids = extra["uncond_input_ids"]         # [B, L_text]
        uncond_mask = extra["uncond_attention_mask"]   # [B, L_text]
        images = output.output                         # [B, 3, H, W] in [-1, 1]
        images_for_reward = extra["images_for_reward"]
        rollout_context = extra["context"]

        repeated_prompts = [p for p in prompts for _ in range(n_per)]
        group_ids = torch.arange(len(prompts), device=device).repeat_interleave(n_per)

        rewards = await self._score(
            images_for_reward, repeated_prompts, rollout_metadata,
        )

        # Per-token mask: every continuous image-token position counts.
        token_mask = torch.ones_like(old_logprobs)

        # OnlineTrainer CEA convention: observations / log_probs carry a
        # singleton time dim so trainer's ``[:, j]`` indexing yields the
        # correct per-prompt slice.
        observations = prompt_ids.unsqueeze(1)                 # [B, 1, L_text]
        log_probs_3d = old_logprobs.detach().unsqueeze(1)      # [B, 1, L_img]

        return ExperienceBatch(
            observations=observations,           # [B, 1, L_text]
            actions=tokens,                      # continuous tokens [B, L_img, D_token]
            rewards=rewards,                     # [B]
            dones=torch.ones(len(repeated_prompts), dtype=torch.bool, device=device),
            group_ids=group_ids,                 # [B]
            extras={
                "log_probs": log_probs_3d,                     # [B, 1, L_img]
                "prompt_attention_mask": prompt_mask,          # [B, L_text]
                "uncond_input_ids": uncond_ids,                # [B, L_text]
                "uncond_attention_mask": uncond_mask,          # [B, L_text]
                "token_mask": token_mask,                      # [B, L_img]
                "saved_noise": saved_noise,                    # [B, L_img, D_token]
            },
            context=dict(rollout_context),
            videos=images.unsqueeze(2),          # [B, 3, 1, H, W] — reward layer expects T dim
            prompts=repeated_prompts,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _score(
        self,
        images: torch.Tensor,
        prompts: list[str],
        rollout_metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        from vrl.algorithms.types import Rollout, Trajectory

        device = self.model.device
        if self.reward_fn is None:
            return torch.zeros(images.shape[0], device=device)

        meta: dict[str, Any] = dict(rollout_metadata or {})

        rollouts = [
            Rollout(
                request=None,
                trajectory=Trajectory(
                    prompt=prompts[i], seed=0, steps=[], output=images[i],
                ),
                metadata=dict(meta),
            )
            for i in range(images.shape[0])
        ]

        batch_fn = getattr(self.reward_fn, "score_batch", None)
        if batch_fn is not None and inspect.iscoroutinefunction(batch_fn):
            raw = await batch_fn(rollouts)
        else:
            raw = [await self.reward_fn.score(r) for r in rollouts]

        return torch.tensor([float(s) for s in raw], device=device, dtype=torch.float32)
