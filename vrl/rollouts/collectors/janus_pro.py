"""Janus-Pro rollout collector for token-level GRPO.

Pairs ``vrl.models.families.janus_pro.JanusProPolicy`` with the generic
``OnlineTrainer`` CEA pipeline:

    Collector  →  GenerationRuntime (engine)  →  JanusProPipelineExecutor
                                                          |
                                                          v
                                                    OutputBatch
                                                          |
                Collector  ←  reward / ExperienceBatch  ←  ┘

Per-call lifecycle (post-Phase-7):

  1. Build a ``GenerationRequest`` from prompts + config + per-call
     kwargs (group size, target text, references, seed).
  2. Submit it through ``GenerationRuntime`` — the executor handles
     prompt tokenisation, AR sampling under CFG, and VQ decode, and
     returns an ``OutputBatch`` with images + per-token logprobs +
     prompt/uncond tokens.
  3. Score rewards on the decoded images.
  4. Pack the trainer-shaped :class:`ExperienceBatch` (preserving the
     exact field layout that ``JanusProPolicy.replay_forward`` and
     ``TokenLogProbEvaluator`` read).

The runtime construction (engine loop + scheduler + executor registry)
is lazy in ``__init__`` — the public collector signature is unchanged
for ``vrl/scripts/janus_pro/train.py``.
"""

from __future__ import annotations

import inspect
import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vrl.rollouts.types import ExperienceBatch

if TYPE_CHECKING:  # pragma: no cover
    from vrl.engine.generation import GenerationRuntime, OutputBatch
    from vrl.models.families.janus_pro.policy import JanusProPolicy
    from vrl.rewards.base import RewardFunction

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class JanusProCollectorConfig:
    """Configuration for ``JanusProCollector``."""

    n_samples_per_prompt: int = 8
    cfg_weight: float = 5.0
    temperature: float = 1.0
    image_token_num: int = 576
    image_size: int = 384
    # Hand to the reward layer in [0, 1] (PIL-style); set False to keep [-1, 1].
    rescale_to_unit: bool = True
    # Optional cap on per-rollout text length (truncates prompt encoding).
    max_text_length: int = 256

    # Engine-side same-shape request batching. Default 1 preserves the
    # current trainer behavior; raise this for concurrent collect callers.
    max_batch_requests: int = 1


class JanusProCollector:
    """Collect on-policy rollouts from a ``JanusProPolicy`` wrapper.

    Implements the same ``Collector`` Protocol as ``Wan_2_1Collector``
    so ``OnlineTrainer`` can use it without code changes.
    """

    def __init__(
        self,
        model: JanusProPolicy,
        reward_fn: RewardFunction | None = None,
        config: JanusProCollectorConfig | None = None,
    ) -> None:
        self.model = model
        self.reward_fn = reward_fn
        self.config = config or JanusProCollectorConfig()
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
        same-shape request fusion. We register a single ``JanusProPipelineExecutor`` keyed on
        ``(family="janus_pro", task="ar_t2i")``.
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
        from vrl.models.families.janus_pro.executor import (
            JanusProPipelineExecutor,
        )

        registry = FamilyPipelineRegistry()
        registry.register(JanusProPipelineExecutor(self.model))
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
    # Public: rollout
    # ------------------------------------------------------------------

    async def collect(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> ExperienceBatch:
        """Sample ``n_samples_per_prompt`` rollouts per prompt and score them.

        Synchronous under the hood — the ``async`` signature is purely to
        match the ``Collector`` protocol used by ``OnlineTrainer``.

        ``group_size`` kwarg (passed by ``OnlineTrainer._step_cea``) overrides
        ``n_samples_per_prompt`` for that call; this matches the
        ``Wan_2_1Collector`` contract so the two collectors are swap-ins
        for each other.
        """
        from vrl.engine.generation import GenerationRequest

        cfg = self.config
        device = self.model.device

        n_per = int(kwargs.get("group_size") or cfg.n_samples_per_prompt)

        # Forward PromptExample-level metadata so OCR/ref-based rewards
        # can read target_text / references. Shared across all group samples.
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

        seed = kwargs.get("seed")

        sampling: dict[str, Any] = {
            "cfg_weight": cfg.cfg_weight,
            "temperature": cfg.temperature,
            "image_token_num": cfg.image_token_num,
            "image_size": cfg.image_size,
            "max_text_length": cfg.max_text_length,
        }
        if seed is not None:
            sampling["seed"] = seed

        request = GenerationRequest(
            request_id=f"janus_pro-{uuid.uuid4()}",
            family="janus_pro",
            task="ar_t2i",
            prompts=prompts,
            samples_per_prompt=n_per,
            sampling=sampling,
            return_artifacts={"output", "token_ids", "token_log_probs"},
            metadata=dict(rollout_metadata),
        )

        output = await self.runtime.generate(request)
        if output.error:
            raise RuntimeError(
                f"Janus-Pro generation failed (request_id={request.request_id}): "
                f"{output.error}",
            )

        return await self._output_batch_to_experience_batch(
            output,
            prompts=prompts,
            n_per=n_per,
            device=device,
            rollout_metadata=rollout_metadata,
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
        device: torch.device,
        rollout_metadata: dict[str, Any],
    ) -> ExperienceBatch:
        """Translate engine ``OutputBatch`` → trainer ``ExperienceBatch``.

        Field map — preserves exactly the pre-migration shape so
        ``JanusProPolicy.replay_forward`` and ``TokenLogProbEvaluator``
        keep working without changes:

        - ``observations`` ← ``extra["prompt_input_ids"]`` unsqueezed to
          ``[B, 1, L_text]`` (OnlineTrainer's CEA path expects
          ``observations.shape[1] == num_timesteps`` — AR has T=1).
        - ``actions`` ← ``extra["token_ids"]`` ``[B, L_img]``.
        - ``extras["log_probs"]`` ← ``extra["token_log_probs"]`` unsqueezed
          to ``[B, 1, L_img]``.
        - ``extras["token_mask"]`` ← ``extra["token_mask"]`` ``[B, L_img]``.
        - ``extras["prompt_attention_mask"]`` ← ``extra["prompt_attention_mask"]``.
        - ``extras["uncond_input_ids"]`` ← ``extra["uncond_input_ids"]``.
        - ``extras["uncond_attention_mask"]`` ← ``extra["uncond_attention_mask"]``.
        - ``videos`` ← ``output.output.unsqueeze(2)`` (shape
          ``[B, 3, 1, H, W]`` — reward layer expects T dim).
        - ``group_ids`` ← prompt-major indices ``[0,0,...,1,1,...]``.
        """
        cfg = self.config
        images = output.output  # [B, 3, H, W] in [-1, 1]

        repeated_prompts = [p for p in prompts for _ in range(n_per)]
        group_ids = torch.arange(len(prompts), device=device).repeat_interleave(
            n_per,
        )

        if cfg.rescale_to_unit:
            images_for_reward = (images + 1.0) * 0.5
            images_for_reward = images_for_reward.clamp(0.0, 1.0)
        else:
            images_for_reward = images

        rewards = await self._score(
            images_for_reward, repeated_prompts, rollout_metadata,
        )

        # Pull executor extras.
        token_ids = output.extra["token_ids"]              # [B, L_img]
        token_log_probs = output.extra["token_log_probs"]  # [B, L_img]
        token_mask = output.extra["token_mask"]            # [B, L_img]
        prompt_ids = output.extra["prompt_input_ids"]      # [B, L_text]
        prompt_mask = output.extra["prompt_attention_mask"]
        uncond_ids = output.extra["uncond_input_ids"]
        uncond_mask = output.extra["uncond_attention_mask"]
        rollout_context = dict(output.extra.get("context", {}))

        # OnlineTrainer CEA convention (see Wan_2_1Collector):
        #   observations shape[1] == num_timesteps  (AR has 1 "step")
        #   extras["log_probs"] shape == [B, num_timesteps, ...]
        observations = prompt_ids.unsqueeze(1)             # [B, 1, L_text]
        log_probs_3d = token_log_probs.detach().unsqueeze(1)  # [B, 1, L_img]

        return ExperienceBatch(
            observations=observations,          # [B, 1, L_text]
            actions=token_ids,                  # sampled image tokens [B, L_img]
            rewards=rewards,                    # [B]
            dones=torch.ones(
                len(repeated_prompts), dtype=torch.bool, device=device,
            ),
            group_ids=group_ids,                # [B]
            extras={
                "log_probs": log_probs_3d,                   # [B, 1, L_img]
                "prompt_attention_mask": prompt_mask,        # [B, L_text]
                "uncond_input_ids": uncond_ids,              # [B, L_text]
                "uncond_attention_mask": uncond_mask,        # [B, L_text]
                "token_mask": token_mask,                    # [B, L_img]
            },
            context=rollout_context,
            videos=images.unsqueeze(2),         # [B, 3, 1, H, W]
            prompts=repeated_prompts,
        )

    # ------------------------------------------------------------------
    # Reward scoring
    # ------------------------------------------------------------------

    async def _score(
        self,
        images: torch.Tensor,    # [B, 3, H, W] in [0, 1] (or [-1, 1])
        prompts: list[str],
        rollout_metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Run reward model. Returns ``[B]`` float tensor on the model device.

        Reward-fn contract (shared with ``Wan_2_1Collector``): each
        ``RewardFunction`` exposes ``async score(rollout: Rollout) -> float``
        and reads ``rollout.trajectory.output`` (tensor in [0, 1]) to score.
        We wrap each sample into a minimal ``Rollout`` so aesthetic / CLIP /
        PickScore / MultiReward / OCR work unchanged.

        ``rollout_metadata`` is shared across every rollout in this call
        (typically ``target_text`` + ``references`` from ``PromptExample``)
        so e.g. ``OCRReward`` can look up ``rollout.metadata["target_text"]``.
        """
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

        return torch.tensor(
            [float(s) for s in raw], device=device, dtype=torch.float32,
        )
