"""NextStep-1 pipeline executor.

Owns the continuous-token autoregressive sampling loop previously inlined
in the NextStep-1 rollout collector. The
collector keeps reward scoring and ``RolloutBatch`` packing.

Boundary:

- This module MUST NOT import ``vrl.rollouts.*`` or ``RolloutBatch``.
- This module MUST NOT compute reward.
- Inputs come from ``GenerationRequest.sampling`` (collector packs them).
- Outputs are the canonical ``OutputBatch``. NextStep-1 is AR with
  continuous tokens + a flow-matching head, so there is no diffusion
  trajectory; instead, ``output`` is the decoded image and ``extra``
  carries the three replay-determinism artifacts:

      * ``tokens``      [B, L_img, D_token]  — sampled continuous tokens
                                              (used as ``RolloutBatch.actions``)
      * ``saved_noise`` [B, L_img, D_token]  — per-token x_0 prior for the
                                              flow ODE; replay reads this so
                                              ``recompute_logprobs`` can
                                              re-run the same trajectory.
      * ``log_probs``   [B, L_img]           — Gaussian per-token log-prob
                                              from sampling time
                                              (i.e. ``old_log_prob``).

Determinism contract: same prompts + same generator state ⇒ same
``tokens``/``saved_noise``/``log_probs``. The model's
``sample_image_tokens`` is a single black-box call, so determinism reduces
to "we call it once with the same arguments".
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vrl.engine.ar import (
    ActiveSequence,
    ARGenerationSpec,
    ARPipelineExecutorBase,
    ARTokenScheduler,
    max_peak_memory_mb,
    ordered_chunks,
)
from vrl.engine.core.protocols import PipelineChunkResult
from vrl.engine.core.types import (
    GenerationMetrics,
    GenerationRequest,
    GenerationSampleSpec,
    OutputBatch,
    RolloutTrajectoryData,
    WorkloadSignature,
)
from vrl.engine.microbatching import MicroBatchPlan

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NextStep1ARChunkResult(PipelineChunkResult):
    """Output of one prompt/sample NextStep-1 AR chunk."""

    prompt_index: int
    sample_start: int
    sample_count: int
    output: torch.Tensor
    tokens: torch.Tensor
    saved_noise: torch.Tensor
    log_probs: torch.Tensor
    images_for_reward: torch.Tensor
    prompt_input_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor
    uncond_input_ids: torch.Tensor
    uncond_attention_mask: torch.Tensor
    context: dict[str, Any]
    peak_memory_mb: float | None = None


class NextStep1PipelineExecutor(ARPipelineExecutorBase):
    """Continuous-token AR executor for NextStep-1 text-to-image rollouts.

    The collector constructs a ``GenerationRequest`` whose ``sampling``
    dict holds:

    - ``cfg_scale``: float
    - ``num_flow_steps``: int
    - ``noise_level``: float
    - ``image_token_num``: int (L_img — number of continuous image tokens)
    - ``image_size``: int (passed to ``decode_image_tokens``)
    - ``max_text_length``: int
    - ``rescale_to_unit``: bool (post-decode pixel rescale [-1,1] → [0,1])
    - ``seed``: int | None

    And whose ``metadata`` may carry ``rollout_metadata`` (target_text,
    references, etc.) for the collector's reward layer.

    The executor returns an ``OutputBatch`` whose ``extra`` carries the
    three NextStep-1-specific artifacts ``tokens``/``saved_noise``/
    ``log_probs`` plus the prompt-side replay context
    (``prompt_input_ids``, ``prompt_attention_mask``, ``uncond_*``).
    There is no DiT trajectory: ``rollout_trajectory_data.dit_trajectory``
    is ``None`` and ``denoising_env`` is ``None``.
    """

    family: str = "nextstep_1"
    task: str = "ar_t2i"
    default_image_token_num: int | None = None
    default_image_size: int | None = None

    def __init__(
        self,
        model: Any,  # NextStep1Policy
    ) -> None:
        self.model = model

    # -- protocol ------------------------------------------------------

    def workload_signature(self, request: GenerationRequest) -> WorkloadSignature:
        return WorkloadSignature.from_request(request)

    def forward(
        self,
        request: GenerationRequest,
        sample_specs: list[GenerationSampleSpec],
    ) -> OutputBatch:
        sampling = request.sampling
        spec: ARGenerationSpec = self.parse_spec(request)
        prompts = list(request.prompts)

        cfg_scale = float(sampling["cfg_scale"])
        num_flow_steps = int(sampling["num_flow_steps"])
        noise_level = float(sampling["noise_level"])
        rescale_to_unit = bool(sampling.get("rescale_to_unit", True))

        # Repeat each prompt ``samples_per_prompt`` times so the AR loop
        # sees a flat ``[B, ...]`` batch where ``B = num_prompts x G``.
        repeated_prompts = self.expand_prompts(request)

        prompt_ids, prompt_mask = self._tokenize_prompts(
            repeated_prompts,
            max_text_length=spec.max_text_length,
        )
        uncond_ids, uncond_mask = self._tokenize_prompts(
            [""] * len(repeated_prompts),
            max_text_length=spec.max_text_length,
        )
        pad_id = getattr(self.model.processor, "pad_token_id", None) or 0
        prompt_ids, prompt_mask, uncond_ids, uncond_mask = self.align_pair(
            prompt_ids,
            prompt_mask,
            uncond_ids,
            uncond_mask,
            pad_id=pad_id,
        )

        cond_embeds = self._embed(prompt_ids)
        uncond_embeds = self._embed(uncond_ids)

        # Optional deterministic generator. ``sample_image_tokens`` accepts
        # a ``generator`` kwarg (see NextStep1Policy.sample_image_tokens).
        generator: torch.Generator | None = None
        if spec.seed is not None:
            device = self.model.device
            generator = torch.Generator(device=device)
            generator.manual_seed(spec.seed)

        sample_kwargs: dict[str, Any] = {
            "cfg_scale": cfg_scale,
            "num_flow_steps": num_flow_steps,
            "noise_level": noise_level,
            "image_token_num": spec.image_token_num,
        }
        if generator is not None:
            sample_kwargs["generator"] = generator

        if spec.use_ar_scheduler:
            tokens, saved_noise, old_logprobs = self._sample_with_ar_scheduler(
                request=request,
                sample_specs=sample_specs,
                cond_embeds=cond_embeds,
                uncond_embeds=uncond_embeds,
                prompt_mask=prompt_mask,
                uncond_mask=uncond_mask,
                image_token_num=spec.image_token_num,
                sample_kwargs=sample_kwargs,
            )
        else:
            tokens, saved_noise, old_logprobs = self.model.sample_image_tokens(
                cond_embeds,
                uncond_embeds,
                prompt_mask,
                uncond_mask,
                **sample_kwargs,
            )
        # tokens:        [B, L_img, D_token]
        # saved_noise:   [B, L_img, D_token]
        # old_logprobs:  [B, L_img]

        images = self.model.decode_image_tokens(tokens, image_size=spec.image_size)

        if rescale_to_unit:
            images_for_reward = (images + 1.0) * 0.5
            images_for_reward = images_for_reward.clamp(0.0, 1.0)
        else:
            images_for_reward = images

        peak_mem_mb = self.peak_memory_mb()
        metrics = GenerationMetrics(
            num_prompts=len(prompts),
            num_samples=len(sample_specs),
            num_steps=spec.image_token_num,
            micro_batches=1,
            peak_memory_mb=peak_mem_mb,
        )

        # No DiT trajectory and no denoising env for AR. We still build a
        # ``RolloutTrajectoryData`` so future replay-style helpers can
        # wedge into the same field if needed; both inner fields are None.
        rollout_trajectory_data = RolloutTrajectoryData(
            rollout_log_probs=old_logprobs,
            denoising_env=None,
            dit_trajectory=None,
        )

        # ``extra`` is the contract surface for the collector. Everything
        # the OutputBatch → RolloutBatch translation needs lives here.
        extra: dict[str, Any] = {
            "tokens": tokens,
            "saved_noise": saved_noise,
            "log_probs": old_logprobs,
            "images_for_reward": images_for_reward,
            "prompt_input_ids": prompt_ids,
            "prompt_attention_mask": prompt_mask,
            "uncond_input_ids": uncond_ids,
            "uncond_attention_mask": uncond_mask,
            "context": {
                "cfg_scale": cfg_scale,
                "num_flow_steps": num_flow_steps,
                "noise_level": noise_level,
                "image_token_num": spec.image_token_num,
                "image_size": spec.image_size,
                "rescale_to_unit": rescale_to_unit,
            },
        }

        return OutputBatch(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=prompts,
            sample_specs=sample_specs,
            output=images,
            rollout_trajectory_data=rollout_trajectory_data,
            extra=extra,
            metrics=metrics,
            peak_memory_mb=peak_mem_mb or 0.0,
        )

    def forward_chunk(
        self,
        request: GenerationRequest,
        chunk: MicroBatchPlan,
    ) -> NextStep1ARChunkResult:
        """Run one prompt-major AR chunk through the black-box sampling path."""

        self.validate_chunk(request, chunk)
        sampling = request.sampling
        spec: ARGenerationSpec = self.parse_spec(request)

        cfg_scale = float(sampling["cfg_scale"])
        num_flow_steps = int(sampling["num_flow_steps"])
        noise_level = float(sampling["noise_level"])
        rescale_to_unit = bool(sampling.get("rescale_to_unit", True))

        repeated_prompts = [chunk.prompt] * chunk.sample_count
        prompt_ids, prompt_mask = self._tokenize_prompts(
            repeated_prompts,
            max_text_length=spec.max_text_length,
        )
        uncond_ids, uncond_mask = self._tokenize_prompts(
            [""] * chunk.sample_count,
            max_text_length=spec.max_text_length,
        )
        pad_id = getattr(self.model.processor, "pad_token_id", None) or 0
        prompt_ids, prompt_mask, uncond_ids, uncond_mask = self.align_pair(
            prompt_ids,
            prompt_mask,
            uncond_ids,
            uncond_mask,
            pad_id=pad_id,
        )

        cond_embeds = self._embed(prompt_ids)
        uncond_embeds = self._embed(uncond_ids)

        generator: torch.Generator | None = None
        if spec.seed is not None:
            generator = torch.Generator(device=self.model.device)
            generator.manual_seed(spec.seed + self.chunk_seed_offset(request, chunk))

        sample_kwargs: dict[str, Any] = {
            "cfg_scale": cfg_scale,
            "num_flow_steps": num_flow_steps,
            "noise_level": noise_level,
            "image_token_num": spec.image_token_num,
        }
        if generator is not None:
            sample_kwargs["generator"] = generator

        # Distributed AR chunks stay at prompt/sample granularity. The
        # token-level scheduler remains executor-internal for direct execution.
        tokens, saved_noise, old_logprobs = self.model.sample_image_tokens(
            cond_embeds,
            uncond_embeds,
            prompt_mask,
            uncond_mask,
            **sample_kwargs,
        )

        images = self.model.decode_image_tokens(tokens, image_size=spec.image_size)
        if rescale_to_unit:
            images_for_reward = (images + 1.0) * 0.5
            images_for_reward = images_for_reward.clamp(0.0, 1.0)
        else:
            images_for_reward = images
        peak_mem_mb = self.peak_memory_mb()

        return NextStep1ARChunkResult(
            prompt_index=chunk.prompt_index,
            sample_start=chunk.sample_start,
            sample_count=chunk.sample_count,
            output=images,
            tokens=tokens,
            saved_noise=saved_noise,
            log_probs=old_logprobs,
            images_for_reward=images_for_reward,
            prompt_input_ids=prompt_ids,
            prompt_attention_mask=prompt_mask,
            uncond_input_ids=uncond_ids,
            uncond_attention_mask=uncond_mask,
            context={
                "cfg_scale": cfg_scale,
                "num_flow_steps": num_flow_steps,
                "noise_level": noise_level,
                "image_token_num": spec.image_token_num,
                "image_size": spec.image_size,
                "rescale_to_unit": rescale_to_unit,
            },
            peak_memory_mb=peak_mem_mb,
        )

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_specs: Sequence[GenerationSampleSpec],
        chunks: Sequence[NextStep1ARChunkResult],
    ) -> OutputBatch:
        return NextStep1ChunkGatherer().gather_chunks(request, sample_specs, chunks)

    # -- internals -----------------------------------------------------

    def _sample_with_ar_scheduler(
        self,
        *,
        request: GenerationRequest,
        sample_specs: list[GenerationSampleSpec],
        cond_embeds: torch.Tensor,
        uncond_embeds: torch.Tensor | None,
        prompt_mask: torch.Tensor,
        uncond_mask: torch.Tensor | None,
        image_token_num: int,
        sample_kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run NextStep sampling through the executor-internal AR scheduler."""
        required = ("init_ar_state", "step_ar", "finalize_ar_state")
        missing = [name for name in required if not hasattr(self.model, name)]
        if missing:
            raise TypeError(
                "use_ar_scheduler=True requires model step API methods: " + ", ".join(missing)
            )

        if cond_embeds.shape[0] != len(sample_specs):
            raise ValueError(
                "Scheduled AR expects one sample spec per embedded row: "
                f"{len(sample_specs)} specs for {cond_embeds.shape[0]} rows"
            )

        state = self.model.init_ar_state(
            cond_embeds,
            uncond_embeds,
            prompt_mask,
            uncond_mask,
            **sample_kwargs,
        )
        sequences = [
            ActiveSequence(
                request_id=request.request_id,
                sample_id=spec.sample_id,
                family=request.family,
                task=request.task,
                tokenizer_key="nextstep_1",
                dtype=str(cond_embeds.dtype),
                max_new_tokens=image_token_num,
                metadata={
                    **dict(spec.metadata),
                    "row_index": row_index,
                    "prompt_index": spec.prompt_index,
                    "sample_index": spec.sample_index,
                },
            )
            for row_index, spec in enumerate(sample_specs)
        ]
        scheduler = ARTokenScheduler(
            max_batch_size=max(
                1,
                int(request.sampling.get("ar_scheduler_batch_size", len(sequences))),
            )
        )
        scheduler.add_many(sequences)

        while True:
            batch = scheduler.pop_batch()
            if batch is None:
                break
            result = self.model.step_ar(
                state,
                batch.sequences,
                generator=sample_kwargs.get("generator"),
            )
            if "saved_noise" not in result.replay_extras:
                raise ValueError("NextStep step_ar must return replay_extras['saved_noise']")
            for sequence in batch.sequences:
                sequence.advance()
            scheduler.push_back_unfinished(batch)

        return self.model.finalize_ar_state(state)

    def _tokenize_prompts(
        self,
        prompts: list[str],
        *,
        max_text_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenise via the upstream NextStep tokenizer.

        Mirrors ``NextStep1Collector._tokenize_prompts`` exactly so the
        old direct path and the engine path produce bitwise-identical
        ``input_ids``/``attention_mask`` pairs.
        """
        tok = self.model.processor
        device = self.model.device

        enc = tok(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_text_length,
        )
        ids = enc["input_ids"]
        mask = enc["attention_mask"]
        if ids.shape[1] < max_text_length:
            pad_id = getattr(tok, "pad_token_id", None) or 0
            extra_len = max_text_length - ids.shape[1]
            ids = torch.cat(
                [ids, torch.full((ids.shape[0], extra_len), pad_id, dtype=ids.dtype)],
                dim=1,
            )
            mask = torch.cat(
                [mask, torch.zeros((mask.shape[0], extra_len), dtype=mask.dtype)],
                dim=1,
            )
        return ids.to(device), mask.to(device)

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        embed = self.model.language_model.get_input_embeddings()
        return embed(token_ids)


class NextStep1ChunkGatherer:
    """Pure driver-side gatherer for NextStep-1 AR chunk payloads."""

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_specs: Sequence[GenerationSampleSpec],
        chunks: Sequence[NextStep1ARChunkResult],
    ) -> OutputBatch:
        """Pack prompt/sample AR chunks back into the canonical OutputBatch."""

        ordered_ar_chunks = ordered_chunks(
            request,
            sample_specs,
            chunks,
            row_fields=(
                "output",
                "tokens",
                "saved_noise",
                "log_probs",
                "images_for_reward",
                "prompt_input_ids",
                "prompt_attention_mask",
                "uncond_input_ids",
                "uncond_attention_mask",
            ),
        )
        tokens = torch.cat([chunk.tokens for chunk in ordered_ar_chunks], dim=0)
        saved_noise = torch.cat(
            [chunk.saved_noise for chunk in ordered_ar_chunks],
            dim=0,
        )
        log_probs = torch.cat([chunk.log_probs for chunk in ordered_ar_chunks], dim=0)
        output = torch.cat([chunk.output for chunk in ordered_ar_chunks], dim=0)
        peak_mem_mb = max_peak_memory_mb(ordered_ar_chunks)
        rollout_trajectory_data = RolloutTrajectoryData(
            rollout_log_probs=log_probs,
            denoising_env=None,
            dit_trajectory=None,
        )
        metrics = GenerationMetrics(
            num_prompts=len(request.prompts),
            num_samples=len(sample_specs),
            num_steps=int(request.sampling["image_token_num"]),
            micro_batches=len(ordered_ar_chunks),
            peak_memory_mb=peak_mem_mb,
        )
        extra: dict[str, Any] = {
            "tokens": tokens,
            "saved_noise": saved_noise,
            "log_probs": log_probs,
            "images_for_reward": torch.cat(
                [chunk.images_for_reward for chunk in ordered_ar_chunks],
                dim=0,
            ),
            "prompt_input_ids": torch.cat(
                [chunk.prompt_input_ids for chunk in ordered_ar_chunks],
                dim=0,
            ),
            "prompt_attention_mask": torch.cat(
                [chunk.prompt_attention_mask for chunk in ordered_ar_chunks],
                dim=0,
            ),
            "uncond_input_ids": torch.cat(
                [chunk.uncond_input_ids for chunk in ordered_ar_chunks],
                dim=0,
            ),
            "uncond_attention_mask": torch.cat(
                [chunk.uncond_attention_mask for chunk in ordered_ar_chunks],
                dim=0,
            ),
            "context": dict(ordered_ar_chunks[0].context),
        }

        return OutputBatch(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=list(request.prompts),
            sample_specs=list(sample_specs),
            output=output,
            rollout_trajectory_data=rollout_trajectory_data,
            extra=extra,
            metrics=metrics,
            peak_memory_mb=peak_mem_mb or 0.0,
        )


__all__ = [
    "NextStep1ARChunkResult",
    "NextStep1ChunkGatherer",
    "NextStep1PipelineExecutor",
]
