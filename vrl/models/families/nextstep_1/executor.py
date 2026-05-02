"""NextStep-1 pipeline executor.

Owns the continuous-token autoregressive sampling loop previously inlined
in ``vrl.rollouts.collectors.nextstep_1.NextStep1Collector.collect``. The
collector keeps reward scoring and ``ExperienceBatch`` packing.

Boundary:

- This module MUST NOT import ``vrl.rollouts.*`` or ``ExperienceBatch``.
- This module MUST NOT compute reward.
- Inputs come from ``GenerationRequest.sampling`` (collector packs them).
- Outputs are the canonical ``OutputBatch``. NextStep-1 is AR with
  continuous tokens + a flow-matching head, so there is no diffusion
  trajectory; instead, ``output`` is the decoded image and ``extra``
  carries the three replay-determinism artifacts:

      * ``tokens``      [B, L_img, D_token]  — sampled continuous tokens
                                              (used as ``ExperienceBatch.actions``)
      * ``saved_noise`` [B, L_img, D_token]  — per-token x_0 prior for the
                                              flow ODE; replay reads this so
                                              ``recompute_logprobs`` can
                                              re-run the same trajectory.
      * ``log_probs``   [B, L_img]           — Gaussian per-token log-prob
                                              from sampling time
                                              (i.e. ``old_log_prob``).

Parity contract: same prompts + same generator state ⇒ same
``tokens``/``saved_noise``/``log_probs`` as the pre-migration collector
path. The model's ``sample_image_tokens`` is a single black-box call,
so parity reduces to "we call it once with the same arguments".
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from vrl.engine.generation.types import (
    GenerationMetrics,
    GenerationRequest,
    GenerationSampleSpec,
    OutputBatch,
    RolloutTrajectoryData,
    WorkloadSignature,
)
from vrl.executors.base import FamilyPipelineExecutor
from vrl.executors.batching import forward_batch_by_merging_prompts

logger = logging.getLogger(__name__)


class NextStep1PipelineExecutor:
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
        prompts = list(request.prompts)
        samples_per_prompt = int(request.samples_per_prompt)

        cfg_scale = float(sampling["cfg_scale"])
        num_flow_steps = int(sampling["num_flow_steps"])
        noise_level = float(sampling["noise_level"])
        image_token_num = int(sampling["image_token_num"])
        image_size = int(sampling["image_size"])
        max_text_length = int(sampling.get("max_text_length", 256))
        rescale_to_unit = bool(sampling.get("rescale_to_unit", True))
        seed = sampling.get("seed")

        # Repeat each prompt ``samples_per_prompt`` times so the AR loop
        # sees a flat ``[B, ...]`` batch where ``B = num_prompts x G``.
        repeated_prompts = [p for p in prompts for _ in range(samples_per_prompt)]

        prompt_ids, prompt_mask = self._tokenize_prompts(
            repeated_prompts, max_text_length=max_text_length,
        )
        uncond_ids, uncond_mask = self._tokenize_prompts(
            [""] * len(repeated_prompts), max_text_length=max_text_length,
        )
        pad_id = getattr(self.model.processor, "pad_token_id", None) or 0
        prompt_ids, prompt_mask, uncond_ids, uncond_mask = self._align_pair(
            prompt_ids, prompt_mask, uncond_ids, uncond_mask, pad_id=pad_id,
        )

        cond_embeds = self._embed(prompt_ids)
        uncond_embeds = self._embed(uncond_ids)

        # Optional deterministic generator. ``sample_image_tokens`` accepts
        # a ``generator`` kwarg (see NextStep1Policy.sample_image_tokens).
        generator: torch.Generator | None = None
        if seed is not None:
            device = self.model.device
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))

        sample_kwargs: dict[str, Any] = {
            "cfg_scale": cfg_scale,
            "num_flow_steps": num_flow_steps,
            "noise_level": noise_level,
            "image_token_num": image_token_num,
        }
        if generator is not None:
            sample_kwargs["generator"] = generator

        tokens, saved_noise, old_logprobs = self.model.sample_image_tokens(
            cond_embeds, uncond_embeds, prompt_mask, uncond_mask,
            **sample_kwargs,
        )
        # tokens:        [B, L_img, D_token]
        # saved_noise:   [B, L_img, D_token]
        # old_logprobs:  [B, L_img]

        images = self.model.decode_image_tokens(tokens, image_size=image_size)

        if rescale_to_unit:
            images_for_reward = (images + 1.0) * 0.5
            images_for_reward = images_for_reward.clamp(0.0, 1.0)
        else:
            images_for_reward = images

        peak_mem_mb = _peak_memory_mb()
        metrics = GenerationMetrics(
            num_prompts=len(prompts),
            num_samples=len(sample_specs),
            num_steps=image_token_num,
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
        # the OutputBatch → ExperienceBatch translation needs lives here.
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
                "image_token_num": image_token_num,
                "image_size": image_size,
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

    def forward_batch(
        self,
        requests: list[GenerationRequest],
        sample_specs_by_request: dict[str, list[GenerationSampleSpec]],
    ) -> dict[str, OutputBatch]:
        return forward_batch_by_merging_prompts(
            self, requests, sample_specs_by_request,
        )

    # -- internals -----------------------------------------------------

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

    @staticmethod
    def _align_pair(
        a_ids: torch.Tensor,
        a_mask: torch.Tensor,
        b_ids: torch.Tensor,
        b_mask: torch.Tensor,
        pad_id: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        L = max(a_ids.shape[1], b_ids.shape[1])

        def _pad(
            ids: torch.Tensor, mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            cur = ids.shape[1]
            if cur == L:
                return ids, mask
            extra_len = L - cur
            pad_ids = torch.full(
                (ids.shape[0], extra_len), pad_id, dtype=ids.dtype, device=ids.device,
            )
            pad_mask = torch.zeros(
                (mask.shape[0], extra_len), dtype=mask.dtype, device=mask.device,
            )
            return (
                torch.cat([ids, pad_ids], dim=1),
                torch.cat([mask, pad_mask], dim=1),
            )

        a_ids, a_mask = _pad(a_ids, a_mask)
        b_ids, b_mask = _pad(b_ids, b_mask)
        return a_ids, a_mask, b_ids, b_mask

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        embed = self.model.language_model.get_input_embeddings()
        return embed(token_ids)


def _peak_memory_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    try:
        peak_bytes = torch.cuda.max_memory_allocated()
    except Exception:
        return None
    return peak_bytes / (1024 * 1024)


__all__ = ["NextStep1PipelineExecutor"]


# Confirm we satisfy the protocol at import time.
_executor_protocol: type[FamilyPipelineExecutor] = FamilyPipelineExecutor
