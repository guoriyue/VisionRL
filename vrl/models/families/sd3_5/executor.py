"""SD3.5-Medium pipeline executor.

Owns the diffusion denoise loop previously inlined in
``vrl.rollouts.collectors.sd3_5.SD3_5Collector.collect``. The collector
keeps reward scoring, KL adjustment, and ``ExperienceBatch`` packing.

Boundary:

- This module MUST NOT import ``vrl.rollouts.*`` or ``ExperienceBatch``.
- This module MUST NOT compute reward.
- Inputs come from ``GenerationRequest.sampling`` (collector packs them).
- Outputs are the canonical ``OutputBatch`` (with rollout trajectory data,
  decoded images, replay-time embeds in ``RolloutDenoisingEnv``).

Parity contract: same prompts + same seed + same SDE window ⇒ same
log_probs/latents/timesteps/decoded images as the pre-migration
collector path. The seed-derivation rule mirrors the old code exactly:
``gen.manual_seed(seed + chunk_offset)`` per micro-batch.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from vrl.engine.generation.gather import gather_diffusion_chunks
from vrl.engine.generation.types import (
    GenerationRequest,
    GenerationSampleSpec,
    OutputBatch,
    WorkloadSignature,
)
from vrl.executors.base import (
    BatchedFamilyPipelineExecutor,
    ChunkedFamilyPipelineExecutor,
)
from vrl.executors.batching import forward_batch_by_merging_prompts
from vrl.executors.diffusion import (
    DiffusionChunkResult,
    DiffusionDenoiseConfig,
    repeat_tensor_batch,
    run_diffusion_denoise_chunk,
    select_sde_window,
)
from vrl.executors.microbatching import (
    MicroBatchPlan,
    plan_prompt_group_microbatches,
    run_microbatches_with_oom_retry,
)

logger = logging.getLogger(__name__)


class SD3_5PipelineExecutor(
    ChunkedFamilyPipelineExecutor,
    BatchedFamilyPipelineExecutor,
):
    """Diffusion executor for SD3.5-M text-to-image rollouts.

    The collector constructs a ``GenerationRequest`` whose ``sampling``
    dict holds:

    - ``num_steps``: int
    - ``guidance_scale``: float
    - ``height`` / ``width``: int
    - ``noise_level``: float (Flow-GRPO Eq.9 ``a``)
    - ``cfg``: bool
    - ``sample_batch_size``: int (rollout-side micro-batching)
    - ``sde_window_size``: int (0 = always inject noise)
    - ``sde_window_range``: list[int] (length-2 [lo, hi])
    - ``same_latent``: bool
    - ``max_sequence_length``: int
    - ``seed``: int | None
    - ``negative_prompt``: str | None
    - ``return_kl``: bool — controls whether ``log_prob.abs()`` is
      additionally tracked as a per-step KL surrogate.

    The executor returns an ``OutputBatch`` whose
    ``rollout_trajectory_data.denoising_env`` carries every field the
    evaluator's ``DiffusionPolicy.restore_eval_state`` will read.
    """

    family: str = "sd3_5"
    task: str = "t2i"

    def __init__(
        self,
        model: Any,  # SD3_5Policy
        *,
        sample_batch_size: int = 8,
    ) -> None:
        self.model = model
        self.default_sample_batch_size = max(1, int(sample_batch_size))

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
        sample_batch_size = max(
            1,
            int(sampling.get("sample_batch_size", self.default_sample_batch_size)),
        )

        plan = plan_prompt_group_microbatches(
            prompts,
            samples_per_prompt=samples_per_prompt,
            max_samples_per_microbatch=sample_batch_size,
        )
        chunks = run_microbatches_with_oom_retry(
            plan.micro_batches,
            lambda micro_batch: self.forward_chunk(request, micro_batch),
        )
        return self.gather_chunks(request, sample_specs, chunks)

    def forward_chunk(
        self,
        request: GenerationRequest,
        chunk: MicroBatchPlan,
    ) -> DiffusionChunkResult:
        sampling = request.sampling
        num_steps = int(sampling["num_steps"])
        guidance_scale = float(sampling["guidance_scale"])
        height = int(sampling["height"])
        width = int(sampling["width"])
        noise_level = float(sampling.get("noise_level", 1.0))
        sde_window_size = int(sampling.get("sde_window_size", 0))
        sde_window_range = tuple(sampling.get("sde_window_range", (0, num_steps)))
        sde_window = select_sde_window(sde_window_size, sde_window_range)
        same_latent = bool(sampling.get("same_latent", False))
        max_sequence_length = int(sampling.get("max_sequence_length", 256))
        seed = sampling.get("seed")
        negative_prompt = sampling.get("negative_prompt")
        return_kl = bool(sampling.get("return_kl", False))

        from vrl.models.diffusion import VideoGenerationRequest

        req_kwargs: dict[str, Any] = {
            "prompt": chunk.prompt,
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "frame_count": 1,
            "extra": {"max_sequence_length": max_sequence_length},
        }
        if negative_prompt is not None:
            req_kwargs["negative_prompt"] = negative_prompt
        if seed is not None:
            req_kwargs["seed"] = seed
        video_request = VideoGenerationRequest(**req_kwargs)

        encoded = self.model.encode_prompt(
            chunk.prompt,
            video_request.negative_prompt or None,
            max_sequence_length=max_sequence_length,
            guidance_scale=guidance_scale,
        )
        return self._run_chunk(
            prompt=chunk.prompt,
            request=video_request,
            encoded=encoded,
            chunk_g=chunk.sample_count,
            chunk_offset=chunk.sample_start,
            seed=seed,
            same_latent=same_latent,
            sde_window=sde_window,
            noise_level=noise_level,
            return_kl=return_kl,
        )

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_specs: Sequence[GenerationSampleSpec],
        chunks: Sequence[DiffusionChunkResult],
    ) -> OutputBatch:
        return gather_diffusion_chunks(
            request,
            sample_specs,
            chunks,
            model_family=self.family,
        )

    def forward_batch(
        self,
        requests: list[GenerationRequest],
        sample_specs_by_request: dict[str, list[GenerationSampleSpec]],
    ) -> dict[str, OutputBatch]:
        return forward_batch_by_merging_prompts(
            self,
            requests,
            sample_specs_by_request,
        )

    # -- internals -----------------------------------------------------

    def _run_prompt(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        samples_per_prompt: int,
        num_steps: int,
        guidance_scale: float,
        height: int,
        width: int,
        noise_level: float,
        sample_batch_size: int,
        sde_window: tuple[int, int] | None,
        same_latent: bool,
        max_sequence_length: int,
        seed: int | None,
        return_kl: bool,
    ) -> list[DiffusionChunkResult]:
        """Run all micro-batch chunks for one prompt; return per-chunk dicts."""
        from vrl.models.diffusion import VideoGenerationRequest

        req_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "frame_count": 1,  # SD3 is image-only
            "extra": {"max_sequence_length": max_sequence_length},
        }
        if negative_prompt is not None:
            req_kwargs["negative_prompt"] = negative_prompt
        if seed is not None:
            req_kwargs["seed"] = seed
        request = VideoGenerationRequest(**req_kwargs)

        # 1. Encode prompt once (re-used across micro-batches)
        encoded = self.model.encode_prompt(
            prompt,
            request.negative_prompt or None,
            max_sequence_length=max_sequence_length,
            guidance_scale=guidance_scale,
        )

        plan = plan_prompt_group_microbatches(
            [prompt],
            samples_per_prompt=samples_per_prompt,
            max_samples_per_microbatch=sample_batch_size,
        )

        def _run(micro_batch: MicroBatchPlan) -> DiffusionChunkResult:
            return self._run_chunk(
                prompt=prompt,
                request=request,
                encoded=encoded,
                chunk_g=micro_batch.sample_count,
                chunk_offset=micro_batch.sample_start,
                seed=seed,
                same_latent=same_latent,
                sde_window=sde_window,
                noise_level=noise_level,
                return_kl=return_kl,
            )

        return run_microbatches_with_oom_retry(plan.micro_batches, _run)

    def _run_chunk(
        self,
        *,
        prompt: str,
        request: Any,  # VideoGenerationRequest
        encoded: dict[str, Any],
        chunk_g: int,
        chunk_offset: int,
        seed: int | None,
        same_latent: bool,
        sde_window: tuple[int, int] | None,
        noise_level: float,
        return_kl: bool,
    ) -> DiffusionChunkResult:
        """Run one micro-batch chunk: prepare → denoise loop → decode."""
        # Build per-chunk encoded dict by repeating embeds chunk_g times
        chunk_encoded: dict[str, Any] = {
            "prompt_embeds": repeat_tensor_batch(encoded["prompt_embeds"], chunk_g),
            "pooled_prompt_embeds": repeat_tensor_batch(
                encoded["pooled_prompt_embeds"],
                chunk_g,
            ),
        }
        neg = encoded.get("negative_prompt_embeds")
        neg_pool = encoded.get("negative_pooled_prompt_embeds")
        if neg is not None:
            chunk_encoded["negative_prompt_embeds"] = repeat_tensor_batch(neg, chunk_g)
        if neg_pool is not None:
            chunk_encoded["negative_pooled_prompt_embeds"] = repeat_tensor_batch(
                neg_pool,
                chunk_g,
            )

        return run_diffusion_denoise_chunk(
            policy=self.model,
            request=request,
            encoded=chunk_encoded,
            config=DiffusionDenoiseConfig(
                prompt=prompt,
                sample_start=chunk_offset,
                seed=seed,
                same_latent=same_latent,
                sde_window=sde_window,
                return_kl=return_kl,
                noise_level=noise_level,
                sde_type="cps",
            ),
        )


__all__ = ["SD3_5PipelineExecutor"]
