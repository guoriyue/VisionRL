"""Wan 2.1 t2v pipeline executor.

Owns the diffusion denoise loop previously inlined in
``vrl.rollouts.collectors.wan_2_1.Wan_2_1Collector.collect``. The collector
keeps reward scoring, KL adjustment, and ``ExperienceBatch`` packing.

Boundary:

- This module MUST NOT import ``vrl.rollouts.*`` or ``ExperienceBatch``.
- This module MUST NOT compute reward.
- Inputs come from ``GenerationRequest.sampling`` (collector packs them).
- Outputs are the canonical ``OutputBatch`` (with rollout trajectory data,
  decoded videos, replay-time embeds in ``RolloutDenoisingEnv``).

This is the video twin of ``SD3_5PipelineExecutor``: same SDE math, same
trajectory shape (``[B, T, ...]``), but the model produces 5D latents
``[B, C, T_v, H, W]`` and Wan's text encoder lacks a pooled CLIP embed.

Determinism contract: same prompts + same seed ⇒ same log_probs / latents /
timesteps / decoded video. Each micro-batch uses
``gen.manual_seed(seed + chunk_offset)``. ``same_latent=True`` requires an
explicit seed.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from vrl.engine.generation.batching import forward_batch_by_merging_prompts
from vrl.engine.generation.diffusion import (
    DiffusionChunkResult,
    DiffusionDenoiseConfig,
    repeat_tensor_batch,
    run_diffusion_denoise_chunk,
    select_sde_window,
)
from vrl.engine.generation.gather import gather_diffusion_chunks
from vrl.engine.generation.microbatching import (
    MicroBatchPlan,
    plan_prompt_group_microbatches,
    run_microbatches_with_oom_retry,
)
from vrl.engine.generation.protocols import (
    BatchedFamilyPipelineExecutor,
    ChunkedFamilyPipelineExecutor,
)
from vrl.engine.generation.types import (
    GenerationRequest,
    GenerationSampleSpec,
    OutputBatch,
    WorkloadSignature,
)

logger = logging.getLogger(__name__)


class Wan_2_1PipelineExecutor(
    ChunkedFamilyPipelineExecutor,
    BatchedFamilyPipelineExecutor,
):
    """Diffusion executor for Wan 2.1 text-to-video rollouts.

    The collector constructs a ``GenerationRequest`` whose ``sampling``
    dict holds:

    - ``num_steps``: int
    - ``guidance_scale``: float
    - ``height`` / ``width``: int
    - ``num_frames``: int
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

    family: str = "wan_2_1"
    task: str = "t2v"

    def __init__(
        self,
        model: Any,  # Wan_2_1Policy (WanT2VDiffusersPolicy or WanT2VOfficialPolicy)
        *,
        sample_batch_size: int = 1,
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
        num_frames = int(sampling.get("num_frames", sampling.get("frame_count", 1)))
        noise_level = float(sampling.get("noise_level", 1.0))
        sde_window_size = int(sampling.get("sde_window_size", 0))
        sde_window_range = tuple(sampling.get("sde_window_range", (0, num_steps)))
        sde_window = select_sde_window(sde_window_size, sde_window_range)
        same_latent = bool(sampling.get("same_latent", False))
        max_sequence_length = int(sampling.get("max_sequence_length", 512))
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
            "frame_count": num_frames,
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
        num_frames: int,
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
            "frame_count": num_frames,
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
        # Build per-chunk encoded dict by repeating embeds chunk_g times.
        # Wan's text encoder returns prompt_embeds (no pooled CLIP), plus
        # an optional negative_prompt_embeds when CFG is active.
        chunk_encoded: dict[str, Any] = {
            "prompt_embeds": repeat_tensor_batch(encoded["prompt_embeds"], chunk_g),
        }
        neg = encoded.get("negative_prompt_embeds")
        if neg is not None:
            chunk_encoded["negative_prompt_embeds"] = repeat_tensor_batch(neg, chunk_g)

        return run_diffusion_denoise_chunk(
            policy=self.model,
            request=request,
            encoded=chunk_encoded,
            config=DiffusionDenoiseConfig(
                sample_start=chunk_offset,
                seed=seed,
                same_latent=same_latent,
                sde_window=sde_window,
                return_kl=return_kl,
                noise_level=noise_level,
            ),
        )


__all__ = ["Wan_2_1PipelineExecutor"]
