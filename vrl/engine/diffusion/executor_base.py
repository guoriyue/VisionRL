"""Shared executor scaffolding for diffusion generation families."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vrl.engine.batching import forward_batch_by_merging_prompts
from vrl.engine.core.protocols import (
    BatchedFamilyPipelineExecutor,
    ChunkedFamilyPipelineExecutor,
)
from vrl.engine.core.types import (
    GenerationRequest,
    GenerationSampleSpec,
    OutputBatch,
    WorkloadSignature,
)
from vrl.engine.diffusion.denoise import (
    DiffusionChunkResult,
    DiffusionDenoiseConfig,
    repeat_tensor_batch,
    run_diffusion_denoise_chunk,
    select_sde_window,
)
from vrl.engine.diffusion.spec import (
    BaseDiffusionGenerationSpec,
    DiffusionGenerationSpec,
    SDEDiffusionSpec,
)
from vrl.engine.gather import gather_diffusion_chunks
from vrl.engine.microbatching import (
    MicroBatchPlan,
    plan_prompt_group_microbatches,
    run_microbatches_with_oom_retry,
)
from vrl.models.diffusion import VideoGenerationRequest


class DiffusionPipelineExecutorBase(
    ChunkedFamilyPipelineExecutor,
    BatchedFamilyPipelineExecutor,
):
    """Common GenerationRequest -> diffusion OutputBatch execution path."""

    family: str
    task: str
    model: Any
    default_sample_batch_size: int = 1
    default_num_frames: int = 1
    default_fps: int | None = None
    default_max_sequence_length: int = 512
    respect_cfg_flag: bool = True
    sde_type: str = "sde"
    include_max_sequence_length_extra: bool = True

    # -- protocol ------------------------------------------------------

    def workload_signature(self, request: GenerationRequest) -> WorkloadSignature:
        return WorkloadSignature.from_request(request)

    def parse_spec(self, request: GenerationRequest) -> DiffusionGenerationSpec:
        """Parse shared diffusion sampling fields from GenerationRequest."""

        sampling = request.sampling
        num_steps = int(sampling["num_steps"])
        fps_value = sampling.get("fps", self.default_fps)
        seed = sampling.get("seed")
        base = BaseDiffusionGenerationSpec(
            num_steps=num_steps,
            guidance_scale=float(sampling["guidance_scale"]),
            height=int(sampling["height"]),
            width=int(sampling["width"]),
            num_frames=int(
                sampling.get(
                    "num_frames",
                    sampling.get("frame_count", self.default_num_frames),
                )
            ),
            fps=None if fps_value is None else int(fps_value),
            sample_batch_size=max(
                1,
                int(
                    sampling.get(
                        "sample_batch_size",
                        self.default_sample_batch_size,
                    )
                ),
            ),
            max_sequence_length=int(
                sampling.get(
                    "max_sequence_length",
                    self.default_max_sequence_length,
                )
            ),
            seed=None if seed is None else int(seed),
            negative_prompt=sampling.get("negative_prompt"),
        )
        sde = SDEDiffusionSpec(
            noise_level=float(sampling.get("noise_level", 1.0)),
            sde_type=_parse_sde_type(sampling.get("sde_type", self.sde_type)),
            sde_window_size=int(sampling.get("sde_window_size", 0)),
            sde_window_range=_parse_sde_window_range(
                sampling.get("sde_window_range", (0, num_steps)),
            ),
            same_latent=bool(sampling.get("same_latent", False)),
            return_kl=bool(sampling.get("return_kl", False)),
        )
        return DiffusionGenerationSpec(base=base, sde=sde)

    def build_video_request(
        self,
        prompt: str,
        spec: DiffusionGenerationSpec,
    ) -> VideoGenerationRequest:
        """Build the backend-agnostic model request for one prompt chunk."""

        base = spec.base
        req_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "num_steps": base.num_steps,
            "guidance_scale": base.guidance_scale,
            "height": base.height,
            "width": base.width,
            "frame_count": base.num_frames,
        }
        if base.fps is not None:
            req_kwargs["fps"] = base.fps
        if base.negative_prompt is not None:
            req_kwargs["negative_prompt"] = base.negative_prompt
        if base.seed is not None:
            req_kwargs["seed"] = base.seed

        extra = self.build_video_request_extra(spec)
        if extra:
            req_kwargs["extra"] = extra
        return VideoGenerationRequest(**req_kwargs)

    def build_video_request_extra(
        self,
        spec: DiffusionGenerationSpec,
    ) -> dict[str, Any]:
        """Return family-neutral request.extra payload."""

        if not self.include_max_sequence_length_extra:
            return {}
        return {"max_sequence_length": spec.base.max_sequence_length}

    def build_denoise_config(
        self,
        spec: DiffusionGenerationSpec,
        chunk: MicroBatchPlan,
    ) -> DiffusionDenoiseConfig:
        """Build the SDE denoise config for one micro-batch."""

        if spec.sde is None:
            raise NotImplementedError(
                f"{type(self).__name__} must override denoise for non-SDE diffusion",
            )
        sde_window = select_sde_window(
            spec.sde.sde_window_size,
            spec.sde.sde_window_range,
        )
        return DiffusionDenoiseConfig(
            sample_start=chunk.sample_start,
            seed=spec.base.seed,
            same_latent=spec.sde.same_latent,
            sde_window=sde_window,
            return_kl=spec.sde.return_kl,
            noise_level=spec.sde.noise_level,
            sde_type=spec.sde.sde_type,
        )

    def forward(
        self,
        request: GenerationRequest,
        sample_specs: list[GenerationSampleSpec],
    ) -> OutputBatch:
        spec = self.parse_spec(request)
        plan = plan_prompt_group_microbatches(
            list(request.prompts),
            samples_per_prompt=int(request.samples_per_prompt),
            max_samples_per_microbatch=spec.base.sample_batch_size,
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
        spec = self.parse_spec(request)
        video_request = self.build_video_request(chunk.prompt, spec)
        encoded = self.encode_prompt_for_chunk(
            generation_request=request,
            video_request=video_request,
            spec=spec,
            chunk=chunk,
        )
        chunk_encoded = self.build_chunk_encoded(
            encoded=encoded,
            generation_request=request,
            video_request=video_request,
            spec=spec,
            chunk=chunk,
        )
        return run_diffusion_denoise_chunk(
            policy=self.model,
            request=video_request,
            encoded=chunk_encoded,
            config=self.build_denoise_config(spec, chunk),
            prepare_kwargs=self.build_prepare_kwargs(
                encoded=encoded,
                generation_request=request,
                video_request=video_request,
                spec=spec,
                chunk=chunk,
            ),
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
            respect_cfg_flag=self.respect_cfg_flag,
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

    # -- family hooks --------------------------------------------------

    def encode_prompt_for_chunk(
        self,
        *,
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        spec: DiffusionGenerationSpec,
        chunk: MicroBatchPlan,
    ) -> dict[str, Any]:
        """Encode prompt conditioning for a single prompt chunk."""

        del generation_request
        return self.model.encode_prompt(
            chunk.prompt,
            video_request.negative_prompt or None,
            max_sequence_length=spec.base.max_sequence_length,
            guidance_scale=spec.base.guidance_scale,
            request=video_request,
        )

    def build_chunk_encoded(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        spec: DiffusionGenerationSpec,
        chunk: MicroBatchPlan,
    ) -> dict[str, Any]:
        """Build per-sample encoded tensors for one micro-batch."""

        del generation_request, video_request, spec
        return {
            key: repeat_tensor_batch(value, chunk.sample_count) for key, value in encoded.items()
        }

    def build_prepare_kwargs(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        spec: DiffusionGenerationSpec,
        chunk: MicroBatchPlan,
    ) -> dict[str, Any] | None:
        """Return additional family kwargs for policy.prepare_sampling."""

        del encoded, generation_request, video_request, spec, chunk
        return None


def _parse_sde_window_range(value: Any) -> tuple[int, int]:
    try:
        lo = int(value[0])
        hi = int(value[1])
    except (TypeError, IndexError, ValueError) as exc:
        raise ValueError(
            "sampling.sde_window_range must contain two integer values",
        ) from exc
    return lo, hi


def _parse_sde_type(value: Any) -> str:
    sde_type = str(value)
    if sde_type not in {"sde", "cps"}:
        raise ValueError("sampling.sde_type must be 'sde' or 'cps'")
    return sde_type


__all__ = ["DiffusionPipelineExecutorBase"]
