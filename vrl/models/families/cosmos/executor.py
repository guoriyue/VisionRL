"""Cosmos Predict2 Video2World diffusion pipeline executor."""

from __future__ import annotations

from typing import Any

from vrl.engine.generation.diffusion import (
    DiffusionGenerationSpec,
    DiffusionPipelineExecutorBase,
    repeat_tensor_batch,
)
from vrl.engine.generation.microbatching import MicroBatchPlan
from vrl.engine.generation.types import GenerationRequest
from vrl.models.diffusion import VideoGenerationRequest


class CosmosPipelineExecutor(DiffusionPipelineExecutorBase):
    """Diffusion executor for Cosmos Predict2 Video2World rollouts."""

    family: str = "cosmos"
    task: str = "v2w"
    default_num_frames: int = 93
    default_fps: int | None = 16
    default_max_sequence_length: int = 512
    respect_cfg_flag: bool = False
    include_max_sequence_length_extra: bool = False

    def __init__(
        self,
        model: Any,  # CosmosPredict2Policy
        *,
        reference_image: Any = None,
        sample_batch_size: int = 8,
    ) -> None:
        self.model = model
        self.reference_image = _load_reference_image(reference_image)
        self.default_sample_batch_size = max(1, int(sample_batch_size))

    def encode_prompt_for_chunk(
        self,
        *,
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        spec: DiffusionGenerationSpec,
        chunk: MicroBatchPlan,
    ) -> dict[str, Any]:
        """Encode Cosmos text and preserve the Video2World reference image."""

        reference_image = self._reference_image_for_request(generation_request)
        return self.model.encode_prompt(
            chunk.prompt,
            video_request.negative_prompt or None,
            max_sequence_length=spec.base.max_sequence_length,
            guidance_scale=spec.base.guidance_scale,
            reference_image=reference_image,
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
        """Repeat Cosmos text embeds and pass reference image through unchanged."""

        del video_request, spec
        chunk_g = chunk.sample_count
        reference_image = self._reference_image_for_request(generation_request)
        chunk_encoded: dict[str, Any] = {
            "prompt_embeds": repeat_tensor_batch(
                encoded["prompt_embeds"],
                chunk_g,
            ),
            "reference_image": encoded.get("reference_image", reference_image),
        }
        neg = encoded.get("negative_prompt_embeds")
        if neg is not None:
            chunk_encoded["negative_prompt_embeds"] = repeat_tensor_batch(
                neg,
                chunk_g,
            )
        else:
            chunk_encoded["negative_prompt_embeds"] = None
        return chunk_encoded

    def build_prepare_kwargs(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        spec: DiffusionGenerationSpec,
        chunk: MicroBatchPlan,
    ) -> dict[str, Any]:
        """Thread the active reference image into Cosmos prepare_sampling."""

        del encoded, video_request, spec, chunk
        return {
            "reference_image": self._reference_image_for_request(
                generation_request,
            ),
        }

    def _reference_image_for_request(self, request: GenerationRequest) -> Any:
        return request.metadata.get("reference_image", self.reference_image)


def _load_reference_image(reference_image: Any) -> Any:
    if not isinstance(reference_image, str) or not reference_image:
        return reference_image
    from PIL import Image

    return Image.open(reference_image).convert("RGB")


__all__ = ["CosmosPipelineExecutor"]
