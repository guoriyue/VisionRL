"""SD3.5-Medium diffusion pipeline executor."""

from __future__ import annotations

from typing import Any

from vrl.engine.core.types import GenerationRequest
from vrl.engine.diffusion import (
    DiffusionGenerationSpec,
    DiffusionPipelineExecutorBase,
    repeat_tensor_batch,
)
from vrl.engine.microbatching import MicroBatchPlan
from vrl.models.diffusion import VideoGenerationRequest


class SD3_5PipelineExecutor(DiffusionPipelineExecutorBase):
    """Diffusion executor for SD3.5-M text-to-image rollouts."""

    family: str = "sd3_5"
    task: str = "t2i"
    default_num_frames: int = 1
    default_max_sequence_length: int = 128

    def __init__(
        self,
        model: Any,  # SD3_5Policy
        *,
        sample_batch_size: int = 8,
    ) -> None:
        self.model = model
        self.default_sample_batch_size = max(1, int(sample_batch_size))

    def build_chunk_encoded(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        spec: DiffusionGenerationSpec,
        chunk: MicroBatchPlan,
    ) -> dict[str, Any]:
        """Repeat SD3 prompt and pooled embeds across the chunk batch."""

        del generation_request, video_request, spec
        chunk_g = chunk.sample_count
        chunk_encoded: dict[str, Any] = {
            "prompt_embeds": repeat_tensor_batch(
                encoded["prompt_embeds"],
                chunk_g,
            ),
            "pooled_prompt_embeds": repeat_tensor_batch(
                encoded["pooled_prompt_embeds"],
                chunk_g,
            ),
        }
        neg = encoded.get("negative_prompt_embeds")
        neg_pool = encoded.get("negative_pooled_prompt_embeds")
        if neg is not None:
            chunk_encoded["negative_prompt_embeds"] = repeat_tensor_batch(
                neg,
                chunk_g,
            )
        if neg_pool is not None:
            chunk_encoded["negative_pooled_prompt_embeds"] = repeat_tensor_batch(
                neg_pool,
                chunk_g,
            )
        return chunk_encoded


__all__ = ["SD3_5PipelineExecutor"]
