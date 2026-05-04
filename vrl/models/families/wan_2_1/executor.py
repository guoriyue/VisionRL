"""Wan 2.1 diffusion pipeline executor."""

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


class Wan_2_1PipelineExecutor(DiffusionPipelineExecutorBase):
    """Diffusion executor for Wan 2.1 text-to-video rollouts."""

    family: str = "wan_2_1"
    task: str = "t2v"
    default_num_frames: int = 1
    default_max_sequence_length: int = 512

    def __init__(
        self,
        model: Any,  # Wan_2_1Policy
        *,
        sample_batch_size: int = 1,
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
        """Repeat Wan text embeds across the chunk batch."""

        del generation_request, video_request, spec
        chunk_g = chunk.sample_count
        chunk_encoded: dict[str, Any] = {
            "prompt_embeds": repeat_tensor_batch(
                encoded["prompt_embeds"],
                chunk_g,
            ),
        }
        neg = encoded.get("negative_prompt_embeds")
        if neg is not None:
            chunk_encoded["negative_prompt_embeds"] = repeat_tensor_batch(
                neg,
                chunk_g,
            )
        return chunk_encoded


__all__ = ["Wan_2_1PipelineExecutor"]
