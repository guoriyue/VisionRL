"""Shared test fixtures for engine serving tests."""

from __future__ import annotations

from typing import Any

import numpy as np

from wm_infra.engine.model_executor.pipeline import ComposedPipeline
from wm_infra.engine.model_executor.stages.base import PipelineStage
from wm_infra.models.base import VideoGenerationModel
from wm_infra.schemas.video_generation import StageResult, VideoGenerationRequest


class ModelMethodStage(PipelineStage):
    """Wraps a single VideoGenerationModel method as a PipelineStage."""

    def __init__(self, name: str, method) -> None:
        self.name = name
        self._method = method

    async def forward(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> StageResult:
        return await self._method(request, state)


def model_to_pipeline(model: VideoGenerationModel) -> ComposedPipeline:
    """Build a ComposedPipeline from a VideoGenerationModel's 5 stage methods."""
    stages = [
        ModelMethodStage("encode_text", model.encode_text),
        ModelMethodStage("encode_conditioning", model.encode_conditioning),
        ModelMethodStage("denoise", model.denoise),
        ModelMethodStage("decode_vae", model.decode_vae),
        ModelMethodStage("postprocess", model.postprocess),
    ]
    return ComposedPipeline(stages=stages)


class StubWanModel(VideoGenerationModel):
    """Deterministic in-memory stub for Wan model, no GPU required."""

    model_family = "wan-stub"

    async def load(self) -> None:
        pass

    def describe(self) -> dict[str, Any]:
        return {"name": "wan-stub", "family": self.model_family}

    async def encode_text(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> StageResult:
        return StageResult(
            state_updates={
                "prompt_embeds": f"encoded:{request.prompt[:20]}",
                "negative_prompt_embeds": f"encoded:{request.negative_prompt[:20]}",
            },
            outputs={"token_count": max(1, len(request.prompt.split()))},
        )

    async def encode_conditioning(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> StageResult:
        return StageResult(
            state_updates={"conditioning_ready": True},
            outputs={"reference_count": len(request.references)},
        )

    async def denoise(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> StageResult:
        seed = request.seed if request.seed is not None else 42
        rng = np.random.RandomState(seed)
        h, w = request.height, request.width
        frames = request.frame_count
        latents = rng.randn(frames, h // 8, w // 8, 4).astype(np.float32)
        return StageResult(
            state_updates={"latent_frames": latents, "seed": seed},
            outputs={"latent_shape": list(latents.shape), "num_steps": request.num_steps},
        )

    async def decode_vae(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> StageResult:
        latents = state["latent_frames"]
        # Stub: expand latent spatial dims by 8x, map to [0,1]
        frames = np.clip(
            np.repeat(np.repeat(latents[..., :3], 8, axis=1), 8, axis=2),
            0.0,
            1.0,
        )
        return StageResult(
            state_updates={"video_frames": frames},
            outputs={"decoded_shape": list(frames.shape)},
        )

    async def postprocess(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> StageResult:
        frames = np.asarray(state["video_frames"])
        if frames.dtype != np.uint8:
            frames = np.clip(frames * 255.0, 0.0, 255.0).astype(np.uint8)
        return StageResult(
            state_updates={
                "video_frames": frames,
                "_pipeline_output": frames,
            },
            outputs={"frame_count": int(frames.shape[0]), "fps": request.fps},
        )
