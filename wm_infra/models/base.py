"""Unified model contract for staged temporal generation.

All models — Wan, Cosmos, action-conditioned interactive video generators
(Matrix-Game-3), and future temporal models — implement the five-stage
contract: encode_text -> encode_conditioning -> denoise -> decode_vae -> postprocess.

Backend and serving concerns remain separate from this layer (no controlplane
schemas, no WanExecutionContext, no workload/compute_profile lifecycle).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from wm_infra.schemas.video_generation import StageResult, VideoGenerationRequest


class VideoGenerationModel(ABC):
    """Unified abstract base class for all temporal generation models.

    This is THE model contract for wm-infra. The five-stage interface —
    ``encode_text`` / ``encode_conditioning`` / ``denoise`` / ``decode_vae`` /
    ``postprocess`` — covers Wan, Cosmos, and action-conditioned interactive
    video generators such as Matrix-Game-3.

    Action-conditioned models override ``encode_conditioning`` to incorporate
    per-frame action inputs via the ``action_sequence`` /
    ``action_conditioning_mode`` fields on ``VideoGenerationRequest``.

    The model owns the forward logic but **not** serving infra (workload
    lifecycle, CUDA graph capture, compute profiles, queue scheduling).
    """

    model_family: str = "video_generation"

    @abstractmethod
    async def load(self) -> None:
        """Load / resolve model resources (checkpoints, modules, etc.)."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable model metadata dict."""

    @abstractmethod
    async def encode_text(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> StageResult:
        """Produce prompt or text embeddings for the next stage."""

    async def encode_conditioning(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> StageResult:
        """Produce conditioning tensors from references, actions, or metadata.

        When ``request.action_sequence`` is set and
        ``request.action_conditioning_mode != "none"``, subclasses should
        encode the action inputs into conditioning tensors appropriate for
        their architecture (concatenation, cross-attention, adaptive layer
        norm, etc.).
        """
        return StageResult(notes=["No conditioning inputs were provided."])

    @abstractmethod
    async def denoise(self, request: VideoGenerationRequest, state: dict[str, Any]) -> StageResult:
        """Run the main generation step, usually diffusion or a related sampler."""

    @abstractmethod
    async def decode_vae(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> StageResult:
        """Decode latent samples into frames or other media outputs."""

    async def postprocess(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> StageResult:
        """Assemble decoded outputs into delivery-ready artifacts."""
        return StageResult(notes=["Postprocess stage is a passthrough."])
