"""Aesthetic reward function using CLIP + MLP predictor.

Ported from flow_grpo/aesthetic_scorer.py.  Wraps the aesthetic scorer
as a RewardFunction subclass for use with the training loop.
"""

from __future__ import annotations

from typing import Any

from vrl.algorithms.types import Rollout
from vrl.rewards.base import RewardFunction


class AestheticReward(RewardFunction):
    """Aesthetic score reward using CLIP ViT-L/14 + MLP head.

    Requires ``transformers`` and the aesthetic predictor weights.
    The scorer is loaded lazily on first call.
    """

    def __init__(self, device: str = "cuda", dtype: str = "float32") -> None:
        self._device = device
        self._dtype_str = dtype
        self._scorer: Any = None

    def _ensure_loaded(self) -> None:
        if self._scorer is not None:
            return
        import torch
        import torch.nn as nn
        from transformers import CLIPModel, CLIPProcessor

        dtype = getattr(torch, self._dtype_str, torch.float32)

        class _MLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(768, 1024),
                    nn.Dropout(0.2),
                    nn.Linear(1024, 128),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.Dropout(0.1),
                    nn.Linear(64, 16),
                    nn.Linear(16, 1),
                )

            @torch.no_grad()
            def forward(self, embed: torch.Tensor) -> torch.Tensor:
                return self.layers(embed)

        class _AestheticScorer(nn.Module):
            def __init__(self, device: str, dtype: torch.dtype) -> None:
                super().__init__()
                self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
                self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
                self.mlp = _MLP()
                self.dtype = dtype
                self.eval()

            @torch.no_grad()
            def __call__(self, images: Any) -> torch.Tensor:
                device = next(self.parameters()).device
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
                embed = self.clip.get_image_features(**inputs)
                embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
                return self.mlp(embed).squeeze(1)

        self._scorer = _AestheticScorer(self._device, dtype).to(self._device)

    async def score(self, rollout: Rollout) -> float:
        self._ensure_loaded()
        import torch

        output = rollout.trajectory.output
        if isinstance(output, torch.Tensor):
            images = (output * 255).round().clamp(0, 255).to(torch.uint8)
            if images.ndim == 4:
                images = images.cpu().numpy().transpose(0, 2, 3, 1)
            else:
                # Video: take middle frame
                mid = images.shape[0] // 2
                images = images[mid].cpu().numpy().transpose(1, 2, 0)
                images = [images]
        else:
            images = [output]

        scores = self._scorer(images)
        return float(scores.mean().item())

    async def score_batch(self, rollouts: list[Rollout]) -> list[float]:
        return [await self.score(r) for r in rollouts]
