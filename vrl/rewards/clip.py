"""CLIP text-image similarity reward.

Ported from flow_grpo/clip_scorer.py.  Wraps the CLIP scorer as a
RewardFunction subclass.
"""

from __future__ import annotations

from typing import Any

from vrl.algorithms.types import Rollout
from vrl.rewards.base import RewardFunction


class CLIPScoreReward(RewardFunction):
    """CLIP text-image cosine similarity / 30 (normalised to ~[0, 1]).

    Requires ``transformers`` and ``torchvision``.
    Loaded lazily on first call.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "openai/clip-vit-large-patch14",
    ) -> None:
        self._device = device
        self._model_name = model_name
        self._scorer: Any = None

    def _ensure_loaded(self) -> None:
        if self._scorer is not None:
            return
        import torch
        import torch.nn as nn
        import torchvision.transforms as T
        from transformers import CLIPModel, CLIPProcessor, AutoImageProcessor

        def _get_size(size: Any) -> tuple[int, int] | int:
            if isinstance(size, int):
                return (size, size)
            if "height" in size and "width" in size:
                return (size["height"], size["width"])
            if "shortest_edge" in size:
                return size["shortest_edge"]
            raise ValueError(f"Invalid size: {size}")

        def _get_transform(processor: AutoImageProcessor) -> T.Compose:
            cfg = processor.to_dict()
            resize = T.Resize(_get_size(cfg.get("size"))) if cfg.get("do_resize") else nn.Identity()
            crop = T.CenterCrop(_get_size(cfg.get("crop_size"))) if cfg.get("do_center_crop") else nn.Identity()
            normalise = T.Normalize(mean=processor.image_mean, std=processor.image_std) if cfg.get("do_normalize") else nn.Identity()
            return T.Compose([resize, crop, normalise])

        model_name = self._model_name

        class _ClipScorer(nn.Module):
            def __init__(self, device: str) -> None:
                super().__init__()
                self.device = device
                self.model = CLIPModel.from_pretrained(model_name).to(device)
                self.processor = CLIPProcessor.from_pretrained(model_name)
                self.tform = _get_transform(self.processor.image_processor)
                self.eval()

            @torch.no_grad()
            def __call__(self, pixels: torch.Tensor, prompts: list[str]) -> torch.Tensor:
                texts = self.processor(text=prompts, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
                pixels = self.tform(pixels.to(dtype=pixels.dtype)).to(self.device)
                outputs = self.model(pixel_values=pixels, **texts)
                return outputs.logits_per_image.diagonal() / 30

        self._scorer = _ClipScorer(self._device)

    async def score(self, rollout: Rollout) -> float:
        self._ensure_loaded()
        import torch
        import numpy as np

        output = rollout.trajectory.output
        prompt = rollout.trajectory.prompt

        if isinstance(output, torch.Tensor):
            pixels = output
            if pixels.ndim == 5:
                pixels = pixels[:, pixels.shape[1] // 2]  # middle frame
            if pixels.max() <= 1.0:
                pass  # already [0,1]
            else:
                pixels = pixels.float() / 255.0
        elif isinstance(output, np.ndarray):
            pixels = torch.from_numpy(output.transpose(0, 3, 1, 2)).float() / 255.0
        else:
            return 0.0

        if pixels.ndim == 3:
            pixels = pixels.unsqueeze(0)

        scores = self._scorer(pixels, [prompt] * pixels.shape[0])
        return float(scores.mean().item())

    async def score_batch(self, rollouts: list[Rollout]) -> list[float]:
        return [await self.score(r) for r in rollouts]
