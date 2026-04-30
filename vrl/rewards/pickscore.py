"""PickScore reward function.

Ported from flow_grpo/pickscore_scorer.py.  Wraps the PickScore v1
model as a RewardFunction subclass.
"""

from __future__ import annotations

from typing import Any

from vrl.algorithms.types import Rollout
from vrl.rewards.base import RewardFunction


class PickScoreReward(RewardFunction):
    """PickScore v1 reward (CLIP ViT-H/14 fine-tuned for preference).

    Scores are normalised by /26 to roughly [0, 1].
    Requires ``transformers``.  Loaded lazily on first call.
    """

    def __init__(
        self,
        device: str = "cuda",
        dtype: str = "float32",
        processor_name: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        model_name: str = "yuvalkirstain/PickScore_v1",
    ) -> None:
        self._device = device
        self._dtype_str = dtype
        self._processor_name = processor_name
        self._model_name = model_name
        self._scorer: Any = None

    def _ensure_loaded(self) -> None:
        if self._scorer is not None:
            return
        import torch
        from transformers import CLIPProcessor, CLIPModel

        dtype = getattr(torch, self._dtype_str, torch.float32)

        processor_name = self._processor_name
        model_name = self._model_name

        class _PickScoreScorer(torch.nn.Module):
            def __init__(self, device: str, dtype: torch.dtype) -> None:
                super().__init__()
                self.device = device
                self.processor = CLIPProcessor.from_pretrained(processor_name)
                self.model = CLIPModel.from_pretrained(model_name).eval().to(device, dtype=dtype)

            @torch.no_grad()
            def __call__(self, prompts: list[str], images: list[Any]) -> torch.Tensor:
                image_inputs = self.processor(images=images, padding=True, truncation=True, max_length=77, return_tensors="pt")
                image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
                text_inputs = self.processor(text=prompts, padding=True, truncation=True, max_length=77, return_tensors="pt")
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                image_embs = self.model.get_image_features(**image_inputs)
                image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)
                text_embs = self.model.get_text_features(**text_inputs)
                text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
                logit_scale = self.model.logit_scale.exp()
                scores = logit_scale * (text_embs @ image_embs.T)
                return scores.diag() / 26

        self._scorer = _PickScoreScorer(self._device, dtype)

    async def score(self, rollout: Rollout) -> float:
        self._ensure_loaded()
        import torch
        from PIL import Image
        import numpy as np

        output = rollout.trajectory.output
        prompt = rollout.trajectory.prompt

        if isinstance(output, torch.Tensor):
            arr = (output * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            if arr.ndim == 4:
                arr = arr.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            elif arr.ndim == 5:
                mid = arr.shape[1] // 2
                arr = arr[:, mid].transpose(0, 2, 3, 1)
            images = [Image.fromarray(a) for a in arr]
        elif isinstance(output, np.ndarray):
            if output.ndim == 3:
                images = [Image.fromarray(output)]
            else:
                images = [Image.fromarray(a) for a in output]
        elif isinstance(output, Image.Image):
            images = [output]
        else:
            return 0.0

        scores = self._scorer([prompt] * len(images), images)
        return float(scores.mean().item())

    async def score_batch(self, rollouts: list[Rollout]) -> list[float]:
        return [await self.score(r) for r in rollouts]
