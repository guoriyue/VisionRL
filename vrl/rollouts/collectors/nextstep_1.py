"""NextStep-1 rollout collector for token-level GRPO.

Mirrors ``vrl.rollouts.collectors.janus_pro.JanusProCollector`` but for
continuous-token AR generation:

    Collector  →  ContinuousTokenLogProbEvaluator  →  TokenGRPO  →  OnlineTrainer

Per-call lifecycle:

  1. Tokenise N prompts (cond + uncond) with the upstream NextStep tokenizer.
  2. Sample ``image_token_num`` continuous tokens autoregressively under CFG,
     capturing the per-token Gaussian log-probability — these become the
     GRPO ``old_log_prob``.
  3. Stash the per-token flow-prior noise so training-time replay is
     deterministic (modulo LoRA-induced velocity drift, which is exactly
     what the GRPO ratio is supposed to capture).
  4. Decode tokens → pixels via the f8ch16 VAE; score with the reward fn.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vrl.rollouts.types import ExperienceBatch

if TYPE_CHECKING:  # pragma: no cover
    from vrl.models.families.nextstep_1.policy import NextStep1Policy
    from vrl.rewards.base import RewardFunction

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NextStep1CollectorConfig:
    """Configuration for ``NextStep1Collector``."""

    n_samples_per_prompt: int = 4
    cfg_scale: float = 4.5
    num_flow_steps: int = 20
    noise_level: float = 1.0
    image_token_num: int = 1024     # 32 × 32 patches per 256² image
    image_size: int = 256
    rescale_to_unit: bool = True    # convert [-1, 1] → [0, 1] for the reward layer
    max_text_length: int = 256


class NextStep1Collector:
    """Collect on-policy rollouts from a ``NextStep1Policy`` wrapper."""

    def __init__(
        self,
        model: "NextStep1Policy",
        reward_fn: "RewardFunction | None" = None,
        config: NextStep1CollectorConfig | None = None,
    ) -> None:
        self.model = model
        self.reward_fn = reward_fn
        self.config = config or NextStep1CollectorConfig()

    # ------------------------------------------------------------------
    # Public: rollout
    # ------------------------------------------------------------------

    async def collect(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> ExperienceBatch:
        cfg = self.config
        device = self.model.device

        n_per = int(kwargs.get("group_size") or cfg.n_samples_per_prompt)

        repeated_prompts = [p for p in prompts for _ in range(n_per)]
        group_ids = torch.arange(len(prompts), device=device).repeat_interleave(n_per)

        prompt_ids, prompt_mask = self._tokenize_prompts(repeated_prompts)
        uncond_ids, uncond_mask = self._tokenize_prompts([""] * len(repeated_prompts))
        pad_id = getattr(self.model.processor, "pad_token_id", None) or 0
        prompt_ids, prompt_mask, uncond_ids, uncond_mask = self._align_pair(
            prompt_ids, prompt_mask, uncond_ids, uncond_mask, pad_id=pad_id,
        )

        cond_embeds = self._embed(prompt_ids)
        uncond_embeds = self._embed(uncond_ids)

        tokens, saved_noise, old_logprobs = self.model.sample_image_tokens(
            cond_embeds, uncond_embeds, prompt_mask, uncond_mask,
            cfg_scale=cfg.cfg_scale,
            num_flow_steps=cfg.num_flow_steps,
            noise_level=cfg.noise_level,
            image_token_num=cfg.image_token_num,
        )
        # tokens:       [B, L_img, D_token]
        # saved_noise:  [B, L_img, D_token]
        # old_logprobs: [B, L_img]

        images = self.model.decode_image_tokens(tokens, image_size=cfg.image_size)

        if cfg.rescale_to_unit:
            images_for_reward = (images + 1.0) * 0.5
            images_for_reward = images_for_reward.clamp(0.0, 1.0)
        else:
            images_for_reward = images

        # Forward PromptExample-level metadata for OCR / reference rewards
        rollout_metadata: dict[str, Any] = {}
        target_text = kwargs.get("target_text")
        if target_text:
            rollout_metadata["target_text"] = target_text
        references = kwargs.get("references")
        if references:
            rollout_metadata["references"] = references
        sample_md = kwargs.get("sample_metadata")
        if sample_md:
            rollout_metadata.update(sample_md)

        rewards = await self._score(
            images_for_reward, repeated_prompts, rollout_metadata,
        )

        # Per-token mask: every continuous image-token position counts.
        token_mask = torch.ones_like(old_logprobs)

        # OnlineTrainer CEA convention: observations / log_probs carry a
        # singleton time dim so trainer's ``[:, j]`` indexing yields the
        # correct per-prompt slice.
        observations = prompt_ids.unsqueeze(1)                 # [B, 1, L_text]
        log_probs_3d = old_logprobs.detach().unsqueeze(1)      # [B, 1, L_img]

        return ExperienceBatch(
            observations=observations,           # [B, 1, L_text]
            actions=tokens,                      # continuous tokens [B, L_img, D_token]
            rewards=rewards,                     # [B]
            dones=torch.ones(len(repeated_prompts), dtype=torch.bool, device=device),
            group_ids=group_ids,                 # [B]
            extras={
                "log_probs": log_probs_3d,                     # [B, 1, L_img]
                "prompt_attention_mask": prompt_mask,          # [B, L_text]
                "uncond_input_ids": uncond_ids,                # [B, L_text]
                "uncond_attention_mask": uncond_mask,          # [B, L_text]
                "token_mask": token_mask,                      # [B, L_img]
                "saved_noise": saved_noise,                    # [B, L_img, D_token]
            },
            context={
                "cfg_scale": cfg.cfg_scale,
                "num_flow_steps": cfg.num_flow_steps,
                "noise_level": cfg.noise_level,
                "image_token_num": cfg.image_token_num,
            },
            videos=images.unsqueeze(2),          # [B, 3, 1, H, W] — reward layer expects T dim
            prompts=repeated_prompts,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize_prompts(
        self, prompts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenise via the upstream NextStep tokenizer.

        TODO(nextstep-binding): NextStep-1 uses a Qwen-derived tokenizer.
        The exact T2I prompt template (chat-style? plain? begin_of_image
        sentinel?) lives in upstream ``inference/gen_pipeline.py``. For
        the scaffold we use plain encoding.
        """
        tok = self.model.processor
        device = self.model.device
        cap = self.config.max_text_length

        enc = tok(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=cap,
        )
        ids = enc["input_ids"]
        mask = enc["attention_mask"]
        if ids.shape[1] < cap:
            pad_id = getattr(tok, "pad_token_id", None) or 0
            extra = cap - ids.shape[1]
            ids = torch.cat(
                [ids, torch.full((ids.shape[0], extra), pad_id, dtype=ids.dtype)], dim=1,
            )
            mask = torch.cat(
                [mask, torch.zeros((mask.shape[0], extra), dtype=mask.dtype)], dim=1,
            )
        return ids.to(device), mask.to(device)

    @staticmethod
    def _align_pair(
        a_ids: torch.Tensor, a_mask: torch.Tensor,
        b_ids: torch.Tensor, b_mask: torch.Tensor,
        pad_id: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        L = max(a_ids.shape[1], b_ids.shape[1])

        def _pad(ids: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            cur = ids.shape[1]
            if cur == L:
                return ids, mask
            extra = L - cur
            pad_ids = torch.full(
                (ids.shape[0], extra), pad_id, dtype=ids.dtype, device=ids.device,
            )
            pad_mask = torch.zeros(
                (mask.shape[0], extra), dtype=mask.dtype, device=mask.device,
            )
            return torch.cat([ids, pad_ids], dim=1), torch.cat([mask, pad_mask], dim=1)

        a_ids, a_mask = _pad(a_ids, a_mask)
        b_ids, b_mask = _pad(b_ids, b_mask)
        return a_ids, a_mask, b_ids, b_mask

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._embed_with(self.model, token_ids)

    @staticmethod
    def _embed_with(model: "NextStep1Policy", token_ids: torch.Tensor) -> torch.Tensor:
        embed = model.language_model.get_input_embeddings()
        return embed(token_ids)

    async def _score(
        self,
        images: torch.Tensor,
        prompts: list[str],
        rollout_metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        from vrl.algorithms.types import Rollout, Trajectory

        device = self.model.device
        if self.reward_fn is None:
            return torch.zeros(images.shape[0], device=device)

        meta: dict[str, Any] = dict(rollout_metadata or {})

        rollouts = [
            Rollout(
                request=None,
                trajectory=Trajectory(
                    prompt=prompts[i], seed=0, steps=[], output=images[i],
                ),
                metadata=dict(meta),
            )
            for i in range(images.shape[0])
        ]

        batch_fn = getattr(self.reward_fn, "score_batch", None)
        if batch_fn is not None and inspect.iscoroutinefunction(batch_fn):
            raw = await batch_fn(rollouts)
        else:
            raw = [await self.reward_fn.score(r) for r in rollouts]

        return torch.tensor([float(s) for s in raw], device=device, dtype=torch.float32)
