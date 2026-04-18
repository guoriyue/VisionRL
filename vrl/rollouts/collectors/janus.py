"""Janus-Pro rollout collector for token-level GRPO.

Pairs ``vrl.models.families.janus.JanusProT2I`` with the generic
``OnlineTrainer`` CEA pipeline:

    Collector  →  TokenLogProbEvaluator  →  TokenGRPO  →  OnlineTrainer

Per-call lifecycle:

  1. Tokenise N prompts via ``VLChatProcessor`` →
     conditional + unconditional ids+masks.
  2. Sample ``image_token_num`` tokens autoregressively under CFG,
     capturing the *guided* log-probability of each sampled token.
     This per-token tensor is the GRPO ``old_log_prob``.
  3. Decode tokens → pixels in ``[-1, 1]`` and convert to ``[0, 1]``
     for the reward layer (``vrl/rewards/multi.py``).
  4. Score → fill ``ExperienceBatch.rewards``.

Single-prompt → multiple samples is implemented by repeating the prompt
``n_samples_per_prompt`` times before tokenisation; the ``group_ids``
field carries the original prompt index so GRPO can normalise within
each group.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from vrl.rollouts.types import ExperienceBatch

if TYPE_CHECKING:  # pragma: no cover
    from vrl.models.families.janus.model import JanusProT2I
    from vrl.rewards.base import RewardFunction

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class JanusCollectorConfig:
    """Configuration for ``JanusCollector``."""

    n_samples_per_prompt: int = 8
    cfg_weight: float = 5.0
    temperature: float = 1.0
    image_token_num: int = 576
    image_size: int = 384
    # Hand to the reward layer in [0, 1] (PIL-style); set False to keep [-1, 1].
    rescale_to_unit: bool = True
    # Optional cap on per-rollout text length (truncates prompt encoding).
    max_text_length: int = 256


class JanusCollector:
    """Collect on-policy rollouts from a ``JanusProT2I`` wrapper.

    Implements the same ``Collector`` Protocol as ``WanDiffusersCollector``
    so ``OnlineTrainer`` can use it without code changes.
    """

    def __init__(
        self,
        model: "JanusProT2I",
        reward_fn: "RewardFunction | None" = None,
        config: JanusCollectorConfig | None = None,
    ) -> None:
        self.model = model
        self.reward_fn = reward_fn
        self.config = config or JanusCollectorConfig()

    # ------------------------------------------------------------------
    # Public: rollout
    # ------------------------------------------------------------------

    async def collect(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> ExperienceBatch:
        """Sample ``n_samples_per_prompt`` rollouts per prompt and score them.

        Synchronous under the hood — the ``async`` signature is purely to
        match the ``Collector`` protocol used by ``OnlineTrainer``.

        ``group_size`` kwarg (passed by ``OnlineTrainer._step_cea``) overrides
        ``n_samples_per_prompt`` for that call; this matches the
        ``WanDiffusersCollector`` contract so the two collectors are swap-ins
        for each other.
        """
        cfg = self.config
        device = self.model.device

        n_per = int(kwargs.get("group_size") or cfg.n_samples_per_prompt)

        repeated_prompts = [p for p in prompts for _ in range(n_per)]
        group_ids = torch.arange(len(prompts), device=device).repeat_interleave(n_per)

        prompt_ids, prompt_mask = self._tokenize_prompts(repeated_prompts)
        uncond_ids, uncond_mask = self._tokenize_prompts([""] * len(repeated_prompts))
        # Independent tokenisation can yield different padded lengths.
        # Right-pad the shorter side so cond/uncond can share a single
        # trunk forward in sample_image_tokens (which torch.cat-s along B).
        pad_id = getattr(self.model.processor.tokenizer, "pad_token_id", None) or 0
        prompt_ids, prompt_mask, uncond_ids, uncond_mask = self._align_pair(
            prompt_ids, prompt_mask, uncond_ids, uncond_mask, pad_id=pad_id,
        )

        cond_embeds = self._embed(prompt_ids)
        uncond_embeds = self._embed(uncond_ids)

        image_token_ids, old_logprobs = self.model.sample_image_tokens(
            cond_embeds, uncond_embeds, prompt_mask, uncond_mask,
            cfg_weight=cfg.cfg_weight,
            temperature=cfg.temperature,
            image_token_num=cfg.image_token_num,
        )  # both [B, L_img]

        images = self.model.decode_image_tokens(
            image_token_ids, image_size=cfg.image_size,
        )  # [B, 3, H, W] in [-1, 1]

        if cfg.rescale_to_unit:
            images_for_reward = (images + 1.0) * 0.5
            images_for_reward = images_for_reward.clamp(0.0, 1.0)
        else:
            images_for_reward = images

        rewards = await self._score(images_for_reward, repeated_prompts)

        # Per-token mask: every image-token position counts.
        token_mask = torch.ones_like(old_logprobs)

        # OnlineTrainer CEA convention (see WanDiffusersCollector):
        #   observations shape[1] == num_timesteps  (AR has 1 "step")
        #   extras["log_probs"] shape == [B, num_timesteps, ...]
        # so trainer's ``old_log_probs[:, j]`` with j=0 yields ``[B, L_img]``,
        # which is exactly what ``TokenGRPO.compute_signal_loss`` expects.
        observations = prompt_ids.unsqueeze(1)                # [B, 1, L_text]
        log_probs_3d = old_logprobs.detach().unsqueeze(1)     # [B, 1, L_img]

        return ExperienceBatch(
            observations=observations,          # [B, 1, L_text]
            actions=image_token_ids,            # sampled image tokens [B, L_img]
            rewards=rewards,                    # [B]
            dones=torch.ones(len(repeated_prompts), dtype=torch.bool, device=device),
            group_ids=group_ids,                # [B]
            extras={
                "log_probs": log_probs_3d,                    # [B, 1, L_img]
                "prompt_attention_mask": prompt_mask,         # [B, L_text]
                "uncond_input_ids": uncond_ids,               # [B, L_text]
                "uncond_attention_mask": uncond_mask,         # [B, L_text]
                "token_mask": token_mask,                     # [B, L_img]
            },
            context={
                "cfg_weight": cfg.cfg_weight,
                "image_token_num": cfg.image_token_num,
            },
            videos=images.unsqueeze(2),         # [B, 3, 1, H, W] — reward layer expects T dim
            prompts=repeated_prompts,
        )

    # ------------------------------------------------------------------
    # Public: training-time forward (Collector protocol)
    # ------------------------------------------------------------------

    def forward_step(
        self,
        model: "JanusProT2I",
        batch: ExperienceBatch,
        timestep_idx: int = 0,
    ) -> dict[str, Any]:
        """Single forward producing per-token logits over the image vocab.

        AR has no notion of "denoising step" — ``timestep_idx`` is accepted
        for protocol compatibility but ignored.
        """
        # observations may be [B, L_text] (direct-use) or [B, 1, L_text]
        # (OnlineTrainer path). Squeeze the T=1 axis if present.
        obs = batch.observations
        prompt_ids = obs.squeeze(1) if obs.dim() == 3 else obs
        prompt_mask = batch.extras["prompt_attention_mask"]
        image_token_ids = batch.actions

        prompt_embeds = self._embed_with(model, prompt_ids)
        logits = model.forward_image_logits(
            prompt_embeds, prompt_mask, image_token_ids,
        )  # [B, L_img, V_img]
        return {"logits": logits, "image_token_ids": image_token_ids}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize_prompts(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Wrap each prompt in Janus' T2I conversation format and tokenise."""
        proc = self.model.processor
        device = self.model.device
        cap = self.config.max_text_length

        # Janus' VLChatProcessor exposes its tokenizer; wrap each prompt in
        # the canonical T2I template documented in the upstream repo.
        # Using a simple format here — the upstream chat-template path
        # depends on the processor version and is brittle for batch use.
        tokenizer = proc.tokenizer
        formatted = [self._format_t2i_prompt(p) for p in prompts]
        # padding="max_length" so every collect() call produces the same
        # L_text → stack_batches across prompts in OnlineTrainer can concat
        # along dim=0. Real HF tokenizers honour this; fallbacks below catch
        # stubs that ignore the padding arg.
        enc = tokenizer(
            formatted,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=cap,
        )
        ids = enc["input_ids"]
        mask = enc["attention_mask"]
        # Belt-and-braces: force L_text == cap even if the tokenizer ignored
        # padding="max_length" (stubs, or tokenisers without a pad_token).
        if ids.shape[1] < cap:
            pad_id = getattr(tokenizer, "pad_token_id", None) or 0
            extra = cap - ids.shape[1]
            ids = torch.cat(
                [ids, torch.full((ids.shape[0], extra), pad_id, dtype=ids.dtype)],
                dim=1,
            )
            mask = torch.cat(
                [mask, torch.zeros((mask.shape[0], extra), dtype=mask.dtype)],
                dim=1,
            )
        return ids.to(device), mask.to(device)

    @staticmethod
    def _align_pair(
        a_ids: torch.Tensor, a_mask: torch.Tensor,
        b_ids: torch.Tensor, b_mask: torch.Tensor,
        pad_id: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Right-pad two ``[B, L]`` token tensors to a common length."""
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

    @staticmethod
    def _format_t2i_prompt(prompt: str) -> str:
        """Format a prompt for Janus T2I generation.

        Mirrors ``deepseek-ai/Janus/generation_inference.py``: a short
        chat-style header followed by the BOS image-generation tag.
        """
        # Keeping this minimal — for serious deployments use the upstream
        # ``apply_sft_template_for_multi_turn_prompts`` helper.
        return (
            f"<｜User｜>: {prompt}\n\n"
            f"<｜Assistant｜>:<begin_of_image>"
        )

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._embed_with(self.model, token_ids)

    @staticmethod
    def _embed_with(model: "JanusProT2I", token_ids: torch.Tensor) -> torch.Tensor:
        embed = model.language_model.get_input_embeddings()
        return embed(token_ids)

    async def _score(
        self,
        images: torch.Tensor,    # [B, 3, H, W] in [0, 1] (or [-1, 1])
        prompts: list[str],
    ) -> torch.Tensor:
        """Run reward model. Returns ``[B]`` float tensor on the model device."""
        device = self.model.device
        if self.reward_fn is None:
            return torch.zeros(images.shape[0], device=device)

        # Reward layer convention: same shape as videos in WanCollector,
        # i.e. [B, 3, T, H, W] with T=1 for images.
        videos = images.unsqueeze(2)
        # Detect coroutine-ness via introspection rather than try/except —
        # try/except TypeError would also swallow real signature bugs in
        # the reward function and re-raise from a misleading line.
        if inspect.iscoroutinefunction(self.reward_fn.score):
            scores = await self.reward_fn.score(videos=videos, prompts=prompts)
        else:
            scores = self.reward_fn.score(videos=videos, prompts=prompts)

        if isinstance(scores, dict):
            # Multi-reward dict — accept canonical names only. Falling back
            # to ``next(iter(scores.values()))`` would silently optimise a
            # single sub-component (e.g. just ``aesthetic``) when the user
            # intended the composite, with no warning.
            for key in ("reward", "total", "composite"):
                if key in scores:
                    scores = scores[key]
                    break
            else:
                raise KeyError(
                    f"reward returned a dict without a canonical composite "
                    f"key. Got keys: {sorted(scores.keys())}. Expected one "
                    "of {'reward', 'total', 'composite'} — refusing to "
                    "guess which sub-component to optimise."
                )
        if not torch.is_tensor(scores):
            scores = torch.tensor(scores, device=device, dtype=torch.float32)
        return scores.to(device).float()
