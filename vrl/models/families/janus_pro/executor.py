"""Janus-Pro AR text-to-image pipeline executor.

Owns the autoregressive image-token sampling + VQ decode previously
inlined in :mod:`vrl.rollouts.collectors.janus_pro`. The collector keeps
reward scoring and ``ExperienceBatch`` packing.

Boundary:

- This module MUST NOT import ``vrl.rollouts.*`` or ``ExperienceBatch``.
- This module MUST NOT compute reward.
- Inputs come from ``GenerationRequest.sampling`` + ``prompts`` (the
  collector packs them).
- Outputs are the canonical ``OutputBatch`` whose ``output`` is the
  decoded image tensor and whose ``extra`` carries per-token ids,
  per-token log-probs, the prompt token ids/masks (needed for replay
  forward), and the unconditional token ids/masks (needed for replay /
  audit).

Difference from diffusion executors: AR is a *single* black-box call —
the model's ``sample_image_tokens`` runs the entire AR loop internally
with KV cache and CFG. There is no per-step protocol here.

Parity contract: same prompts + same seed (when seeded) produce
bitwise-equal token ids, log-probs, and images, since
``sample_image_tokens`` is wrapped under ``torch.no_grad`` and the only
randomness is ``torch.multinomial``. The collector must apply
``torch.manual_seed(seed)`` before calling the runtime to make this
reproducible.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from vrl.engine.generation.types import (
    GenerationMetrics,
    GenerationRequest,
    GenerationSampleSpec,
    OutputBatch,
    WorkloadSignature,
)
from vrl.executors.base import BatchedFamilyPipelineExecutor
from vrl.executors.batching import forward_batch_by_merging_prompts

logger = logging.getLogger(__name__)


class JanusProPipelineExecutor(BatchedFamilyPipelineExecutor):
    """AR executor for Janus-Pro text-to-image rollouts.

    The collector constructs a ``GenerationRequest`` whose ``sampling``
    dict holds:

    - ``cfg_weight``: float — classifier-free guidance scale.
    - ``temperature``: float — sampling temperature.
    - ``image_token_num``: int — number of AR image tokens to generate.
    - ``image_size``: int — VQ decoder output side length (pixels).
    - ``max_text_length``: int — pad/truncate prompts to this length so
      ``L_text`` is constant across multi-prompt requests.
    - ``seed``: int | None — when set, ``torch.manual_seed(seed)`` is
      applied before sampling for parity tests.

    The executor returns an ``OutputBatch`` whose ``output`` is the
    decoded ``[B, 3, H, W]`` image tensor in ``[-1, 1]`` and whose
    ``extra`` dict carries:

    - ``token_ids``: ``[B, L_img]`` int64 — sampled image-token ids.
    - ``token_log_probs``: ``[B, L_img]`` float — per-token log-probs
      under the conditional (un-guided) policy. These are GRPO's
      ``old_log_prob``.
    - ``token_mask``: ``[B, L_img]`` float — ones tensor (Janus has no
      padding in the image-token sequence).
    - ``prompt_input_ids``: ``[B, L_text]`` int64.
    - ``prompt_attention_mask``: ``[B, L_text]`` int64.
    - ``uncond_input_ids``: ``[B, L_text]`` int64.
    - ``uncond_attention_mask``: ``[B, L_text]`` int64.

    These keys mirror the pre-migration collector's ``ExperienceBatch``
    fields exactly (see ``vrl/rollouts/collectors/janus_pro.py``) so
    the trainer's ``replay_forward`` contract is preserved.
    """

    family: str = "janus_pro"
    task: str = "ar_t2i"

    def __init__(self, model: Any) -> None:
        """Construct the executor.

        Args:
          model: a ``JanusProPolicy`` (or a stub exposing the same
            interface: ``processor``, ``device``, ``language_model``,
            ``sample_image_tokens``, ``decode_image_tokens``).
        """
        self.model = model

    # -- protocol ------------------------------------------------------

    def workload_signature(self, request: GenerationRequest) -> WorkloadSignature:
        return WorkloadSignature.from_request(request)

    def forward(
        self,
        request: GenerationRequest,
        sample_specs: list[GenerationSampleSpec],
    ) -> OutputBatch:
        sampling = request.sampling
        prompts = list(request.prompts)
        samples_per_prompt = int(request.samples_per_prompt)

        cfg_weight = float(sampling.get("cfg_weight", 5.0))
        temperature = float(sampling.get("temperature", 1.0))
        image_token_num = int(sampling.get("image_token_num", 576))
        image_size = int(sampling.get("image_size", 384))
        max_text_length = int(sampling.get("max_text_length", 256))
        seed = sampling.get("seed")

        if seed is not None:
            # AR sampling uses torch.multinomial — we seed the global RNG
            # because that's the only entropy source in sample_image_tokens.
            # This makes parity tests deterministic.
            torch.manual_seed(int(seed))

        # Repeat prompts samples_per_prompt times so the AR loop runs
        # samples_per_prompt independent sequences per prompt. Order is
        # prompt-major to match GenerationIdFactory.build_sample_specs.
        repeated_prompts = [p for p in prompts for _ in range(samples_per_prompt)]

        # 1. Tokenise conditional + unconditional prompts.
        prompt_ids, prompt_mask = self._tokenize_prompts(
            repeated_prompts, max_text_length=max_text_length,
        )
        uncond_ids, uncond_mask = self._tokenize_prompts(
            [""] * len(repeated_prompts), max_text_length=max_text_length,
        )
        pad_id = (
            getattr(self.model.processor.tokenizer, "pad_token_id", None) or 0
        )
        prompt_ids, prompt_mask, uncond_ids, uncond_mask = self._align_pair(
            prompt_ids, prompt_mask, uncond_ids, uncond_mask, pad_id=pad_id,
        )

        # 2. Embed both halves with the language model's input embedding.
        cond_embeds = self._embed(prompt_ids)
        uncond_embeds = self._embed(uncond_ids)

        # 3. Run the AR sampling loop — black-box, owns its own KV cache.
        token_ids, token_log_probs = self.model.sample_image_tokens(
            cond_embeds,
            uncond_embeds,
            prompt_mask,
            uncond_mask,
            cfg_weight=cfg_weight,
            temperature=temperature,
            image_token_num=image_token_num,
        )  # both [B, L_img]

        # 4. VQ decode tokens → pixels in [-1, 1].
        images = self.model.decode_image_tokens(
            token_ids, image_size=image_size,
        )  # [B, 3, H, W]

        # 5. Token mask: every image-token position is meaningful (no
        # padding). Match the dtype of token_log_probs so trainer-side
        # multiplications don't trigger float upcasts.
        token_mask = torch.ones_like(token_log_probs)

        peak_mem_mb = _peak_memory_mb()
        metrics = GenerationMetrics(
            num_prompts=len(prompts),
            num_samples=len(sample_specs),
            num_steps=image_token_num,
            micro_batches=1,
            peak_memory_mb=peak_mem_mb,
        )

        extra: dict[str, Any] = {
            "token_ids": token_ids,
            "token_log_probs": token_log_probs,
            "token_mask": token_mask,
            "prompt_input_ids": prompt_ids,
            "prompt_attention_mask": prompt_mask,
            "uncond_input_ids": uncond_ids,
            "uncond_attention_mask": uncond_mask,
            "context": {
                "cfg_weight": cfg_weight,
                "image_token_num": image_token_num,
                "model_family": getattr(self.model, "model_family", "janus_pro"),
            },
        }

        return OutputBatch(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=prompts,
            sample_specs=sample_specs,
            output=images,
            rollout_trajectory_data=None,  # AR has no DiT trajectory
            extra=extra,
            metrics=metrics,
            peak_memory_mb=peak_mem_mb or 0.0,
        )

    def forward_batch(
        self,
        requests: list[GenerationRequest],
        sample_specs_by_request: dict[str, list[GenerationSampleSpec]],
    ) -> dict[str, OutputBatch]:
        return forward_batch_by_merging_prompts(
            self, requests, sample_specs_by_request,
        )

    # -- internals -----------------------------------------------------

    def _tokenize_prompts(
        self,
        prompts: list[str],
        *,
        max_text_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenise a list of prompts with the Janus chat template.

        Mirrors the pre-migration ``JanusProCollector._tokenize_prompts``
        contract: ``[B, max_text_length]`` ids + mask, right-padded with
        ``pad_token_id`` (or 0 if none), all on the model device.
        """
        tokenizer = self.model.processor.tokenizer
        device = self.model.device

        formatted = [self._format_t2i_prompt(p) for p in prompts]
        enc = tokenizer(
            formatted,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_text_length,
        )
        ids = enc["input_ids"]
        mask = enc["attention_mask"]

        # Belt-and-braces: enforce L_text == max_text_length even if the
        # tokenizer ignored padding="max_length" (stubs / tokenisers
        # without a pad_token).
        if ids.shape[1] < max_text_length:
            pad_id = getattr(tokenizer, "pad_token_id", None) or 0
            extra_len = max_text_length - ids.shape[1]
            ids = torch.cat(
                [ids, torch.full(
                    (ids.shape[0], extra_len), pad_id, dtype=ids.dtype,
                )],
                dim=1,
            )
            mask = torch.cat(
                [mask, torch.zeros(
                    (mask.shape[0], extra_len), dtype=mask.dtype,
                )],
                dim=1,
            )
        return ids.to(device), mask.to(device)

    @staticmethod
    def _align_pair(
        a_ids: torch.Tensor,
        a_mask: torch.Tensor,
        b_ids: torch.Tensor,
        b_mask: torch.Tensor,
        pad_id: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Right-pad two ``[B, L]`` token tensors to a common length."""
        L = max(a_ids.shape[1], b_ids.shape[1])

        def _pad(
            ids: torch.Tensor, mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            cur = ids.shape[1]
            if cur == L:
                return ids, mask
            extra_len = L - cur
            pad_ids = torch.full(
                (ids.shape[0], extra_len), pad_id,
                dtype=ids.dtype, device=ids.device,
            )
            pad_mask = torch.zeros(
                (mask.shape[0], extra_len),
                dtype=mask.dtype, device=mask.device,
            )
            return (
                torch.cat([ids, pad_ids], dim=1),
                torch.cat([mask, pad_mask], dim=1),
            )

        a_ids, a_mask = _pad(a_ids, a_mask)
        b_ids, b_mask = _pad(b_ids, b_mask)
        return a_ids, a_mask, b_ids, b_mask

    @staticmethod
    def _format_t2i_prompt(prompt: str) -> str:
        """Format a prompt with Janus' T2I chat template.

        Mirrors ``deepseek-ai/Janus/generation_inference.py``: a short
        chat-style header followed by the BOS image-generation tag.
        """
        return (
            f"<｜User｜>: {prompt}\n\n"  # noqa: RUF001
            f"<｜Assistant｜>:<begin_of_image>"  # noqa: RUF001
        )

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        embed = self.model.language_model.get_input_embeddings()
        return embed(token_ids)


def _peak_memory_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    try:
        peak_bytes = torch.cuda.max_memory_allocated()
    except Exception:
        return None
    return peak_bytes / (1024 * 1024)


# F is imported for potential future uses (entropy etc.) — keep silent
# usage so linters don't strip the import; Janus' executor itself only
# uses model.sample_image_tokens which already does softmax internally.
_ = F


__all__ = ["JanusProPipelineExecutor"]
