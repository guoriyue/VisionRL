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

Difference from diffusion executors: AR runs a token loop. The default path
uses the model's black-box ``sample_image_tokens`` method; setting
``sampling.use_ar_scheduler`` routes through the executor-internal
``ARTokenScheduler`` and the model's ``init_ar_state`` / ``step_ar`` /
``finalize_ar_state`` contract.

Parity contract: same prompts + same seed (when seeded) produce
bitwise-equal token ids, log-probs, and images, since
``sample_image_tokens`` is wrapped under ``torch.no_grad`` and the only
randomness is ``torch.multinomial``. The collector must apply
``torch.manual_seed(seed)`` before calling the runtime to make this
reproducible.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
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
from vrl.executors.ar import ActiveSequence, ARTokenScheduler
from vrl.executors.base import (
    BatchedFamilyPipelineExecutor,
    ChunkedFamilyPipelineExecutor,
    PipelineChunkResult,
)
from vrl.executors.batching import forward_batch_by_merging_prompts
from vrl.executors.microbatching import MicroBatchPlan

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class JanusProARChunkResult(PipelineChunkResult):
    """Output of one prompt/sample Janus-Pro AR chunk."""

    prompt_index: int
    sample_start: int
    sample_count: int
    output: torch.Tensor
    token_ids: torch.Tensor
    token_log_probs: torch.Tensor
    token_mask: torch.Tensor
    prompt_input_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor
    uncond_input_ids: torch.Tensor
    uncond_attention_mask: torch.Tensor
    context: dict[str, Any]
    peak_memory_mb: float | None = None


class JanusProPipelineExecutor(
    ChunkedFamilyPipelineExecutor,
    BatchedFamilyPipelineExecutor,
):
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

        # 3. Run the AR sampling loop.
        sample_kwargs = {
            "cfg_weight": cfg_weight,
            "temperature": temperature,
            "image_token_num": image_token_num,
        }
        if bool(sampling.get("use_ar_scheduler", False)):
            token_ids, token_log_probs = self._sample_with_ar_scheduler(
                request=request,
                sample_specs=sample_specs,
                cond_embeds=cond_embeds,
                uncond_embeds=uncond_embeds,
                prompt_mask=prompt_mask,
                uncond_mask=uncond_mask,
                image_token_num=image_token_num,
                sample_kwargs=sample_kwargs,
            )
        else:
            token_ids, token_log_probs = self.model.sample_image_tokens(
                cond_embeds,
                uncond_embeds,
                prompt_mask,
                uncond_mask,
                **sample_kwargs,
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

    def forward_chunk(
        self,
        request: GenerationRequest,
        chunk: MicroBatchPlan,
    ) -> JanusProARChunkResult:
        """Run one prompt-major AR chunk through the black-box sampling path."""

        _validate_ar_chunk(request, chunk)
        sampling = request.sampling

        cfg_weight = float(sampling.get("cfg_weight", 5.0))
        temperature = float(sampling.get("temperature", 1.0))
        image_token_num = int(sampling.get("image_token_num", 576))
        image_size = int(sampling.get("image_size", 384))
        max_text_length = int(sampling.get("max_text_length", 256))
        seed = sampling.get("seed")

        if seed is not None:
            torch.manual_seed(int(seed) + _chunk_seed_offset(request, chunk))

        repeated_prompts = [chunk.prompt] * chunk.sample_count
        prompt_ids, prompt_mask = self._tokenize_prompts(
            repeated_prompts, max_text_length=max_text_length,
        )
        uncond_ids, uncond_mask = self._tokenize_prompts(
            [""] * chunk.sample_count, max_text_length=max_text_length,
        )
        pad_id = (
            getattr(self.model.processor.tokenizer, "pad_token_id", None) or 0
        )
        prompt_ids, prompt_mask, uncond_ids, uncond_mask = self._align_pair(
            prompt_ids, prompt_mask, uncond_ids, uncond_mask, pad_id=pad_id,
        )

        cond_embeds = self._embed(prompt_ids)
        uncond_embeds = self._embed(uncond_ids)

        # Distributed AR chunks stay at prompt/sample granularity. The
        # token-level scheduler remains executor-internal for direct execution.
        token_ids, token_log_probs = self.model.sample_image_tokens(
            cond_embeds,
            uncond_embeds,
            prompt_mask,
            uncond_mask,
            cfg_weight=cfg_weight,
            temperature=temperature,
            image_token_num=image_token_num,
        )
        images = self.model.decode_image_tokens(
            token_ids, image_size=image_size,
        )
        token_mask = torch.ones_like(token_log_probs)
        peak_mem_mb = _peak_memory_mb()

        return JanusProARChunkResult(
            prompt_index=chunk.prompt_index,
            sample_start=chunk.sample_start,
            sample_count=chunk.sample_count,
            output=images,
            token_ids=token_ids,
            token_log_probs=token_log_probs,
            token_mask=token_mask,
            prompt_input_ids=prompt_ids,
            prompt_attention_mask=prompt_mask,
            uncond_input_ids=uncond_ids,
            uncond_attention_mask=uncond_mask,
            context={
                "cfg_weight": cfg_weight,
                "image_token_num": image_token_num,
                "model_family": getattr(self.model, "model_family", "janus_pro"),
            },
            peak_memory_mb=peak_mem_mb,
        )

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_specs: Sequence[GenerationSampleSpec],
        chunks: Sequence[JanusProARChunkResult],
    ) -> OutputBatch:
        """Pack prompt/sample AR chunks back into the canonical OutputBatch."""

        ordered_chunks = _ordered_ar_chunks(request, sample_specs, chunks)
        token_ids = torch.cat([chunk.token_ids for chunk in ordered_chunks], dim=0)
        token_log_probs = torch.cat(
            [chunk.token_log_probs for chunk in ordered_chunks], dim=0,
        )
        output = torch.cat([chunk.output for chunk in ordered_chunks], dim=0)
        peak_mem_mb = _max_peak_memory_mb(ordered_chunks)
        metrics = GenerationMetrics(
            num_prompts=len(request.prompts),
            num_samples=len(sample_specs),
            num_steps=int(request.sampling.get("image_token_num", 576)),
            micro_batches=len(ordered_chunks),
            peak_memory_mb=peak_mem_mb,
        )
        extra: dict[str, Any] = {
            "token_ids": token_ids,
            "token_log_probs": token_log_probs,
            "token_mask": torch.cat(
                [chunk.token_mask for chunk in ordered_chunks], dim=0,
            ),
            "prompt_input_ids": torch.cat(
                [chunk.prompt_input_ids for chunk in ordered_chunks], dim=0,
            ),
            "prompt_attention_mask": torch.cat(
                [chunk.prompt_attention_mask for chunk in ordered_chunks], dim=0,
            ),
            "uncond_input_ids": torch.cat(
                [chunk.uncond_input_ids for chunk in ordered_chunks], dim=0,
            ),
            "uncond_attention_mask": torch.cat(
                [chunk.uncond_attention_mask for chunk in ordered_chunks], dim=0,
            ),
            "context": dict(ordered_chunks[0].context),
        }

        return OutputBatch(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=list(request.prompts),
            sample_specs=list(sample_specs),
            output=output,
            rollout_trajectory_data=None,
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

    def _sample_with_ar_scheduler(
        self,
        *,
        request: GenerationRequest,
        sample_specs: list[GenerationSampleSpec],
        cond_embeds: torch.Tensor,
        uncond_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        uncond_mask: torch.Tensor,
        image_token_num: int,
        sample_kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Janus-Pro sampling through the executor-internal AR scheduler."""

        required = ("init_ar_state", "step_ar", "finalize_ar_state")
        missing = [name for name in required if not hasattr(self.model, name)]
        if missing:
            raise TypeError(
                "use_ar_scheduler=True requires model step API methods: "
                + ", ".join(missing)
            )

        if cond_embeds.shape[0] != len(sample_specs):
            raise ValueError(
                "Scheduled AR expects one sample spec per embedded row: "
                f"{len(sample_specs)} specs for {cond_embeds.shape[0]} rows"
            )

        state = self.model.init_ar_state(
            cond_embeds,
            uncond_embeds,
            prompt_mask,
            uncond_mask,
            **sample_kwargs,
        )
        sequences = [
            ActiveSequence(
                request_id=request.request_id,
                sample_id=spec.sample_id,
                family=request.family,
                task=request.task,
                tokenizer_key="janus_pro",
                dtype=str(cond_embeds.dtype),
                max_new_tokens=image_token_num,
                metadata={
                    **dict(spec.metadata),
                    "row_index": row_index,
                    "prompt_index": spec.prompt_index,
                    "sample_index": spec.sample_index,
                },
            )
            for row_index, spec in enumerate(sample_specs)
        ]
        scheduler = ARTokenScheduler(
            max_batch_size=max(
                1,
                int(request.sampling.get("ar_scheduler_batch_size", len(sequences))),
            )
        )
        scheduler.add_many(sequences)

        while True:
            batch = scheduler.pop_batch()
            if batch is None:
                break
            self.model.step_ar(state, batch.sequences)
            for sequence in batch.sequences:
                sequence.advance()
            scheduler.push_back_unfinished(batch)

        return self.model.finalize_ar_state(state)

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


def _validate_ar_chunk(request: GenerationRequest, chunk: MicroBatchPlan) -> None:
    if chunk.prompt_index >= len(request.prompts):
        raise ValueError(
            f"chunk.prompt_index={chunk.prompt_index} is out of range",
        )
    if chunk.prompt != request.prompts[chunk.prompt_index]:
        raise ValueError(
            "chunk.prompt does not match request.prompts[chunk.prompt_index]",
        )
    if chunk.sample_end > request.samples_per_prompt:
        raise ValueError(
            "chunk sample range exceeds request.samples_per_prompt: "
            f"{chunk.sample_start}:{chunk.sample_end} > "
            f"{request.samples_per_prompt}",
        )


def _ordered_ar_chunks(
    request: GenerationRequest,
    sample_specs: Sequence[GenerationSampleSpec],
    chunks: Sequence[JanusProARChunkResult],
) -> list[JanusProARChunkResult]:
    if not chunks:
        raise ValueError("chunks must be non-empty")

    ordered = sorted(chunks, key=lambda chunk: (chunk.prompt_index, chunk.sample_start))
    expected = [(spec.prompt_index, spec.sample_index) for spec in sample_specs]
    actual: list[tuple[int, int]] = []
    for chunk in ordered:
        _validate_ar_chunk(
            request,
            MicroBatchPlan(
                prompt_index=chunk.prompt_index,
                prompt=request.prompts[chunk.prompt_index],
                sample_start=chunk.sample_start,
                sample_count=chunk.sample_count,
            ),
        )
        _require_rows("output", chunk.output, chunk.sample_count)
        _require_rows("token_ids", chunk.token_ids, chunk.sample_count)
        _require_rows("token_log_probs", chunk.token_log_probs, chunk.sample_count)
        _require_rows("token_mask", chunk.token_mask, chunk.sample_count)
        _require_rows("prompt_input_ids", chunk.prompt_input_ids, chunk.sample_count)
        _require_rows(
            "prompt_attention_mask", chunk.prompt_attention_mask, chunk.sample_count,
        )
        _require_rows("uncond_input_ids", chunk.uncond_input_ids, chunk.sample_count)
        _require_rows(
            "uncond_attention_mask", chunk.uncond_attention_mask, chunk.sample_count,
        )
        actual.extend(
            (chunk.prompt_index, sample_index)
            for sample_index in range(
                chunk.sample_start, chunk.sample_start + chunk.sample_count,
            )
        )
    if actual != expected:
        raise ValueError(
            "AR chunks do not cover sample_specs in prompt-major sample order",
        )
    return ordered


def _require_rows(name: str, value: torch.Tensor, count: int) -> None:
    if value.shape[0] != count:
        raise ValueError(
            f"chunk {name} has {value.shape[0]} rows, expected {count}",
        )


def _chunk_seed_offset(request: GenerationRequest, chunk: MicroBatchPlan) -> int:
    return chunk.prompt_index * int(request.samples_per_prompt) + chunk.sample_start


def _max_peak_memory_mb(chunks: Sequence[JanusProARChunkResult]) -> float | None:
    peaks = [
        chunk.peak_memory_mb
        for chunk in chunks
        if chunk.peak_memory_mb is not None
    ]
    return max(peaks) if peaks else None


# F is imported for potential future uses (entropy etc.) — keep silent
# usage so linters don't strip the import; Janus' executor itself only
# uses model.sample_image_tokens which already does softmax internally.
_ = F


__all__ = ["JanusProARChunkResult", "JanusProPipelineExecutor"]
