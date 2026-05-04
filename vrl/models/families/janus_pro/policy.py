"""Janus-Pro-1B wrapper for autoregressive text-to-image RL.

This file isolates every Janus-specific detail (image-token vocab range,
``gen_head`` projection, CFG sampling, VQ decode) behind a small surface
that the generic GRPO trainer can call:

  * ``JanusProPolicy.forward_image_logits(...)``
        Train-time forward — returns logits over the *image* vocab for
        each image-token position. Used by the evaluator to recompute
        new log-probs under the current policy.

  * ``JanusProPolicy.sample_image_tokens(...)``
        Inference-time AR sampler with classifier-free guidance.
        Returns ``(image_token_ids, sampling_logprobs)`` — these
        log-probs are the ``old_logprob`` of GRPO.

  * ``JanusProPolicy.decode_image_tokens(...)``
        Decode 24x24 image tokens → pixels via the frozen VQ model.

  * ``JanusProPolicy.disable_adapter()``
        Context manager that turns LoRA off so the same module can serve
        as the reference policy (DPO-style ``disable_adapter`` trick).

Why a custom forward instead of stock ``forward()``?
====================================================
Janus' generation path is *not* the same as its understanding-path
``forward``. For T2I we must:
  1. Embed text-prompt tokens with ``language_model.get_input_embeddings()``
  2. Embed previously-sampled image tokens with ``prepare_gen_img_embeds``
  3. Run the language-model trunk
  4. Project the *image-token* hidden states with ``gen_head``
     (NOT ``language_model.lm_head`` — that produces text logits!)

Doing this wrong silently optimises against text logits and trains nothing.

References
----------
DeepSeek's reference implementation:
  https://github.com/deepseek-ai/Janus/blob/main/generation_inference.py
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vrl.models.ar import (
    ARStepResult,
    AutoregressivePolicy,
    ar_concat_rows,
    ar_split_rows,
)

logger = logging.getLogger(__name__)

# Janus-Pro-1B image-tokenizer constants (from deepseek-ai/Janus config)
JANUS_IMAGE_TOKEN_NUM = 576           # 24 x 24 latent grid per image
JANUS_IMAGE_VOCAB_SIZE = 16_384       # gen_vision_model codebook size
JANUS_IMAGE_PATCH_SIZE = 16           # decoder upsample factor → 384 px
JANUS_IMAGE_PIXEL_SIZE = 384


@dataclass(slots=True)
class JanusProConfig:
    """Hyper-parameters for the Janus-Pro wrapper.

    The defaults target Janus-Pro-1B (single-H100 RL feasible).
    """

    model_path: str = "deepseek-ai/Janus-Pro-1B"
    dtype: str = "bfloat16"           # "bfloat16" | "float16" | "float32"

    # LoRA
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    # Janus' language-model uses LLaMA-style projection names.
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
    )
    lora_init: str = "gaussian"       # PEFT ``init_lora_weights``

    # Generation defaults — used by sample_image_tokens
    cfg_weight: float = 5.0
    temperature: float = 1.0
    image_token_num: int = JANUS_IMAGE_TOKEN_NUM

    # Misc
    trust_remote_code: bool = True
    device: str = "cuda"

    # Frozen sub-modules (we never train these for T2I RL)
    freeze_vq: bool = True            # gen_vision_model
    freeze_vision_encoder: bool = True  # SigLIP understanding tower
    freeze_aligner: bool = True       # gen_aligner / aligner

    # VQ decoder shape — None ⇒ auto-detect from ``vq_model.config``.
    # Janus-Pro-1B uses 8 latent channels; Janus-Pro-7B may differ.
    # Override only if auto-detect fails.
    vq_latent_channels: int | None = None

    # Cached references — populated at __post_init__ time by the wrapper
    _frame_constants: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class JanusProARState:
    """Mutable per-row state for scheduled Janus-Pro AR sampling."""

    cond_past_rows: list[Any | None]
    uncond_past_rows: list[Any | None]
    cond_cur_embeds_rows: list[torch.Tensor]
    uncond_cur_embeds_rows: list[torch.Tensor]
    cond_attn_rows: list[torch.Tensor]
    uncond_attn_rows: list[torch.Tensor]
    token_ids: torch.Tensor
    logprobs: torch.Tensor
    cfg_weight: float
    temperature: float
    image_token_num: int
    positions: torch.Tensor
    position: int = 0


# ---------------------------------------------------------------------------
# Functional helper: project hidden states to image-token logits
# ---------------------------------------------------------------------------


def image_token_logits_from_hidden(
    mmgpt: nn.Module,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Apply Janus' generation head to hidden states.

    Args:
      mmgpt: a ``MultiModalityCausalLM`` instance (or LoRA-wrapped peer).
      hidden_states: trunk output at *image-token* positions, shape
        ``[B, L_img, hidden_size]``.

    Returns:
      Logits over the image vocabulary, shape
      ``[B, L_img, JANUS_IMAGE_VOCAB_SIZE]``.
    """
    # ``gen_head`` lives on the underlying mmgpt; PEFT wrapping preserves it.
    # See JanusProPolicy._base for why we can't use hasattr(base_model) as the key.
    inner = getattr(mmgpt, "base_model", None)
    if inner is not None and hasattr(inner, "model") and inner.model is not mmgpt:
        base = inner.model
    else:
        base = mmgpt
    return base.gen_head(hidden_states)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class JanusProPolicy(nn.Module, AutoregressivePolicy):
    """Train-and-sample wrapper for Janus-Pro text-to-image generation.

    Keeps the LoRA-wrapped language model + frozen vq / vision / aligner
    in a single ``nn.Module`` so it integrates cleanly with FSDP / EMA.
    """

    model_family: str = "janus-pro-t2i"

    def __init__(
        self,
        config: JanusProConfig | None = None,
        *,
        mmgpt: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        """Construct the wrapper.

        Args:
          config: hyper-parameters. ``None`` → defaults for Janus-Pro-1B.
          mmgpt: optional pre-loaded ``MultiModalityCausalLM`` (saves the
            ~3 GB checkpoint download in tests). When ``None``, we load
            from ``config.model_path``.
          processor: optional pre-loaded ``VLChatProcessor``.
        """
        super().__init__()
        self.config = config or JanusProConfig()

        if mmgpt is None:
            mmgpt, processor = _load_janus_from_pretrained(self.config)
        elif processor is None:
            raise ValueError("Must pass `processor` when `mmgpt` is provided")

        self._processor = processor

        # Freeze everything by default — LoRA wrap re-enables only attention
        # projections in the language model.
        for p in mmgpt.parameters():
            p.requires_grad_(False)

        if self.config.use_lora:
            mmgpt = self._apply_lora(mmgpt)

        self.mmgpt = mmgpt

        # Sanity: confirm gen_head + gen_vision_model exist
        base = self._base()
        for attr in ("gen_head", "gen_vision_model", "language_model"):
            if not hasattr(base, attr):
                raise RuntimeError(
                    f"Loaded model is missing `{attr}` — does not look like "
                    "a Janus MultiModalityCausalLM checkpoint."
                )

    # ------------------------------------------------------------------
    # Sub-module accessors
    # ------------------------------------------------------------------

    def _base(self) -> nn.Module:
        """Return the unwrapped MultiModalityCausalLM (peels PEFT wrap).

        Cannot key off ``hasattr(m, "base_model")`` alone: HF
        ``PreTrainedModel`` exposes ``base_model`` as a property returning
        ``self`` even when there's no PEFT wrap, and that object has no
        ``.model`` attr. Only peel when the PEFT inner path exists.
        """
        m = self.mmgpt
        inner = getattr(m, "base_model", None)
        if inner is not None and hasattr(inner, "model") and inner.model is not m:
            return inner.model
        return m

    def _lm_trunk(self) -> nn.Module:
        """Return the LlamaModel trunk that emits ``last_hidden_state``.

        Layering depends on whether LoRA is attached:
          * No LoRA:  ``language_model`` is ``LlamaForCausalLM``; its
            ``.model`` is the ``LlamaModel`` trunk.
          * With LoRA: ``language_model`` is a PEFT ``LoraModel`` wrapping
            ``LlamaForCausalLM``; the trunk is two hops in via
            ``base_model.model.model``.

        Calling ``LlamaForCausalLM`` directly returns a ``CausalLMOutputWithPast``
        which has ``.logits`` over the text vocab — the *wrong* projection
        for image-token generation. We need the raw hidden states, so we
        unwrap all the way down to ``LlamaModel``.
        """
        lm = self._base().language_model
        peft_inner = getattr(lm, "base_model", None)
        if (
            peft_inner is not None
            and hasattr(peft_inner, "model")
            and peft_inner.model is not lm
        ):
            # PEFT-wrapped: peft.base_model.model is LlamaForCausalLM
            cls_lm = peft_inner.model
        else:
            cls_lm = lm
        return cls_lm.model if hasattr(cls_lm, "model") else cls_lm

    @property
    def processor(self) -> Any:
        return self._processor

    @property
    def language_model(self) -> nn.Module:
        return self._base().language_model

    @property
    def vq_model(self) -> nn.Module:
        return self._base().gen_vision_model

    @property
    def device(self) -> torch.device:
        return next(self.mmgpt.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.mmgpt.parameters()).dtype

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return (p for p in self.mmgpt.parameters() if p.requires_grad)

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    # ------------------------------------------------------------------
    # LoRA / reference-policy helpers
    # ------------------------------------------------------------------

    def _apply_lora(self, mmgpt: Any) -> Any:
        """Attach a PEFT LoRA adapter to the language-model trunk."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "PEFT is required for use_lora=True. "
                "pip install peft>=0.12"
            ) from e

        lora_cfg = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            init_lora_weights=self.config.lora_init,
            target_modules=list(self.config.lora_target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        # Wrap ONLY the language model — vq / vision / aligner stay frozen.
        mmgpt.language_model = get_peft_model(mmgpt.language_model, lora_cfg)
        logger.info(
            "Applied LoRA (rank=%d, alpha=%d) to Janus language model. "
            "Trainable params will be reported by trainable_param_count().",
            self.config.lora_rank, self.config.lora_alpha,
        )
        return mmgpt

    @property
    def has_lora_adapter(self) -> bool:
        """True iff this wrapper carries a real PEFT adapter we can disable.

        Probes the *language-model* sub-module — that is where PEFT injects
        ``disable_adapter``. The outer wrapper always exposes the method,
        which is exactly the silent-failure trap callers should not fall
        into.
        """
        lm = self.language_model
        return hasattr(lm, "disable_adapter") and callable(
            lm.disable_adapter
        )

    @contextlib.contextmanager
    def disable_adapter(self) -> Iterator[None]:
        """Temporarily disable the LoRA adapter — for reference forward.

        Refuses-to-fail-silently contract:
          If ``use_lora=False`` (or the language-model otherwise lacks a
          PEFT adapter) we *raise*, never silently yield. The whole point
          of ``disable_adapter`` is to give a different forward — yielding
          a no-op produces ``ref == policy`` and KL ≡ 0, which the caller
          cannot detect.

        Callers without a LoRA adapter must pass an explicit ``ref_model``
        wherever they were going to use this context manager.
        """
        if not self.has_lora_adapter:
            raise RuntimeError(
                "JanusProPolicy.disable_adapter() called but no PEFT adapter "
                "is attached (use_lora=False or LoRA wrap was skipped). "
                "A no-op yield would silently make ref_pred == policy_pred "
                "and KL ≡ 0. Either construct with use_lora=True, or pass "
                "a separate frozen ref_model to whatever needs the "
                "reference forward."
            )
        with self.language_model.disable_adapter():
            yield

    # ------------------------------------------------------------------
    # Train-time forward — image-token logits
    # ------------------------------------------------------------------

    def forward_image_logits(
        self,
        prompt_inputs_embeds: torch.Tensor,    # [B, L_text, H]
        prompt_attention_mask: torch.Tensor,   # [B, L_text]
        image_token_ids: torch.Tensor,         # [B, L_img]
    ) -> torch.Tensor:
        """One forward pass returning per-position image-vocab logits.

        Layout convention: text comes first, then image tokens. We feed
        the *teacher-forced* sequence and extract logits at positions
        that *predict* each image token (i.e. the position immediately
        before it).

        Args:
          prompt_inputs_embeds: text embeddings (already passed through
            ``language_model.get_input_embeddings()``). Shape
            ``[B, L_text, hidden_size]``.
          prompt_attention_mask: 1/0 mask for the text part, ``[B, L_text]``.
          image_token_ids: previously-sampled image tokens to score, shape
            ``[B, L_img]``. ``L_img`` is typically
            ``JANUS_IMAGE_TOKEN_NUM`` (576).

        Returns:
          Logits over image vocab at positions that *predict* each
          image token. Shape ``[B, L_img, JANUS_IMAGE_VOCAB_SIZE]``.
        """
        base = self._base()
        B, L_img = image_token_ids.shape

        # Embed image tokens via Janus' generation embedder.
        img_embeds = base.prepare_gen_img_embeds(image_token_ids)  # [B, L_img, H]

        # Concat: [text | image[:-1]]  — image[-1] doesn't predict anything new
        inputs_embeds = torch.cat(
            [prompt_inputs_embeds, img_embeds[:, :-1, :]], dim=1
        )
        L_text = prompt_inputs_embeds.shape[1]
        attn = torch.cat(
            [
                prompt_attention_mask,
                torch.ones(
                    B, L_img - 1,
                    dtype=prompt_attention_mask.dtype,
                    device=prompt_attention_mask.device,
                ),
            ],
            dim=1,
        )

        outputs = self._lm_trunk()(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            use_cache=False,
            output_hidden_states=False,
        )
        hidden = outputs.last_hidden_state  # [B, L_text + L_img - 1, H]

        # Positions that *predict* image_token_ids[:, 0..L_img-1]
        # are L_text - 1, L_text, ..., L_text + L_img - 2.
        gen_hidden = hidden[:, L_text - 1 : L_text - 1 + L_img, :]
        return image_token_logits_from_hidden(self.mmgpt, gen_hidden)

    # ------------------------------------------------------------------
    # Replay forward — recompute logits at training time
    # ------------------------------------------------------------------

    def replay_forward(
        self,
        batch: Any,
        timestep_idx: int = 0,
    ) -> dict[str, Any]:
        """Single forward producing per-token logits over the image vocab.

        Train-time replay: given a ``RolloutBatch`` that recorded
        ``observations`` (text prompt ids), ``extras["prompt_attention_mask"]``,
        and ``actions`` (sampled image tokens), recompute the conditional
        logits over the image vocab so the evaluator can gather log-probs
        of the sampled tokens under the *current* policy.

        AR has no notion of "denoising step", so ``timestep_idx`` is ignored.

        See ``vrl/models/ar.py::AutoregressivePolicy`` for the shared AR
        replay protocol; see ``SPRINT_ar_support.md`` §5 for why Janus and
        NextStep do not share a return-dict schema.

        Returns:
          ``{"logits": Tensor[B, L_img, V_img], "image_token_ids": Tensor[B, L_img]}``.
        """
        # observations may be [B, L_text] (direct-use) or [B, 1, L_text]
        # (OnlineTrainer path). Squeeze the T=1 axis if present.
        obs = batch.observations
        prompt_ids = obs.squeeze(1) if obs.dim() == 3 else obs
        prompt_mask = batch.extras["prompt_attention_mask"]
        image_token_ids = batch.actions

        embed = self.language_model.get_input_embeddings()
        prompt_embeds = embed(prompt_ids)
        logits = self.forward_image_logits(
            prompt_embeds, prompt_mask, image_token_ids,
        )  # [B, L_img, V_img]
        return {"logits": logits, "image_token_ids": image_token_ids}

    # ------------------------------------------------------------------
    # Inference-time AR sampler with classifier-free guidance
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_image_tokens(
        self,
        cond_inputs_embeds: torch.Tensor,      # [B, L_text, H]
        uncond_inputs_embeds: torch.Tensor,    # [B, L_text, H]
        cond_attention_mask: torch.Tensor,     # [B, L_text]
        uncond_attention_mask: torch.Tensor,   # [B, L_text]
        *,
        cfg_weight: float | None = None,
        temperature: float | None = None,
        image_token_num: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run AR image-token sampling with CFG.

        Implements the standard Janus generation loop, but additionally
        records the per-token log-probability under the *guided*
        distribution that produced each sampled token. These log-probs
        are exactly ``old_logprob`` for GRPO.

        Args:
          cond_inputs_embeds:  conditional text embeddings.
          uncond_inputs_embeds: unconditional embeddings (typically
            from the same prompt with the conditioning text replaced by
            the empty/null token).
          cond_attention_mask / uncond_attention_mask: 1/0 masks.
          cfg_weight: guidance scale. ``None`` → ``self.config.cfg_weight``.
          temperature: sampling temperature.
          image_token_num: how many image tokens to sample (default 576).

        Returns:
          ``(image_token_ids, logprobs)`` — both shape ``[B, L_img]``.
          ``logprobs`` is computed under the conditional policy distribution;
          CFG is used only to choose the sampled token.
        """
        cfg = cfg_weight if cfg_weight is not None else self.config.cfg_weight
        temp = temperature if temperature is not None else self.config.temperature
        L_img = image_token_num or self.config.image_token_num

        state = self.init_ar_state(
            cond_inputs_embeds,
            uncond_inputs_embeds,
            cond_attention_mask,
            uncond_attention_mask,
            cfg_weight=cfg,
            temperature=temp,
            image_token_num=L_img,
        )
        while state.position < state.image_token_num:
            self._sample_ar_step(state)
        return self.finalize_ar_state(state)

    @torch.no_grad()
    def init_ar_state(
        self,
        cond_inputs_embeds: torch.Tensor,
        uncond_inputs_embeds: torch.Tensor,
        cond_attention_mask: torch.Tensor,
        uncond_attention_mask: torch.Tensor,
        *,
        cfg_weight: float | None = None,
        temperature: float | None = None,
        image_token_num: int | None = None,
    ) -> JanusProARState:
        """Initialize per-row state for scheduled Janus-Pro AR sampling."""

        cfg = cfg_weight if cfg_weight is not None else self.config.cfg_weight
        temp = temperature if temperature is not None else self.config.temperature
        image_token_num = image_token_num or self.config.image_token_num
        batch_size = cond_inputs_embeds.shape[0]
        device = cond_inputs_embeds.device

        return JanusProARState(
            cond_past_rows=[None for _ in range(batch_size)],
            uncond_past_rows=[None for _ in range(batch_size)],
            cond_cur_embeds_rows=[
                cond_inputs_embeds[row : row + 1] for row in range(batch_size)
            ],
            uncond_cur_embeds_rows=[
                uncond_inputs_embeds[row : row + 1] for row in range(batch_size)
            ],
            cond_attn_rows=[
                cond_attention_mask[row : row + 1] for row in range(batch_size)
            ],
            uncond_attn_rows=[
                uncond_attention_mask[row : row + 1] for row in range(batch_size)
            ],
            token_ids=torch.empty(
                batch_size, image_token_num, dtype=torch.long, device=device
            ),
            logprobs=torch.empty(
                batch_size, image_token_num, dtype=torch.float32, device=device
            ),
            cfg_weight=float(cfg),
            temperature=float(temp),
            image_token_num=int(image_token_num),
            positions=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    @torch.no_grad()
    def step_ar(
        self,
        state: JanusProARState,
        sequences: list[Any],
        *,
        generator: torch.Generator | None = None,
    ) -> ARStepResult:
        """Run one scheduled AR token step for rows at the same position."""

        del generator
        if not sequences:
            raise ValueError("step_ar requires at least one ActiveSequence")

        row_indices = [int(seq.metadata.get("row_index", -1)) for seq in sequences]
        if any(row < 0 or row >= state.token_ids.shape[0] for row in row_indices):
            raise ValueError(f"invalid Janus row indices: {row_indices}")

        positions = [int(seq.position) for seq in sequences]
        if len(set(positions)) != 1:
            raise ValueError("ActiveSequence positions must match within one AR step")
        expected_positions = [
            int(state.positions[row].item()) for row in row_indices
        ]
        if positions != expected_positions:
            raise ValueError(
                "ActiveSequence positions must match JanusProARState row positions"
            )

        token, log_prob = self._sample_ar_step(
            state,
            row_indices=row_indices,
            position=positions[0],
        )
        return ARStepResult(
            sequence_ids=[str(seq.sample_id) for seq in sequences],
            positions=positions,
            token=token,
            log_prob=log_prob,
        )

    @torch.no_grad()
    def finalize_ar_state(
        self,
        state: JanusProARState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return sampled image-token ids and old log-probs."""

        return state.token_ids, state.logprobs

    def _sample_ar_step(
        self,
        state: JanusProARState,
        *,
        row_indices: list[int] | None = None,
        position: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if row_indices is None:
            row_indices = list(range(state.token_ids.shape[0]))
        if not row_indices:
            raise ValueError("row_indices must be non-empty")
        if any(row < 0 or row >= state.token_ids.shape[0] for row in row_indices):
            raise ValueError(f"invalid Janus row indices: {row_indices}")

        row_positions = [int(state.positions[row].item()) for row in row_indices]
        if len(set(row_positions)) != 1:
            raise ValueError("Janus rows in one AR step must share a position")
        position = row_positions[0] if position is None else int(position)
        if any(pos != position for pos in row_positions):
            raise ValueError("requested position does not match row positions")
        if position >= state.image_token_num:
            raise ValueError("JanusProARState has already finished sampling")

        rows = torch.tensor(
            row_indices, dtype=torch.long, device=state.token_ids.device
        )
        batch_size = len(row_indices)
        cond_embeds = torch.cat(
            [state.cond_cur_embeds_rows[row] for row in row_indices], dim=0
        )
        uncond_embeds = torch.cat(
            [state.uncond_cur_embeds_rows[row] for row in row_indices], dim=0
        )
        cond_attn = torch.cat(
            [state.cond_attn_rows[row] for row in row_indices], dim=0
        )
        uncond_attn = torch.cat(
            [state.uncond_attn_rows[row] for row in row_indices], dim=0
        )

        cond_past = [state.cond_past_rows[row] for row in row_indices]
        uncond_past = [state.uncond_past_rows[row] for row in row_indices]
        past_kv = None
        all_past = [*cond_past, *uncond_past]
        if any(past is not None for past in all_past):
            if any(past is None for past in all_past):
                raise ValueError("Janus AR cache rows are partially initialized")
            past_kv = ar_concat_rows(all_past)

        outputs = self._lm_trunk()(
            inputs_embeds=torch.cat([cond_embeds, uncond_embeds], dim=0),
            attention_mask=torch.cat([cond_attn, uncond_attn], dim=0),
            use_cache=True,
            past_key_values=past_kv,
        )
        past_rows = ar_split_rows(outputs.past_key_values, batch_size * 2)
        for offset, row in enumerate(row_indices):
            state.cond_past_rows[row] = past_rows[offset]
            state.uncond_past_rows[row] = past_rows[batch_size + offset]

        hidden = outputs.last_hidden_state[:, -1:, :]
        logits = image_token_logits_from_hidden(self.mmgpt, hidden).squeeze(1)
        cond_logits, uncond_logits = logits.chunk(2, dim=0)
        guided = uncond_logits + state.cfg_weight * (cond_logits - uncond_logits)

        probs = F.softmax(guided / state.temperature, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

        log_probs = F.log_softmax(cond_logits / state.temperature, dim=-1)
        lp = log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

        state.token_ids[rows, position] = sampled
        state.logprobs[rows, position] = lp

        next_embed = self._base().prepare_gen_img_embeds(
            torch.cat([sampled, sampled], dim=0).unsqueeze(-1)
        )
        cond_next_embed, uncond_next_embed = next_embed.chunk(2, dim=0)
        cond_next_attn = torch.cat(
            [
                cond_attn,
                torch.ones(
                    batch_size, 1, dtype=cond_attn.dtype, device=cond_attn.device
                ),
            ],
            dim=1,
        )
        uncond_next_attn = torch.cat(
            [
                uncond_attn,
                torch.ones(
                    batch_size,
                    1,
                    dtype=uncond_attn.dtype,
                    device=uncond_attn.device,
                ),
            ],
            dim=1,
        )
        for offset, row in enumerate(row_indices):
            state.cond_cur_embeds_rows[row] = cond_next_embed[offset : offset + 1]
            state.uncond_cur_embeds_rows[row] = uncond_next_embed[offset : offset + 1]
            state.cond_attn_rows[row] = cond_next_attn[offset : offset + 1]
            state.uncond_attn_rows[row] = uncond_next_attn[offset : offset + 1]

        state.positions[rows] += 1
        state.position = int(state.positions.min().item())
        return sampled, lp

    # ------------------------------------------------------------------
    # VQ decode — image tokens → pixels
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode_image_tokens(
        self,
        image_token_ids: torch.Tensor,    # [B, L_img]
        *,
        image_size: int = JANUS_IMAGE_PIXEL_SIZE,
    ) -> torch.Tensor:
        """Decode image-token grids to RGB pixels in [-1, 1].

        Returns shape ``[B, 3, image_size, image_size]``.
        """
        B, L = image_token_ids.shape
        side = int(L ** 0.5)
        assert side * side == L, f"expected square grid, got L_img={L}"
        # Janus' decode_code expects token grid + (B, C, H, W) target shape.
        # Latent channels differ across Janus-Pro variants — read from the
        # VQ config so 7B works without a code change.
        latent_channels = self._resolve_vq_latent_channels()
        decoded = self.vq_model.decode_code(
            image_token_ids.to(torch.int32),
            shape=[B, latent_channels, side, side],
        )
        return decoded.clamp(-1.0, 1.0)

    def _resolve_vq_latent_channels(self) -> int:
        """Resolve the VQ decoder's latent-channel dimension.

        ``decode_code`` feeds ``shape[1]`` to ``get_codebook_entry``, which
        uses it as the *codebook-entry* dim, NOT the encoder z_channels dim.
        On Janus-Pro-1B these differ: ``config.z_channels=256`` (encoder
        hidden) but the quantizer codebook is 8-dim — using 256 produces a
        silent reshape explosion.

        Resolution is intentionally strict: use the explicit override or the
        live quantizer embedding shape. If neither is available, fail instead
        of guessing a checkpoint-specific constant.
        """
        override = self.config.vq_latent_channels
        if override is not None:
            if not isinstance(override, int) or override <= 0:
                raise RuntimeError(
                    f"vq_latent_channels override must be a positive int; "
                    f"got {override!r}"
                )
            return override

        # Live probe of the quantizer — authoritative on any Janus variant.
        quant = getattr(self.vq_model, "quantize", None)
        emb = getattr(quant, "embedding", None) if quant is not None else None
        if emb is not None and hasattr(emb, "weight"):
            w = emb.weight
            if w.ndim == 2 and w.shape[-1] > 0:
                return int(w.shape[-1])

        raise RuntimeError(
            "Could not resolve Janus VQ latent channels. Set "
            "JanusProConfig.vq_latent_channels or provide a VQ model with "
            "quantize.embedding.weight."
        )


# ---------------------------------------------------------------------------
# Loader — lazy import so this module is importable without the janus pkg.
# ---------------------------------------------------------------------------


def _load_janus_from_pretrained(config: JanusProConfig) -> tuple[Any, Any]:
    """Load ``MultiModalityCausalLM`` + ``VLChatProcessor`` from disk/HF.

    The ``janus`` package (``deepseek-ai/Janus`` on GitHub, NOT PyPI)
    must be installed:

        git clone https://github.com/deepseek-ai/Janus
        cd Janus && pip install -e .
    """
    try:
        from janus.models import MultiModalityCausalLM, VLChatProcessor
    except ImportError as e:
        raise ImportError(
            "Cannot import deepseek-ai/Janus. Install via:\n"
            "  git clone https://github.com/deepseek-ai/Janus\n"
            "  cd Janus && pip install -e .\n"
            "(The PyPI package called 'janus' is unrelated — it's an "
            "asyncio queue library.)"
        ) from e

    from transformers import AutoModelForCausalLM

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[config.dtype]

    processor = VLChatProcessor.from_pretrained(config.model_path)
    mmgpt = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=dtype,
    )
    assert isinstance(mmgpt, MultiModalityCausalLM), (
        f"Loaded model {type(mmgpt).__name__} is not MultiModalityCausalLM"
    )
    mmgpt = mmgpt.to(device=config.device, dtype=dtype).eval()
    return mmgpt, processor
