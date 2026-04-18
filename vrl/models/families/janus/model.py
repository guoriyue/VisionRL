"""Janus-Pro-1B wrapper for autoregressive text-to-image RL.

This file isolates every Janus-specific detail (image-token vocab range,
``gen_head`` projection, CFG sampling, VQ decode) behind a small surface
that the generic GRPO trainer can call:

  * ``JanusProT2I.forward_image_logits(...)``
        Train-time forward — returns logits over the *image* vocab for
        each image-token position. Used by the evaluator to recompute
        new log-probs under the current policy.

  * ``JanusProT2I.sample_image_tokens(...)``
        Inference-time AR sampler with classifier-free guidance.
        Returns ``(image_token_ids, sampling_logprobs)`` — these
        log-probs are the ``old_logprob`` of GRPO.

  * ``JanusProT2I.decode_image_tokens(...)``
        Decode 24×24 image tokens → pixels via the frozen VQ model.

  * ``JanusProT2I.disable_adapter()``
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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:  # pragma: no cover
    from PIL import Image

logger = logging.getLogger(__name__)

# Janus-Pro-1B image-tokenizer constants (from deepseek-ai/Janus config)
JANUS_IMAGE_TOKEN_NUM = 576           # 24 × 24 latent grid per image
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
    # See JanusProT2I._base for why we can't use hasattr(base_model) as the key.
    inner = getattr(mmgpt, "base_model", None)
    if inner is not None and hasattr(inner, "model") and inner.model is not mmgpt:
        base = inner.model
    else:
        base = mmgpt
    return base.gen_head(hidden_states)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class JanusProT2I(nn.Module):
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
        ``self`` even when there's no PEFT wrap, and that shim has no ``.model``
        attr → AttributeError. Only peel when the PEFT inner path exists.
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
            getattr(lm, "disable_adapter")
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
                "JanusProT2I.disable_adapter() called but no PEFT adapter "
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
          ``logprobs`` is computed as ``log_softmax(guided_logits) ↑ id``.
        """
        cfg = cfg_weight if cfg_weight is not None else self.config.cfg_weight
        temp = temperature if temperature is not None else self.config.temperature
        L_img = image_token_num or self.config.image_token_num

        B = cond_inputs_embeds.shape[0]
        device = cond_inputs_embeds.device

        base = self._base()

        # Stack [cond ; uncond] along batch — single trunk call per step.
        embeds = torch.cat([cond_inputs_embeds, uncond_inputs_embeds], dim=0)
        attn = torch.cat([cond_attention_mask, uncond_attention_mask], dim=0)

        out_tokens = torch.empty(B, L_img, dtype=torch.long, device=device)
        out_logprobs = torch.empty(
            B, L_img, dtype=torch.float32, device=device,
        )

        past_kv = None
        cur_embeds = embeds
        cur_attn = attn

        for step in range(L_img):
            outputs = self._lm_trunk()(
                inputs_embeds=cur_embeds,
                attention_mask=cur_attn,
                use_cache=True,
                past_key_values=past_kv,
            )
            past_kv = outputs.past_key_values
            hidden = outputs.last_hidden_state[:, -1:, :]   # [2B, 1, H]

            logits = image_token_logits_from_hidden(self.mmgpt, hidden)
            logits = logits.squeeze(1)                       # [2B, V_img]

            cond_logits, uncond_logits = logits.chunk(2, dim=0)
            guided = uncond_logits + cfg * (cond_logits - uncond_logits)

            # Sample under guided distribution (this is what CFG changes).
            probs = F.softmax(guided / temp, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]

            # IMPORTANT: store log-prob under the *unguided* cond-only dist,
            # not the guided dist. At training time, forward_image_logits
            # runs only the conditional trunk (no CFG) — storing guided lp
            # here would create a ~constant offset between old_lp and new_lp
            # → ratio far from 1 at step 0, clip_fraction ~0.9, approx_kl ≫0.1.
            # Treat CFG as a sampling-time augmentation of the underlying
            # cond policy, and optimise that policy directly — this mirrors
            # the flow_grpo Wan convention.
            log_probs = F.log_softmax(cond_logits / temp, dim=-1)
            lp = log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

            out_tokens[:, step] = sampled
            out_logprobs[:, step] = lp

            # Build next-step embedding: same token to both cond + uncond.
            tok_doubled = torch.cat([sampled, sampled], dim=0)        # [2B]
            next_embed = base.prepare_gen_img_embeds(
                tok_doubled.unsqueeze(-1)
            )  # [2B, 1, H]
            cur_embeds = next_embed
            cur_attn = torch.cat(
                [
                    cur_attn,
                    torch.ones(
                        cur_attn.shape[0], 1,
                        dtype=cur_attn.dtype, device=device,
                    ),
                ],
                dim=1,
            )

        return out_tokens, out_logprobs

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

        Order of precedence:
          1. Explicit override (``config.vq_latent_channels``).
          2. Live introspection: ``vq_model.quantize.embedding.weight.shape[-1]``
             — the only authoritative source.
          3. Legacy config fallbacks (``embed_dim``, ``latent_channels``) —
             ``z_channels`` is explicitly skipped because it misleads.
          4. Janus-Pro-1B constant (8) with a debug log.
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

        vq_cfg = getattr(self.vq_model, "config", None)
        # z_channels is encoder-hidden-dim on Janus VQModel, NOT codebook dim.
        # Skip it intentionally.
        for attr in ("embed_dim", "latent_channels"):
            if vq_cfg is not None and hasattr(vq_cfg, attr):
                val = getattr(vq_cfg, attr)
                if not isinstance(val, int) or val <= 0:
                    raise RuntimeError(
                        f"vq_model.config.{attr} = {val!r} is not a positive "
                        "int — cannot use as latent-channel count."
                    )
                return val

        logger.debug(
            "Could not auto-detect vq latent channels from vq_model.config; "
            "falling back to 8 (Janus-Pro-1B default). Set "
            "JanusProConfig.vq_latent_channels to silence this."
        )
        return 8


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
