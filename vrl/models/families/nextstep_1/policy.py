"""NextStep-1 wrapper for autoregressive image RL with continuous tokens.

Mirrors ``vrl.models.families.janus_pro.JanusProPolicy`` but for StepFun's
continuous-token AR model. The shape contract is:

  * ``sample_image_tokens(...)`` →
        (continuous_tokens [B, L, D_token],
         saved_noise        [B, L, D_token],   # x_0 prior, for replay
         old_logprobs       [B, L])            # Gaussian per-token log-prob

  * ``recompute_logprobs(...)`` →
        fresh_logprobs       [B, L]            # under current policy

  * ``decode_image_tokens(...)`` → pixels [B, 3, H, W]

  * ``disable_adapter()`` — context manager for the LoRA-off ref pass

The "logits" abstraction does not apply: tokens are continuous so we
work with per-token Gaussian log-probs directly. The ``OnlineTrainer``
+ ``TokenGRPO`` pipeline is shape-agnostic (it only sees ``[B, L]``
log-prob tensors), so no trainer-side change is required.

UPSTREAM BINDING
================
This module is a *scaffolding* — every real call into the upstream
NextStep-1 package is marked ``# TODO(nextstep-binding)``. Once you've
done ``pip install -e .`` from ``stepfun-ai/NextStep-1``, fill in:
    - ``_load_pipeline``     : how the upstream pipeline is constructed
    - ``_run_llm_step``      : single-token LLM forward returning hidden
    - ``_image_in_projector``: continuous-token → LLM-hidden projection
    - ``_decode_via_vae``    : token sequence → pixels via the f8ch16 VAE

The flow head's velocity-call signature is handled in
``vrl.models.families.nextstep_1.flow_step``.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from vrl.models.ar import (
    ARStepResult,
    AutoregressivePolicy,
    ar_concat_rows,
    ar_split_rows,
)
from vrl.models.families.nextstep_1.flow_step import (
    flow_logprob_at,
    flow_sample_with_logprob,
)

logger = logging.getLogger(__name__)


# NextStep-1 image grid: 32x32 continuous patches at f8ch16 = 16-channel,
# 8x downsample VAE (per the model card). Override via config if you load
# a different checkpoint that uses a different grid.
NEXTSTEP_DEFAULT_TOKEN_NUM = 1024     # 32 x 32 patches per 256^2 image
NEXTSTEP_DEFAULT_TOKEN_DIM = 64       # latent_patch_size^2 * f8ch16 channels
NEXTSTEP_DEFAULT_PIXEL_SIZE = 256


@dataclass(slots=True)
class NextStep1Config:
    """Hyper-parameters for the NextStep-1 wrapper.

    Defaults target ``stepfun-ai/NextStep-1.1`` — the RL-post-trained
    14B variant — paired with the f8ch16 VAE tokenizer.
    """

    model_path: str = "stepfun-ai/NextStep-1.1"
    vae_path: str = "stepfun-ai/NextStep-1-f8ch16-Tokenizer"
    dtype: str = "bfloat16"
    device: str = "cuda"

    # LoRA — applied to the LLM trunk (the 14B AR transformer)
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    # NextStep-1's LLM is Qwen-derived; same names as Qwen-2 attention
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
    )
    lora_init: str = "gaussian"

    # Flow-head sampling — used by sample_image_tokens
    num_flow_steps: int = 20             # K Euler steps inside the flow ODE
    noise_level: float = 1.0             # final-step Gaussian std multiplier
    cfg_scale: float = 4.5               # CFG strength on the velocity field

    # AR loop
    image_token_num: int = NEXTSTEP_DEFAULT_TOKEN_NUM
    token_dim: int = NEXTSTEP_DEFAULT_TOKEN_DIM
    image_size: int = NEXTSTEP_DEFAULT_PIXEL_SIZE

    # Frozen sub-modules
    freeze_vae: bool = True
    freeze_image_head: bool = False     # train the 157M flow head with LoRA-style updates

    # Memory
    gradient_checkpointing: bool = True


@dataclass(slots=True)
class NextStep1ARState:
    """Mutable per-row state for one scheduled NextStep AR sampling loop."""

    kv_cond_rows: list[Any]
    kv_uncond_rows: list[Any] | None
    c_cond: torch.Tensor
    c_uncond: torch.Tensor | None
    tokens: torch.Tensor
    saved_noise: torch.Tensor
    logprobs: torch.Tensor
    cfg_scale: float
    num_flow_steps: int
    noise_level: float
    image_token_num: int
    generator: torch.Generator | None = None
    position: int = 0
    positions: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class NextStep1Policy(nn.Module, AutoregressivePolicy):
    """Continuous-token AR T2I wrapper for the GRPO trainer.

    Composes:
      * ``self._pipeline``       : upstream ``NextStepPipeline`` (lazy-loaded)
      * ``self.language_model``  : the LLM trunk (LoRA target)
      * ``self.image_head``      : the 157M flow-matching MLP head
      * ``self.image_in_projector``: continuous-token → LLM-hidden projection
      * ``self.vae``             : f8ch16 VAE for decode
      * ``self.processor``       : tokenizer + chat-template
    """

    def __init__(self, config: NextStep1Config) -> None:
        super().__init__()
        self.config = config

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[config.dtype]
        self.dtype = torch_dtype
        self._device = torch.device(config.device)

        self._pipeline = self._load_pipeline()

        # Upstream NextStepModel inherits Qwen2Model + NextStepMixin —
        # i.e. the AR transformer trunk and the image head/projector live
        # on the same object. There is no separate ``.llm`` attribute.
        # We treat the whole NextStepModel as ``language_model`` so PEFT
        # can attach to its Qwen2 attention modules.
        self.language_model = self._pipeline.model
        self.image_head = self._pipeline.model.image_head
        self._image_in_projector = self._pipeline.model.image_in_projector
        self._image_out_projector = self._pipeline.model.image_out_projector
        self.vae = self._pipeline.vae
        self.processor = self._pipeline.tokenizer  # AutoTokenizer (naming aligns with Janus)
        self.config.token_dim = int(
            getattr(self.image_head, "input_dim", self.config.token_dim),
        )

        # Freeze what shouldn't be trained.
        if config.freeze_vae:
            for p in self.vae.parameters():
                p.requires_grad_(False)

        if config.use_lora:
            self._attach_lora()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_pipeline(self) -> Any:
        """Instantiate the upstream NextStep-1 pipeline.

        Upstream layout: ``inference/`` is a top-level directory in the
        cloned repo (NOT a submodule of ``nextstep``), and its
        ``gen_pipeline.py`` uses bare ``from gen_pipeline import ...``
        imports — so we have to put that directory on ``sys.path`` before
        importing.
        """
        import os
        import sys

        try:
            import nextstep  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "NextStep-1 wrapper requires `stepfun-ai/NextStep-1`. Install with:\n"
                "    git clone https://github.com/stepfun-ai/NextStep-1\n"
                "    cd NextStep-1 && pip install -e ."
            ) from e

        # NextStep ships its inference pipeline outside the package, so we
        # locate the cloned repo via the ``nextstep/`` package's parent.
        repo_root = os.path.dirname(os.path.dirname(nextstep.__file__))
        inference_dir = os.path.join(repo_root, "inference")
        if inference_dir not in sys.path:
            sys.path.insert(0, inference_dir)

        from gen_pipeline import NextStepPipeline  # type: ignore[import-not-found]

        return NextStepPipeline(
            model_name_or_path=self.config.model_path,
            vae_name_or_path=self.config.vae_path,
            device=str(self.device),
            dtype=self.dtype,
            enable_gradient_checkpointing=self.config.gradient_checkpointing,
        )

    def _attach_lora(self) -> None:
        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=list(self.config.lora_target_modules),
            init_lora_weights=self.config.lora_init,
        )
        # The whole NextStepModel becomes the PEFT-wrapped module. Since
        # the pipeline holds the model by reference, the upstream
        # ``decoding()`` path automatically sees the LoRA'd weights.
        self.language_model = get_peft_model(self.language_model, lora_cfg)
        self._pipeline.model = self.language_model

    # ------------------------------------------------------------------
    # Public: trainable param count
    # ------------------------------------------------------------------

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def device(self) -> torch.device:
        """Device where the upstream NextStep pipeline is loaded."""
        return self._device

    # ------------------------------------------------------------------
    # Public: AR sampling with per-token log-probabilities
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_image_tokens(
        self,
        prompt_embeds: torch.Tensor,         # [B, L_text, D_hidden]
        uncond_embeds: torch.Tensor | None,  # [B, L_text, D_hidden] or None
        prompt_mask: torch.Tensor,           # [B, L_text]
        uncond_mask: torch.Tensor | None,    # [B, L_text]
        *,
        cfg_scale: float | None = None,
        num_flow_steps: int | None = None,
        noise_level: float | None = None,
        image_token_num: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the AR loop and return (tokens, saved_noise, log_probs).

        Args:
            prompt_embeds:  text-prompt embeddings (caller invokes
                            ``self.language_model.get_input_embeddings()``)
            uncond_embeds:  unconditional embeddings for CFG; pass ``None`` to
                            skip CFG.
            prompt_mask:    attention mask for ``prompt_embeds``.
            uncond_mask:    attention mask for ``uncond_embeds`` (or None).

        Returns:
            tokens         ``[B, L_img, D_token]`` — continuous tokens.
            saved_noise    ``[B, L_img, D_token]`` — flow ODE prior x_0 used
                           per token. Stash so ``recompute_logprobs`` can
                           replay the same trajectory deterministically.
            log_probs      ``[B, L_img]`` — Gaussian log-prob of each token.
        """
        state = self.init_ar_state(
            prompt_embeds,
            uncond_embeds,
            prompt_mask,
            uncond_mask,
            cfg_scale=cfg_scale,
            num_flow_steps=num_flow_steps,
            noise_level=noise_level,
            image_token_num=image_token_num,
            generator=generator,
        )
        while state.position < state.image_token_num:
            self._sample_ar_step(state)
        return self.finalize_ar_state(state)

    @torch.no_grad()
    def init_ar_state(
        self,
        prompt_embeds: torch.Tensor,
        uncond_embeds: torch.Tensor | None,
        prompt_mask: torch.Tensor,
        uncond_mask: torch.Tensor | None,
        *,
        cfg_scale: float | None = None,
        num_flow_steps: int | None = None,
        noise_level: float | None = None,
        image_token_num: int | None = None,
        generator: torch.Generator | None = None,
    ) -> NextStep1ARState:
        """Initialize full-row AR state for scheduled executor sampling."""
        cfg = self.config
        cfg_scale = cfg_scale if cfg_scale is not None else cfg.cfg_scale
        num_flow_steps = (
            num_flow_steps if num_flow_steps is not None else cfg.num_flow_steps
        )
        noise_level = noise_level if noise_level is not None else cfg.noise_level
        image_token_num = (
            image_token_num if image_token_num is not None else cfg.image_token_num
        )

        batch_size = prompt_embeds.shape[0]
        token_dim = cfg.token_dim
        device = prompt_embeds.device

        tokens = torch.zeros(
            batch_size, image_token_num, token_dim, device=device, dtype=self.dtype
        )
        saved_noise = torch.zeros(
            batch_size, image_token_num, token_dim, device=device, dtype=self.dtype
        )
        logprobs = torch.zeros(
            batch_size, image_token_num, device=device, dtype=torch.float32
        )

        kv_cond = self._init_kv(prompt_embeds, prompt_mask)
        kv_uncond = (
            self._init_kv(uncond_embeds, uncond_mask)
            if uncond_embeds is not None else None
        )
        c_cond = self._last_hidden(kv_cond)
        c_uncond = self._last_hidden(kv_uncond) if kv_uncond is not None else None

        return NextStep1ARState(
            kv_cond_rows=ar_split_rows(kv_cond, batch_size),
            kv_uncond_rows=ar_split_rows(kv_uncond, batch_size)
            if kv_uncond is not None else None,
            c_cond=c_cond,
            c_uncond=c_uncond,
            tokens=tokens,
            saved_noise=saved_noise,
            logprobs=logprobs,
            cfg_scale=float(cfg_scale),
            num_flow_steps=int(num_flow_steps),
            noise_level=float(noise_level),
            image_token_num=int(image_token_num),
            generator=generator,
            positions=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    @torch.no_grad()
    def step_ar(
        self,
        state: NextStep1ARState,
        sequences: list[Any],
        *,
        generator: torch.Generator | None = None,
    ) -> ARStepResult:
        """Run one scheduled AR token step for rows at the same position."""
        if not sequences:
            raise ValueError("step_ar requires at least one ActiveSequence")

        row_indices = [int(seq.metadata.get("row_index", -1)) for seq in sequences]
        if any(row < 0 or row >= state.tokens.shape[0] for row in row_indices):
            raise ValueError(f"invalid NextStep row indices: {row_indices}")

        positions = [int(seq.position) for seq in sequences]
        if len(set(positions)) != 1:
            raise ValueError("ActiveSequence positions must match within one AR step")
        if state.positions is None:
            raise ValueError("NextStep1ARState.positions is required")
        expected_positions = [
            int(state.positions[row].item()) for row in row_indices
        ]
        if positions != expected_positions:
            raise ValueError(
                "ActiveSequence positions must match NextStep1ARState row positions"
            )

        step = self._sample_ar_step(
            state,
            row_indices=row_indices,
            position=positions[0],
            generator=generator,
        )
        return ARStepResult(
            sequence_ids=[str(seq.sample_id) for seq in sequences],
            positions=positions,
            token=step.token,
            log_prob=step.log_prob.float(),
            replay_extras={"saved_noise": step.initial_noise},
        )

    @torch.no_grad()
    def finalize_ar_state(
        self,
        state: NextStep1ARState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return sampled tokens, replay noise, and old log-probs."""
        return state.tokens, state.saved_noise, state.logprobs

    def _sample_ar_step(
        self,
        state: NextStep1ARState,
        *,
        row_indices: list[int] | None = None,
        position: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Any:
        if state.positions is None:
            raise ValueError("NextStep1ARState.positions is required")

        if row_indices is None:
            row_indices = list(range(state.tokens.shape[0]))
        if not row_indices:
            raise ValueError("row_indices must be non-empty")
        if any(row < 0 or row >= state.tokens.shape[0] for row in row_indices):
            raise ValueError(f"invalid NextStep row indices: {row_indices}")

        row_positions = [int(state.positions[row].item()) for row in row_indices]
        if len(set(row_positions)) != 1:
            raise ValueError("NextStep rows in one AR step must share a position")
        position = row_positions[0] if position is None else int(position)
        if any(pos != position for pos in row_positions):
            raise ValueError("requested position does not match row positions")
        if position >= state.image_token_num:
            raise ValueError("NextStep1ARState has already finished sampling")

        step_generator = generator if generator is not None else state.generator
        batch_size = len(row_indices)
        token_dim = state.tokens.shape[-1]
        device = state.tokens.device
        rows = torch.tensor(row_indices, device=device, dtype=torch.long)
        initial_noise = torch.randn(
            batch_size,
            token_dim,
            device=device,
            dtype=self.dtype,
            generator=step_generator,
        )
        c_cond = state.c_cond.index_select(0, rows)
        c_uncond = (
            state.c_uncond.index_select(0, rows)
            if state.c_uncond is not None else None
        )
        step = flow_sample_with_logprob(
            self.image_head,
            cond=c_cond,
            num_flow_steps=state.num_flow_steps,
            noise_level=state.noise_level,
            cfg_uncond=c_uncond,
            cfg_scale=state.cfg_scale,
            generator=step_generator,
            initial_noise=initial_noise,
        )

        state.tokens[rows, position] = step.token
        state.saved_noise[rows, position] = step.initial_noise
        state.logprobs[rows, position] = step.log_prob.float()

        proj = self._image_in_projector(step.token)
        kv_cond = ar_concat_rows([state.kv_cond_rows[row] for row in row_indices])
        kv_cond, c_cond_next = self._step_llm(kv_cond, proj)
        for row, row_kv in zip(row_indices, ar_split_rows(kv_cond, batch_size), strict=True):
            state.kv_cond_rows[row] = row_kv
        state.c_cond.index_copy_(0, rows, c_cond_next)

        if state.kv_uncond_rows is not None:
            proj_u = self._image_in_projector(step.token)
            kv_uncond = ar_concat_rows(
                [state.kv_uncond_rows[row] for row in row_indices]
            )
            kv_uncond, c_uncond_next = self._step_llm(kv_uncond, proj_u)
            for row, row_kv in zip(
                row_indices,
                ar_split_rows(kv_uncond, batch_size),
                strict=True,
            ):
                state.kv_uncond_rows[row] = row_kv
            assert state.c_uncond is not None
            state.c_uncond.index_copy_(0, rows, c_uncond_next)

        state.positions[rows] += 1
        state.position = int(state.positions.min().item())
        return step

    # ------------------------------------------------------------------
    # Public: training-time log-prob recomputation
    # ------------------------------------------------------------------

    def recompute_logprobs(
        self,
        prompt_embeds: torch.Tensor,         # [B, L_text, D_hidden]
        uncond_embeds: torch.Tensor | None,
        prompt_mask: torch.Tensor,
        uncond_mask: torch.Tensor | None,
        tokens: torch.Tensor,                # [B, L_img, D_token]
        saved_noise: torch.Tensor,           # [B, L_img, D_token]
        *,
        cfg_scale: float | None = None,
        num_flow_steps: int | None = None,
        noise_level: float | None = None,
    ) -> torch.Tensor:
        """Re-compute fresh per-token log-probs under the current policy.

        Returns ``[B, L_img]`` log-probs with grad through ``image_head``
        and (if LoRA is attached) through the LLM as well.
        """
        cfg = self.config
        cfg_scale = cfg_scale if cfg_scale is not None else cfg.cfg_scale
        num_flow_steps = num_flow_steps if num_flow_steps is not None else cfg.num_flow_steps
        noise_level = noise_level if noise_level is not None else cfg.noise_level

        B, L_img, _ = tokens.shape

        # Re-prime the LLM so its hidden states reflect the current LoRA'd
        # parameters. Same path as sampling but with grad enabled.
        kv_cond = self._init_kv(prompt_embeds, prompt_mask)
        kv_uncond = (
            self._init_kv(uncond_embeds, uncond_mask)
            if uncond_embeds is not None else None
        )
        c_cond = self._last_hidden(kv_cond)
        c_uncond = self._last_hidden(kv_uncond) if kv_uncond is not None else None

        out = torch.zeros(B, L_img, device=tokens.device, dtype=torch.float32)
        for j in range(L_img):
            lp = flow_logprob_at(
                self.image_head,
                cond=c_cond,
                target_token=tokens[:, j],
                saved_noise=saved_noise[:, j],
                num_flow_steps=num_flow_steps,
                noise_level=noise_level,
                cfg_uncond=c_uncond,
                cfg_scale=cfg_scale,
            )
            out[:, j] = lp.float()

            proj = self._image_in_projector(tokens[:, j])
            kv_cond, c_cond = self._step_llm(kv_cond, proj)
            if kv_uncond is not None:
                proj_u = self._image_in_projector(tokens[:, j])
                kv_uncond, c_uncond = self._step_llm(kv_uncond, proj_u)

        return out

    # ------------------------------------------------------------------
    # Replay forward — AutoregressivePolicy contract
    # ------------------------------------------------------------------

    def replay_forward(
        self,
        batch: Any,
        timestep_idx: int = 0,
    ) -> dict[str, Any]:
        """Re-run the AR loop and return per-token log-probs.

        Train-time replay for ``ContinuousTokenLogProbEvaluator``: reads
        ``observations`` (text prompt ids), ``extras["prompt_attention_mask"]``,
        the optional ``extras["uncond_input_ids"]``/``extras["uncond_attention_mask"]``,
        ``actions`` (sampled continuous tokens), and ``extras["saved_noise"]``;
        returns log-probs of the sampled tokens under the current policy.

        Differs from Janus's ``replay_forward`` (which returns logits) — for
        continuous tokens we go straight to log-probs since there is no
        codebook to softmax over.

        ``timestep_idx`` accepted for protocol compatibility but ignored —
        AR has no notion of "denoising step".

        Returns:
          ``{"log_probs": Tensor[B, L_img], "tokens": Tensor[B, L_img, D_token]}``.
        """
        del timestep_idx
        obs = batch.observations
        prompt_ids = obs.squeeze(1) if obs.dim() == 3 else obs
        prompt_mask = batch.extras["prompt_attention_mask"]
        uncond_ids = batch.extras.get("uncond_input_ids")
        uncond_mask = batch.extras.get("uncond_attention_mask")
        tokens = batch.actions
        saved_noise = batch.extras["saved_noise"]

        embed = self.language_model.get_input_embeddings()
        prompt_embeds = embed(prompt_ids)
        uncond_embeds = embed(uncond_ids) if uncond_ids is not None else None

        log_probs = self.recompute_logprobs(
            prompt_embeds, uncond_embeds, prompt_mask, uncond_mask,
            tokens=tokens, saved_noise=saved_noise,
            cfg_scale=batch.context.get("cfg_scale"),
            num_flow_steps=batch.context.get("num_flow_steps"),
            noise_level=batch.context.get("noise_level"),
        )
        return {"log_probs": log_probs, "tokens": tokens}

    # ------------------------------------------------------------------
    # Public: decode tokens → pixels
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode_image_tokens(
        self,
        tokens: torch.Tensor,        # [B, L_img, D_token]
        image_size: int | None = None,
    ) -> torch.Tensor:
        """Continuous tokens → pixels in ``[-1, 1]`` via the f8ch16 VAE."""
        del image_size
        side = int(tokens.shape[1] ** 0.5)
        if side * side != tokens.shape[1]:
            raise ValueError(
                f"image_token_num must be a square grid, got {tokens.shape[1]}",
            )
        latent = self._pipeline.model.unpatchify(tokens, h=side, w=side)
        latent = (
            latent / self._pipeline.scaling_factor
        ) + self._pipeline.shift_factor
        decoded = self.vae.decode(latent.to(self.vae.dtype))
        pixels = decoded.sample if hasattr(decoded, "sample") else decoded[0]
        return pixels.to(torch.float32)

    # ------------------------------------------------------------------
    # Public: ref-policy hook
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def disable_adapter(self) -> Iterator[None]:
        """Run a forward pass with LoRA disabled (= reference policy)."""
        if hasattr(self.language_model, "disable_adapter"):
            with self.language_model.disable_adapter():
                yield
        else:
            yield

    # ------------------------------------------------------------------
    # Internal: LLM step / KV plumbing
    # ------------------------------------------------------------------

    def _init_kv(
        self,
        embeds: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> Any:
        """Prime the LLM with text-prompt embeddings, return a KV-cache handle.

        TODO(nextstep-binding): the actual KV-cache type depends on the
        underlying ``transformers`` model class (Qwen-2). For HF this is a
        ``DynamicCache``. The pipeline's ``decoding()`` method already does
        this — we'll piggyback once we wire the binding.
        """
        out = self.language_model(
            inputs_embeds=embeds,
            attention_mask=mask,
            use_cache=True,
            output_hidden_states=True,
        )
        return {
            "past_key_values": out.past_key_values,
            "last_hidden": out.hidden_states[-1][:, -1],  # [B, D_hidden]
        }

    @staticmethod
    def _last_hidden(kv: Any) -> torch.Tensor:
        return kv["last_hidden"]

    def _step_llm(
        self,
        kv: Any,
        new_embed: torch.Tensor,         # [B, D_hidden]
    ) -> tuple[Any, torch.Tensor]:
        """One-token LLM forward; returns updated kv + new last hidden."""
        out = self.language_model(
            inputs_embeds=new_embed.unsqueeze(1),
            past_key_values=kv["past_key_values"],
            use_cache=True,
            output_hidden_states=True,
        )
        kv2 = {
            "past_key_values": out.past_key_values,
            "last_hidden": out.hidden_states[-1][:, -1],
        }
        return kv2, kv2["last_hidden"]
