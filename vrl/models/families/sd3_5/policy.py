"""SD 3.5 t2i diffusers adapter (DiffusionPolicy contract).

Single-protocol adapter for Stable Diffusion 3.5-Medium image generation.
The contract is:

    encode_prompt -> prepare_sampling -> forward_step xN -> decode_latents

The collector (or default ``DiffusionPolicy.inference`` loop) owns the
scheduler step / SDE step. ``forward_step`` does only one transformer
forward (with optional batched CFG concat) and returns noise predictions.

Per-family ``SD3SamplingState`` is private to this file — engine /
collector code MUST NOT introspect it beyond the documented attributes
(``latents``, ``timesteps``, ``scheduler``, plus the embeds the eval path
re-builds explicitly).

Timestep shape convention used by ``forward_step``:
- During rollouts ``timesteps`` is a 1-D tensor ``[T]`` of scheduler
  timesteps; ``state.timesteps[step_idx]`` is a scalar that we expand to
  ``[B]`` for the transformer call.
- During eval/training the collector pre-builds a ``SD3SamplingState``
  whose ``timesteps`` is a ``[1, B]`` tensor (per-sample timestep at the
  selected denoise step) and calls ``forward_step(state, 0, ...)``;
  ``state.timesteps[0]`` is then ``[B]`` and ``forward_step``'s
  ``t.expand(bsz)`` is a no-op (because the source already has shape
  ``[B]`` — ``Tensor.expand`` accepts equal sizes).
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import Any

import torch

from vrl.models.diffusion import DiffusionPolicy, VideoGenerationRequest


@dataclass
class SD3SamplingState:
    """Private SD3 sampling state. Engine MUST NOT introspect."""

    latents: torch.Tensor
    timesteps: torch.Tensor
    scheduler: Any
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    negative_pooled_prompt_embeds: torch.Tensor | None
    guidance_scale: float
    do_cfg: bool
    seed: int


class SD3_5Policy(DiffusionPolicy):
    """Diffusers-backed SD 3.5 t2i adapter."""

    family = "sd3_5-diffusers-t2i"

    def __init__(self, *, pipeline: Any, device: Any = None) -> None:
        super().__init__()
        object.__setattr__(self, "_pipeline", pipeline)
        self.transformer = pipeline.transformer
        self._device = device

    @property
    def pipeline(self) -> Any:
        return self._pipeline

    def _set_transformer(self, transformer: Any) -> None:
        self.transformer = transformer
        self.pipeline.transformer = transformer

    @property
    def device(self) -> Any:
        return self._device if self._device is not None else self.pipeline.device

    def describe(self) -> dict[str, Any]:
        return {"family": self.family, "device": str(self.device)}

    # -- backend ownership (called by builder, not by collectors) -------

    @classmethod
    def from_spec(cls, spec: Any) -> SD3_5Policy:
        """Load the diffusers SD3.5 pipeline + freeze non-trainable modules."""
        from diffusers import StableDiffusion3Pipeline

        pipeline = StableDiffusion3Pipeline.from_pretrained(
            spec.model_name_or_path, torch_dtype=spec.dtype,
        )
        pipeline.vae.requires_grad_(False)
        for enc in (
            pipeline.text_encoder,
            pipeline.text_encoder_2,
            pipeline.text_encoder_3,
        ):
            if enc is not None:
                enc.requires_grad_(False)
                enc.to(spec.device, dtype=spec.dtype)
        pipeline.vae.to(spec.device, dtype=torch.float32)
        return cls(pipeline=pipeline, device=spec.device)

    def apply_lora(self, spec: Any) -> None:
        """Wrap the SD3 transformer with PEFT LoRA per spec.lora_*."""
        from peft import LoraConfig, PeftModel, get_peft_model

        self.pipeline.transformer.requires_grad_(False)
        self.pipeline.transformer.to(self.device)

        if spec.lora_path:
            transformer = PeftModel.from_pretrained(
                self.pipeline.transformer, spec.lora_path, is_trainable=True,
            )
            transformer.set_adapter("default")
            self._set_transformer(transformer)
        else:
            assert spec.lora_config is not None
            cfg = LoraConfig(
                r=spec.lora_config["rank"],
                lora_alpha=spec.lora_config["alpha"],
                init_lora_weights="gaussian",
                target_modules=spec.lora_config["target_modules"],
            )
            self._set_transformer(
                get_peft_model(self.pipeline.transformer, cfg),
            )

    def enable_full_finetune(self) -> None:
        """Mark transformer fully trainable (no-LoRA path)."""
        self.pipeline.transformer.requires_grad_(True)
        self.pipeline.transformer.to(self.device)

    def torch_compile_transformer(self, mode: str) -> None:
        """Apply torch.compile to the transformer in-place."""
        self._set_transformer(
            torch.compile(self.pipeline.transformer, mode=mode, fullgraph=False),
        )

    def set_num_steps(self, n: int) -> None:
        """Initialize the scheduler timesteps for sampling."""
        self.pipeline.scheduler.set_timesteps(n, device=self.device)

    @property
    def trainable_modules(self) -> dict[str, Any]:
        return {"transformer": self.transformer}

    @property
    def scheduler(self) -> Any:
        return self.pipeline.scheduler

    @property
    def backend_handle(self) -> Any:
        return self.pipeline

    # -- encode_prompt -------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode prompt via SD3's three text encoders (T5 + 2x CLIP).

        Returns prompt_embeds (joint T5+CLIP sequence), pooled_prompt_embeds
        (CLIP pooled), and their negative counterparts when CFG is active.
        """
        max_seq = kwargs.get("max_sequence_length", 128)
        guidance_scale = kwargs.get("guidance_scale", 4.5)
        do_cfg = guidance_scale > 1.0
        neg = negative_prompt if negative_prompt is not None else ""

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            negative_prompt=neg,
            negative_prompt_2=neg,
            negative_prompt_3=neg,
            do_classifier_free_guidance=do_cfg,
            num_images_per_prompt=1,
            max_sequence_length=max_seq,
            device=self.device,
        )

        td = self.pipeline.transformer.dtype
        prompt_embeds = prompt_embeds.to(td)
        pooled_prompt_embeds = pooled_prompt_embeds.to(td)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(td)
        if negative_pooled_prompt_embeds is not None:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(td)

        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
        }

    # -- prepare_sampling ----------------------------------------------

    def prepare_sampling(
        self,
        request: VideoGenerationRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> SD3SamplingState:
        """Build the per-request SamplingState for a denoise loop."""
        pipe = self.pipeline
        device = self.device

        prompt_embeds = encoded["prompt_embeds"]
        pooled_prompt_embeds = encoded["pooled_prompt_embeds"]
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")
        negative_pooled_prompt_embeds = encoded.get("negative_pooled_prompt_embeds")

        pipe.scheduler.set_timesteps(request.num_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        seed = (
            request.seed if request.seed is not None
            else random.randint(0, sys.maxsize)
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        num_channels_latents = pipe.transformer.config.in_channels
        batch_size = prompt_embeds.shape[0]
        # SD3 prepare_latents: (batch, channels, height, width, dtype, device, generator, latents)
        latents = pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            request.height,
            request.width,
            torch.float32,
            device,
            generator,
            None,
        )

        do_cfg = request.guidance_scale > 1.0

        return SD3SamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=pipe.scheduler,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            guidance_scale=request.guidance_scale,
            do_cfg=do_cfg,
            seed=seed,
        )

    # -- forward_step --------------------------------------------------

    def forward_step(
        self,
        state: SD3SamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """SD3 transformer forward + optional batched CFG.

        Returns noise_pred plus the un/conditional branches; the caller owns
        scheduler.step / SDE.
        """
        m = self.transformer

        t = state.timesteps[step_idx]
        bsz = state.latents.shape[0]
        td = state.prompt_embeds.dtype

        latent_input = state.latents.to(td)
        # SD3 timestep is broadcast across batch as the raw float (not /1000).
        # If t is already shape [B] (eval path packs timesteps as [1, B]),
        # Tensor.expand(bsz) is a no-op on the equal-sized dim.
        timestep_batch = t.expand(bsz) if t.ndim == 0 else t

        if state.do_cfg:
            combined_latents = torch.cat([latent_input, latent_input], dim=0)
            combined_t = torch.cat([timestep_batch, timestep_batch], dim=0)
            combined_embeds = torch.cat(
                [state.negative_prompt_embeds, state.prompt_embeds], dim=0,
            )
            combined_pooled = torch.cat(
                [state.negative_pooled_prompt_embeds, state.pooled_prompt_embeds],
                dim=0,
            )
            combined_out = m(
                hidden_states=combined_latents,
                timestep=combined_t,
                encoder_hidden_states=combined_embeds,
                pooled_projections=combined_pooled,
                return_dict=False,
            )[0]
            noise_pred_uncond, noise_pred_cond = combined_out.chunk(2, dim=0)
            noise_pred_uncond = noise_pred_uncond.to(td)
            noise_pred_cond = noise_pred_cond.to(td)
            noise_pred = noise_pred_uncond + state.guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
        else:
            noise_pred_cond = m(
                hidden_states=latent_input,
                timestep=timestep_batch,
                encoder_hidden_states=state.prompt_embeds,
                pooled_projections=state.pooled_prompt_embeds,
                return_dict=False,
            )[0].to(td)
            noise_pred_uncond = torch.zeros_like(noise_pred_cond)
            noise_pred = noise_pred_cond

        return {
            "noise_pred": noise_pred,
            "noise_pred_cond": noise_pred_cond,
            "noise_pred_uncond": noise_pred_uncond,
        }

    # -- collector boundary --------------------------------------------

    def export_batch_context(self, state: SD3SamplingState) -> dict[str, Any]:
        """Project SD3 sampling state into RolloutBatch.context."""
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
            "model_family": self.family,
        }

    def export_training_extras(self, state: SD3SamplingState) -> dict[str, Any]:
        """Project SD3 sampling state into RolloutBatch.extras."""
        return {
            "prompt_embeds": state.prompt_embeds,
            "pooled_prompt_embeds": state.pooled_prompt_embeds,
            "negative_prompt_embeds": state.negative_prompt_embeds,
            "negative_pooled_prompt_embeds": state.negative_pooled_prompt_embeds,
        }

    def restore_eval_state(
        self,
        batch_extras: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> SD3SamplingState:
        """Rebuild SD3SamplingState from a batch slice for the eval forward path.

        Packs timesteps as ``[1, B]`` so ``state.timesteps[0]`` is ``[B]`` —
        matches the eval-path convention documented in the class docstring.
        """
        ts = batch_extras["timesteps"]
        t = ts[:, step_idx] if ts.ndim > 1 else ts  # [B]
        timesteps = t.unsqueeze(0) if t.ndim == 1 else t  # pack as [1, B]
        return SD3SamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=None,  # not needed for forward_step (no scheduler.step here)
            prompt_embeds=batch_extras["prompt_embeds"],
            pooled_prompt_embeds=batch_extras["pooled_prompt_embeds"],
            negative_prompt_embeds=batch_extras.get("negative_prompt_embeds"),
            negative_pooled_prompt_embeds=batch_extras.get(
                "negative_pooled_prompt_embeds",
            ),
            guidance_scale=batch_context["guidance_scale"],
            do_cfg=batch_context["cfg"] and batch_context["guidance_scale"] > 1.0,
            seed=0,
        )

    # -- decode_latents ------------------------------------------------

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents → image via SD3 VAE (4D, no T dim)."""
        pipe = self.pipeline
        x = latents.to(pipe.vae.dtype)
        scaling_factor = pipe.vae.config.scaling_factor
        shift_factor = getattr(pipe.vae.config, "shift_factor", 0.0) or 0.0
        # SD3 VAE: latents = (z - shift) * scale → invert.
        x = x / scaling_factor + shift_factor
        image = pipe.vae.decode(x, return_dict=False)[0]
        # postprocess to [0, 1] float tensor [B, C, H, W]
        return pipe.image_processor.postprocess(image, output_type="pt")
