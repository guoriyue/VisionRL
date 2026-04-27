"""Diffusers SD3.5-Medium T2I model with step-level denoising runtime.

Mirrors ``vrl/models/families/wan/diffusers_t2v.py`` but for image generation:
- 4D latents [B, C, H, W] (no T dim)
- SD3 has joint prompt_embeds (T5+CLIP concat) AND pooled_prompt_embeds (CLIP pooled)
- Scheduler is FlowMatchEulerDiscreteScheduler (vs Wan's UniPC)
- SDE flow uses sde_type="cps" with paper a=0.7 (Flow-GRPO Eq.9)
"""

from __future__ import annotations

import gc
import random
import sys
from typing import Any

from vrl.models.base import ModelResult, VideoGenerationModel, VideoGenerationRequest


class DiffusersSD3T2IModel(VideoGenerationModel):
    """Diffusers-based SD3 T2I model (targets stabilityai/stable-diffusion-3.5-medium)."""

    model_family = "sd3-diffusers-t2i"

    def __init__(
        self,
        *,
        pipeline: Any,  # diffusers.StableDiffusion3Pipeline (already loaded)
        device: Any = None,
    ) -> None:
        self.pipeline = pipeline
        self._device = device

    @property
    def device(self) -> Any:
        if self._device is not None:
            return self._device
        return self.pipeline.device

    async def load(self) -> None:
        pass

    def describe(self) -> dict[str, Any]:
        return {
            "name": "sd3-diffusers-t2i-model",
            "family": self.model_family,
            "device": str(self.device),
        }

    # -- encode_text ---------------------------------------------------

    async def encode_text(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> ModelResult:
        """Encode prompt via SD3's three text encoders (T5 + 2× CLIP).

        Returns the joint sequence embed (prompt_embeds) and the CLIP
        pooled embed (pooled_prompt_embeds), each with their negative
        counterparts when CFG is active.
        """
        pipe = self.pipeline
        device = self.device

        do_cfg = request.guidance_scale > 1.0
        max_seq = request.extra.get("max_sequence_length", 256)

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=request.prompt,
            prompt_2=request.prompt,
            prompt_3=request.prompt,
            negative_prompt=request.negative_prompt or "",
            negative_prompt_2=request.negative_prompt or "",
            negative_prompt_3=request.negative_prompt or "",
            do_classifier_free_guidance=do_cfg,
            num_images_per_prompt=1,
            max_sequence_length=max_seq,
            device=device,
        )

        transformer_dtype = pipe.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)
        if negative_pooled_prompt_embeds is not None:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(
                transformer_dtype
            )

        return ModelResult(
            state_updates={
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
                "pipeline": pipe,
            },
            outputs={
                "prompt_tokens_estimate": max(1, len(request.prompt.split())),
            },
        )

    # -- step-level denoising for RL training --------------------------

    async def denoise_init(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> Any:
        """Set up per-step denoising state for SD3 T2I.

        Returns a ``DenoiseLoopState`` with ``DiffusersDenoiseState``.
        """
        import torch

        from vrl.engine.model_executor.execution_state import DenoiseLoopState
        from vrl.models.families.diffusers_state import DiffusersDenoiseState

        pipe = self.pipeline
        device = self.device

        prompt_embeds = state["prompt_embeds"]
        negative_prompt_embeds = state.get("negative_prompt_embeds")
        pooled_prompt_embeds = state["pooled_prompt_embeds"]
        negative_pooled_prompt_embeds = state.get("negative_pooled_prompt_embeds")

        guidance_scale = request.guidance_scale
        do_cfg = guidance_scale > 1.0

        pipe.scheduler.set_timesteps(request.num_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        num_channels_latents = pipe.transformer.config.in_channels
        batch_size = prompt_embeds.shape[0]

        seed = (
            request.seed if request.seed is not None
            else random.randint(0, sys.maxsize)
        )

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        # SD3 prepare_latents signature: (batch, channels, height, width, dtype, device, generator, latents)
        latents = pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            request.height,
            request.width,
            torch.float32,
            device,
            generator,
            None,  # latents
        )

        ms = DiffusersDenoiseState(
            latents=latents,
            timesteps=timesteps,
            scheduler=pipe.scheduler,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            guidance_scale=guidance_scale,
            do_cfg=do_cfg,
            pipeline=pipe,
            seed=seed,
            model_family="sd3-diffusers-t2i",
        )

        return DenoiseLoopState(
            current_step=0,
            total_steps=request.num_steps,
            model_state=ms,
        )

    async def predict_noise(
        self,
        denoise_state: Any,
        step_idx: int,
    ) -> dict[str, Any]:
        """Forward pass using the pipeline's own transformer."""
        return self._predict_noise_impl(
            self.pipeline.transformer, denoise_state, step_idx,
        )

    def _predict_noise_with_model(
        self,
        model: Any,
        denoise_state: Any,
        step_idx: int,
    ) -> dict[str, Any]:
        """Training path: forward using externally-provided model (e.g. LoRA'd)."""
        return self._predict_noise_impl(model, denoise_state, step_idx)

    def _predict_noise_impl(
        self,
        model: Any,
        denoise_state: Any,
        step_idx: int,
    ) -> dict[str, Any]:
        """SD3 transformer forward + optional CFG (batched).

        Concatenates uncond + cond into one forward of batch 2*B when CFG
        is active (mirrors Wan path), saving one transformer call per step.
        """
        import torch

        ms = denoise_state.model_state
        t = ms.timesteps[step_idx]
        batch_size = ms.latents.shape[0]
        transformer_dtype = ms.prompt_embeds.dtype

        latent_input = ms.latents.to(transformer_dtype)
        # SD3 timestep is broadcast across batch as the raw float (not /1000)
        timestep_batch = t.expand(batch_size)

        if ms.do_cfg:
            combined_latents = torch.cat([latent_input, latent_input], dim=0)
            combined_t = torch.cat([timestep_batch, timestep_batch], dim=0)
            combined_embeds = torch.cat(
                [ms.negative_prompt_embeds, ms.prompt_embeds], dim=0,
            )
            combined_pooled = torch.cat(
                [ms.negative_pooled_prompt_embeds, ms.pooled_prompt_embeds], dim=0,
            )
            combined_out = model(
                hidden_states=combined_latents,
                timestep=combined_t,
                encoder_hidden_states=combined_embeds,
                pooled_projections=combined_pooled,
                return_dict=False,
            )[0]
            noise_pred_uncond, noise_pred_cond = combined_out.chunk(2, dim=0)
            noise_pred_uncond = noise_pred_uncond.to(ms.prompt_embeds.dtype)
            noise_pred_cond = noise_pred_cond.to(ms.prompt_embeds.dtype)
            noise_pred = noise_pred_uncond + ms.guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
        else:
            noise_pred_cond = model(
                hidden_states=latent_input,
                timestep=timestep_batch,
                encoder_hidden_states=ms.prompt_embeds,
                pooled_projections=ms.pooled_prompt_embeds,
                return_dict=False,
            )[0].to(ms.prompt_embeds.dtype)
            noise_pred_uncond = torch.zeros_like(noise_pred_cond)
            noise_pred = noise_pred_cond

        return {
            "noise_pred": noise_pred,
            "noise_pred_cond": noise_pred_cond,
            "noise_pred_uncond": noise_pred_uncond,
        }

    # -- monolithic decode / generate ----------------------------------

    async def decode_vae(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> ModelResult:
        """Decode latents -> image via SD3 VAE (4D, no T dim)."""
        import torch

        pipe = self.pipeline
        latents = state["latents"]

        latents_for_decode = latents.to(pipe.vae.dtype)
        # SD3 VAE: latents = (z - shift_factor) * scaling_factor → invert
        scaling_factor = pipe.vae.config.scaling_factor
        shift_factor = getattr(pipe.vae.config, "shift_factor", 0.0) or 0.0
        latents_for_decode = latents_for_decode / scaling_factor + shift_factor

        image = pipe.vae.decode(latents_for_decode, return_dict=False)[0]
        # postprocess to [0, 1] float tensor [B, C, H, W]
        image = pipe.image_processor.postprocess(image, output_type="pt")

        return ModelResult(
            state_updates={"video": image},  # reuse "video" key for OnlineTrainer
            outputs={"image_shape": list(image.shape)},
        )

    async def generate(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> ModelResult:
        """Monolithic generation: denoise_init → loop → decode_vae."""
        import torch

        denoise_loop = await self.denoise_init(request, state)
        ms = denoise_loop.model_state
        transformer_dtype = ms.prompt_embeds.dtype

        with torch.amp.autocast("cuda", dtype=transformer_dtype):
            with torch.no_grad():
                for step_idx in range(denoise_loop.total_steps):
                    fwd = await self.predict_noise(denoise_loop, step_idx)
                    ms.latents = ms.scheduler.step(
                        fwd["noise_pred"], ms.timesteps[step_idx], ms.latents,
                        return_dict=False,
                    )[0]
                    denoise_loop.current_step = step_idx + 1

        state["latents"] = ms.latents
        result = await self.decode_vae(request, state)

        gc.collect()
        torch.cuda.empty_cache()

        return result
