"""Wan 2.1 t2v diffusers adapter (DiffusionPolicy contract).

Single-protocol adapter for Wan T2V on the RL path. The contract is:

    encode_prompt -> prepare_sampling -> forward_step xN -> decode_latents

The collector (or default ``DiffusionPolicy.inference`` loop) owns the
scheduler step / SDE step. ``forward_step`` does only one transformer
forward (with optional batched CFG concat) and returns noise predictions.

Per-family ``WanT2VSamplingState`` is private to this file. The engine /
collector code MUST NOT introspect it beyond the documented attributes
(``latents``, ``timesteps``, ``scheduler``, plus the embeds the eval path
re-builds explicitly).

Differences from SD3:
- ``prompt_embeds`` only (no pooled/CLIP); transformer signature lacks
  ``pooled_projections``.
- 5D latents ``[B, C, T, H, W]`` (Wan VAE temporal axis).
- VAE decode applies Wan-specific per-channel ``latents_mean`` /
  ``latents_std`` denormalization (over ``z_dim``).
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import Any

import torch

from vrl.models.diffusion import DiffusionPolicy, VideoGenerationRequest


@dataclass
class WanT2VSamplingState:
    """Private Wan T2V sampling state. Engine MUST NOT introspect."""

    latents: torch.Tensor
    timesteps: torch.Tensor
    scheduler: Any
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    guidance_scale: float
    do_cfg: bool
    seed: int


class WanT2VDiffusersPolicy(DiffusionPolicy):
    """Diffusers-backed Wan 2.1 T2V adapter (1.3B variant)."""

    family = "wan-diffusers-t2v"

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
    def from_spec(cls, spec: Any) -> WanT2VDiffusersPolicy:
        """Load the diffusers WanPipeline + freeze non-trainable modules."""
        from diffusers import WanPipeline

        pipeline = WanPipeline.from_pretrained(
            spec.model_name_or_path, torch_dtype=spec.dtype,
        )
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.vae.to(spec.device, dtype=torch.float32)
        pipeline.text_encoder.to(spec.device, dtype=spec.dtype)
        return cls(pipeline=pipeline, device=spec.device)

    def apply_lora(self, spec: Any) -> None:
        """Wrap the Wan transformer with PEFT LoRA per spec.lora_*."""
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
        self.pipeline.transformer.requires_grad_(True)
        self.pipeline.transformer.to(self.device)

    def torch_compile_transformer(self, mode: str) -> None:
        self._set_transformer(
            torch.compile(self.pipeline.transformer, mode=mode, fullgraph=False),
        )

    def set_num_steps(self, n: int) -> None:
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
        """Encode prompt via Wan's T5 text encoder.

        Returns ``prompt_embeds`` and the matching ``negative_prompt_embeds``
        when CFG is active. Wan does not use a pooled CLIP embedding.
        """
        max_seq = kwargs.get("max_sequence_length", 512) or 512
        guidance_scale = kwargs.get("guidance_scale", 4.5)
        do_cfg = guidance_scale > 1.0
        neg = negative_prompt if negative_prompt is not None else ""

        prompt_embeds, negative_prompt_embeds = self.pipeline.encode_prompt(
            prompt=prompt,
            negative_prompt=neg,
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=1,
            max_sequence_length=max_seq,
            device=self.device,
        )

        td = self.pipeline.transformer.dtype
        prompt_embeds = prompt_embeds.to(td)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(td)

        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
        }

    # -- prepare_sampling ----------------------------------------------

    def prepare_sampling(
        self,
        request: VideoGenerationRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> WanT2VSamplingState:
        """Build the per-request SamplingState for a Wan T2V denoise loop."""
        pipe = self.pipeline
        device = self.device

        prompt_embeds = encoded["prompt_embeds"]
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")

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

        # Wan prepare_latents signature:
        # (batch, channels, height, width, num_frames, dtype, device, generator, latents)
        latents = pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            request.height,
            request.width,
            request.frame_count,
            torch.float32,
            device,
            generator,
            None,
        )

        return WanT2VSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=pipe.scheduler,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            do_cfg=do_cfg,
            seed=seed,
        )

    # -- forward_step --------------------------------------------------

    def forward_step(
        self,
        state: WanT2VSamplingState,
        step_idx: int,
        *,
        model: Any = None,
    ) -> dict[str, Any]:
        """Wan T2V transformer forward + optional batched CFG.

        ``model`` overrides the default transformer (used by the trainer to
        forward through the LoRA-wrapped policy). Returns noise_pred plus
        the un/conditional branches; the caller owns scheduler.step / SDE.

        Timestep shape convention (mirrors SD3 adapter):
        - rollouts: ``state.timesteps`` is 1-D ``[T]``; we expand a scalar to ``[B]``.
        - eval/training: collector packs per-sample timestep as ``[B]``; expand is a no-op.
        """
        m = self._resolve_step_model(model)

        t = state.timesteps[step_idx]
        bsz = state.latents.shape[0]
        td = state.prompt_embeds.dtype

        latent_input = state.latents.to(td)
        timestep_batch = t.expand(bsz) if t.ndim == 0 else t

        if state.do_cfg:
            combined_latents = torch.cat([latent_input, latent_input], dim=0)
            combined_t = torch.cat([timestep_batch, timestep_batch], dim=0)
            combined_embeds = torch.cat(
                [state.negative_prompt_embeds, state.prompt_embeds], dim=0,
            )
            combined_out = m(
                hidden_states=combined_latents,
                timestep=combined_t,
                encoder_hidden_states=combined_embeds,
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

    def export_batch_context(self, state: WanT2VSamplingState) -> dict[str, Any]:
        """Project SamplingState -> ExperienceBatch.context (shared metadata)."""
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
            "model_family": self.family,
        }

    def export_training_extras(self, state: WanT2VSamplingState) -> dict[str, Any]:
        """Project SamplingState -> ExperienceBatch.extras (per-sample tensors)."""
        return {
            "prompt_embeds": state.prompt_embeds,
            "negative_prompt_embeds": state.negative_prompt_embeds,
        }

    def restore_eval_state(
        self,
        batch_extras: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> WanT2VSamplingState:
        """Rebuild SamplingState for the eval forward path from a batch slice."""
        ts = batch_extras["timesteps"]
        t = ts[:, step_idx] if ts.ndim > 1 else ts
        # Pack as [1, B] so forward_step's state.timesteps[0] returns [B]
        # (matches the rollout convention where timesteps is 1-D and indexed
        # by step_idx; here we use step_idx=0 in the eval call).
        timesteps = t.unsqueeze(0) if t.ndim == 1 else t
        return WanT2VSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=None,
            prompt_embeds=batch_extras["prompt_embeds"],
            negative_prompt_embeds=batch_extras.get("negative_prompt_embeds"),
            guidance_scale=batch_context["guidance_scale"],
            do_cfg=batch_context["cfg"] and batch_context["guidance_scale"] > 1.0,
            seed=0,
        )

    # -- decode_latents ------------------------------------------------

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode 5D latents -> video [B, C, T, H, W] via Wan VAE.

        Applies Wan-specific per-channel denormalization using the VAE's
        ``latents_mean`` / ``latents_std`` over the ``z_dim`` channel axis.
        """
        pipe = self.pipeline
        x = latents.to(pipe.vae.dtype)

        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(x.device, x.dtype)
        )
        latents_std = (
            1.0 / torch.tensor(pipe.vae.config.latents_std)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(x.device, x.dtype)
        )
        x = x / latents_std + latents_mean

        video = pipe.vae.decode(x, return_dict=False)[0]
        # video_processor.postprocess_video returns [B, T, C, H, W]
        video = pipe.video_processor.postprocess_video(video, output_type="pt")
        # -> [B, C, T, H, W]
        video = video.permute(0, 2, 1, 3, 4)
        return video
