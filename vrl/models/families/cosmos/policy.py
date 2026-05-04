"""Cosmos Predict2 (Video2World) diffusers adapter — DiffusionPolicy contract.

Single-protocol adapter for the Cosmos Predict2 Video2World pipeline. The
contract is:

    encode_prompt -> prepare_sampling -> forward_step xN -> decode_latents

The collector owns the scheduler step / SDE step. ``forward_step`` does
only one transformer forward (with optional CFG branch) and returns noise
predictions; it does NOT mutate ``state.latents``.

Per-family :class:`CosmosPredict2SamplingState` is private to this file —
collector code MUST NOT introspect it beyond the documented attributes
(``latents``, ``timesteps``, ``scheduler``); cosmos-specific conditioning
fields are projected via ``export_*`` / ``restore_eval_state``.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import Any

import torch

from vrl.models.diffusion import DiffusionPolicy, VideoGenerationRequest


@dataclass
class CosmosPredict2SamplingState:
    """Private Cosmos Predict2 sampling state. Collector MUST NOT introspect.

    Cosmos Predict2 Video2World needs the full conditioning bundle
    (``init_latents`` + cond/uncond masks + indicators + padding mask + fps)
    in the per-step transformer forward and in the SDE step. Keeping the
    bundle local to the adapter avoids leaking these knobs into the engine
    contract.
    """

    latents: torch.Tensor
    timesteps: torch.Tensor
    scheduler: Any
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    guidance_scale: float
    do_cfg: bool
    init_latents: torch.Tensor
    cond_mask: Any
    uncond_mask: Any
    padding_mask: Any
    cond_indicator: Any
    uncond_indicator: Any
    fps: int
    seed: int
    sigma_conditioning: float = 0.0001


class CosmosPredict2Policy(DiffusionPolicy):
    """Diffusers-backed Cosmos Predict2 Video2World adapter (RL path).

    The pipeline is constructed by the family builder
    (:func:`vrl.models.families.cosmos.builder.build_cosmos_predict2_runtime_bundle`)
    and passed in. Scripts must NOT instantiate the diffusers pipeline
    directly.
    """

    family = "cosmos-predict2-diffusers-v2w"

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

    # -- backend ownership (called by builder, not by collectors) -------

    @classmethod
    def from_spec(cls, spec: Any) -> CosmosPredict2Policy:
        """Load Cosmos2VideoToWorldPipeline + freeze non-trainable modules.

        Patches the diffusers safety checker with a passthrough during load
        so RL training does not depend on the safety classifier weights.
        """
        import diffusers.pipelines.cosmos.pipeline_cosmos2_video2world as _v2w_mod
        from diffusers import Cosmos2VideoToWorldPipeline

        class _PassthroughSafetyChecker:
            def to(self, device: Any) -> _PassthroughSafetyChecker:
                return self

            def check_text_safety(self, prompt: str) -> bool:
                return True

            def check_video_safety(self, video: Any) -> Any:
                return video

        _orig = _v2w_mod.CosmosSafetyChecker
        _v2w_mod.CosmosSafetyChecker = _PassthroughSafetyChecker  # type: ignore[assignment]
        try:
            pipeline = Cosmos2VideoToWorldPipeline.from_pretrained(
                spec.model_name_or_path, torch_dtype=spec.dtype,
            )
        finally:
            _v2w_mod.CosmosSafetyChecker = _orig

        # diffusers from_pretrained disables grad globally — re-enable.
        torch.set_grad_enabled(True)

        pipeline.set_progress_bar_config(disable=True)
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.vae.to(spec.device, dtype=torch.float32)
        pipeline.text_encoder.to(spec.device, dtype=spec.dtype)
        return cls(pipeline=pipeline, device=spec.device)

    def apply_lora(self, spec: Any) -> None:
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

    @property
    def device(self) -> Any:
        return self._device if self._device is not None else self.pipeline.device

    def describe(self) -> dict[str, Any]:
        return {"family": self.family, "device": str(self.device)}

    # -- encode_prompt -------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode prompt + optional negative for Cosmos Predict2.

        ``reference_image`` may be provided here so callers can plumb it
        through alongside text encoding; it's threaded into
        ``prepare_sampling`` via the returned dict so the collector can
        keep a single ``encoded`` handle.
        """
        guidance_scale = kwargs.get("guidance_scale", 7.0)
        do_cfg = guidance_scale > 1.0
        neg = negative_prompt if negative_prompt is not None else ""

        device = self.device
        encode_result = self.pipeline.encode_prompt(
            prompt=prompt,
            negative_prompt=neg or None,
            do_classifier_free_guidance=do_cfg,
            device=device,
        )
        prompt_embeds = encode_result[0]
        negative_prompt_embeds = encode_result[1] if do_cfg else None

        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "reference_image": kwargs.get("reference_image"),
        }

    # -- prepare_sampling ----------------------------------------------

    def prepare_sampling(
        self,
        request: VideoGenerationRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> CosmosPredict2SamplingState:
        """Build the per-request Cosmos Predict2 sampling state.

        Uses ``pipeline.prepare_latents(...)`` to materialize the
        Video2World 6-tuple (latents, init_latents, cond_indicator,
        uncond_indicator, cond_mask, uncond_mask) plus a zero
        ``padding_mask`` at pixel resolution that the transformer
        repeats internally.
        """
        pipe = self.pipeline
        device = self.device

        prompt_embeds = encoded["prompt_embeds"]
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")
        reference_image = kwargs.get("reference_image", encoded.get("reference_image"))

        guidance_scale = request.guidance_scale
        do_cfg = guidance_scale > 1.0

        pipe.scheduler.set_timesteps(request.num_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        num_channels_latents = pipe.transformer.config.in_channels - 1
        batch_size = prompt_embeds.shape[0]

        # Reference image preprocessing for Video2World; fall back to zero
        # video conditioning when no reference is provided (degenerate but
        # allows smoke-test paths to run).
        if reference_image is not None:
            video_input = pipe.video_processor.preprocess_video(
                reference_image,
                height=request.height,
                width=request.width,
            ).to(device, dtype=pipe.vae.dtype)
        else:
            video_input = torch.zeros(
                batch_size, 3, 1, request.height, request.width,
                device=device, dtype=pipe.vae.dtype,
            )

        seed = (
            request.seed if request.seed is not None
            else random.randint(0, sys.maxsize)
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        latents_result = pipe.prepare_latents(
            video=video_input,
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=request.height,
            width=request.width,
            num_frames=request.frame_count,
            do_classifier_free_guidance=do_cfg,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=None,
        )
        latents = latents_result[0]
        init_latents = latents_result[1]
        cond_indicator = latents_result[2]
        uncond_indicator = latents_result[3]
        cond_mask = latents_result[4]
        uncond_mask = latents_result[5]

        # Padding mask at pixel resolution; transformer repeats internally.
        padding_mask = latents.new_zeros(
            1, 1, request.height, request.width,
            dtype=prompt_embeds.dtype,
        )

        return CosmosPredict2SamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=pipe.scheduler,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            do_cfg=do_cfg,
            init_latents=init_latents,
            cond_mask=cond_mask,
            uncond_mask=uncond_mask,
            padding_mask=padding_mask,
            cond_indicator=cond_indicator,
            uncond_indicator=uncond_indicator,
            fps=request.fps or 16,
            seed=seed,
            sigma_conditioning=0.0001,
        )

    # -- forward_step --------------------------------------------------

    def forward_step(
        self,
        state: CosmosPredict2SamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """One Cosmos Predict2 transformer forward + optional CFG.

        Mirrors the diffusers Cosmos2VideoToWorldPipeline denoising loop:
        sigma scaling (c_in/c_skip/c_out), spatial timestep blending via
        ``cond_indicator``, and post-transformer conditioning re-application.
        Returns ``{"noise_pred", "noise_pred_cond", "noise_pred_uncond"}``.
        NO scheduler step happens here.
        """
        m = self.transformer

        batch_size = state.latents.shape[0]
        transformer_dtype = state.prompt_embeds.dtype

        # Sigma scaling coefficients from the scheduler.
        current_sigma = state.scheduler.sigmas[step_idx]
        current_t = current_sigma / (current_sigma + 1)
        c_in = 1 - current_t
        c_skip = 1 - current_t
        c_out = -current_t

        # Spatial timestep tensor [B, 1, T, 1, 1]
        timestep = current_t.view(1, 1, 1, 1, 1).expand(
            batch_size, -1, state.latents.size(2), -1, -1,
        )
        t_conditioning = torch.tensor(
            state.sigma_conditioning,
            device=state.latents.device,
            dtype=timestep.dtype,
        ).view(1, 1, 1, 1, 1).expand_as(timestep)

        # Expand cond/uncond fields to runtime batch (prepare_latents may
        # return [1,...] but the collector may have stacked groups).
        cond_mask = state.cond_mask.expand(batch_size, -1, -1, -1, -1)
        uncond_mask = (
            state.uncond_mask.expand(batch_size, -1, -1, -1, -1)
            if state.uncond_mask is not None else None
        )
        # padding_mask kept at [1, 1, H, W] — transformer repeats internally.
        padding_mask = state.padding_mask
        init_latents = state.init_latents.expand(batch_size, -1, -1, -1, -1)
        cond_indicator = state.cond_indicator.expand(batch_size, -1, -1, -1, -1)
        uncond_indicator = (
            state.uncond_indicator.expand(batch_size, -1, -1, -1, -1)
            if state.uncond_indicator is not None else None
        )

        # Conditional pass: blend conditioning latents into the scaled noise.
        cond_latent = state.latents * c_in
        cond_latent = (
            cond_indicator * init_latents
            + (1 - cond_indicator) * cond_latent
        )
        cond_timestep = (
            cond_indicator * t_conditioning
            + (1 - cond_indicator) * timestep
        )

        raw_cond = m(
            hidden_states=cond_latent.to(transformer_dtype),
            timestep=cond_timestep.to(transformer_dtype),
            encoder_hidden_states=state.prompt_embeds,
            fps=state.fps,
            condition_mask=cond_mask,
            padding_mask=padding_mask,
            return_dict=False,
        )[0]
        noise_pred_cond = (
            c_skip * state.latents + c_out * raw_cond.float()
        ).to(transformer_dtype)
        noise_pred_cond = (
            cond_indicator * init_latents
            + (1 - cond_indicator) * noise_pred_cond
        )

        if state.do_cfg:
            uncond_latent = state.latents * c_in
            uncond_latent = (
                uncond_indicator * init_latents
                + (1 - uncond_indicator) * uncond_latent
            )
            uncond_timestep = (
                uncond_indicator * t_conditioning
                + (1 - uncond_indicator) * timestep
            )

            raw_uncond = m(
                hidden_states=uncond_latent.to(transformer_dtype),
                timestep=uncond_timestep.to(transformer_dtype),
                encoder_hidden_states=state.negative_prompt_embeds,
                fps=state.fps,
                condition_mask=uncond_mask,
                padding_mask=padding_mask,
                return_dict=False,
            )[0]
            noise_pred_uncond = (
                c_skip * state.latents + c_out * raw_uncond.float()
            ).to(transformer_dtype)
            noise_pred_uncond = (
                uncond_indicator * init_latents
                + (1 - uncond_indicator) * noise_pred_uncond
            )

            combined = noise_pred_cond + state.guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
        else:
            noise_pred_uncond = torch.zeros_like(noise_pred_cond)
            combined = noise_pred_cond

        # Convert to the velocity-shaped noise_pred consumed by the
        # FlowMatchEulerDiscreteScheduler.
        noise_pred = (state.latents - combined) / current_sigma

        return {
            "noise_pred": noise_pred,
            "noise_pred_cond": noise_pred_cond,
            "noise_pred_uncond": noise_pred_uncond,
        }

    # -- collector boundary --------------------------------------------

    def export_batch_context(
        self,
        state: CosmosPredict2SamplingState,
    ) -> dict[str, Any]:
        """Project SamplingState -> RolloutBatch.context (shared metadata).

        The cond/uncond/padding masks + indicators are shared across the
        batch (not per-sample), so they live in ``context`` rather than
        ``extras``. The scheduler is intentionally NOT packed — the eval
        path reads ``self.pipeline.scheduler`` directly via this adapter
        so the collector never holds a reference to a private mutable
        object.
        """
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
            "model_family": self.family,
            "fps": state.fps,
            "cond_mask": state.cond_mask,
            "uncond_mask": state.uncond_mask,
            "padding_mask": state.padding_mask,
            "cond_indicator": state.cond_indicator,
            "uncond_indicator": state.uncond_indicator,
            "sigma_conditioning": state.sigma_conditioning,
        }

    def export_training_extras(
        self,
        state: CosmosPredict2SamplingState,
    ) -> dict[str, Any]:
        """Project SamplingState -> RolloutBatch.extras (per-sample tensors).

        ``init_latents`` is per-sample because Video2World conditioning
        depends on the reference image.
        """
        return {
            "prompt_embeds": state.prompt_embeds,
            "negative_prompt_embeds": state.negative_prompt_embeds,
            "init_latents": state.init_latents,
        }

    def restore_eval_state(
        self,
        batch_extras: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> CosmosPredict2SamplingState:
        """Rebuild SamplingState for the eval forward path from a batch slice.

        IMPORTANT: Cosmos's ``forward_step`` indexes BOTH
        ``state.timesteps[step_idx]`` AND ``state.scheduler.sigmas[step_idx]``
        for sigma scaling, so the eval path passes through the actual
        ``step_idx`` (NOT 0 like sd3/wan). We hand back the adapter's
        own scheduler + its full timesteps array so that indexing stays
        consistent with the rollout-time scheduler state.
        """
        # ``step_idx`` is consumed by the caller's ``forward_step`` call,
        # not here — but we accept it in the signature for parity with
        # the base contract and to allow defensive checks if needed.
        del step_idx
        return CosmosPredict2SamplingState(
            latents=latents,
            timesteps=self.pipeline.scheduler.timesteps,
            scheduler=self.pipeline.scheduler,
            prompt_embeds=batch_extras["prompt_embeds"],
            negative_prompt_embeds=batch_extras.get("negative_prompt_embeds"),
            guidance_scale=batch_context["guidance_scale"],
            do_cfg=batch_context["cfg"] and batch_context["guidance_scale"] > 1.0,
            init_latents=batch_extras["init_latents"],
            cond_mask=batch_context["cond_mask"],
            uncond_mask=batch_context["uncond_mask"],
            padding_mask=batch_context["padding_mask"],
            cond_indicator=batch_context["cond_indicator"],
            uncond_indicator=batch_context["uncond_indicator"],
            fps=batch_context["fps"],
            seed=0,
            sigma_conditioning=batch_context.get("sigma_conditioning", 0.0001),
        )

    def replay_forward(
        self,
        batch: Any,
        timestep_idx: int,
    ) -> dict[str, Any]:
        """Cosmos replay: forward with the real ``timestep_idx`` (NOT 0).

        Unlike sd3/wan which pack timesteps as ``[1, B]`` and call
        ``forward_step(state, 0)``, Cosmos's ``forward_step`` indexes
        ``state.scheduler.sigmas[step_idx]`` so the eval path must pass
        through the actual ``timestep_idx`` to keep sigma scaling consistent
        with the rollout-time scheduler state.
        """
        state = self.restore_eval_state(
            batch.extras,
            batch.context,
            batch.observations[:, timestep_idx],
            timestep_idx,
        )
        return self.forward_step(state, timestep_idx)

    # -- decode_latents ------------------------------------------------

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to video tensor [B, C, T, H, W] in [0, 1].

        Cosmos VAE expects latents un-normalized via ``sigma_data`` and
        the per-channel ``latents_mean`` / ``latents_std`` stats stored on
        ``vae.config``. We then pass through ``video_processor.postprocess_video``
        and permute to the [B, C, T, H, W] convention used elsewhere.
        """
        pipe = self.pipeline
        sigma_data = pipe.scheduler.config.sigma_data

        x = latents.to(pipe.vae.dtype)
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(x.device, x.dtype)
        )
        latents_std = (
            torch.tensor(pipe.vae.config.latents_std)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(x.device, x.dtype)
        )
        x = x * latents_std / sigma_data + latents_mean
        video = pipe.vae.decode(x, return_dict=False)[0]
        video = pipe.video_processor.postprocess_video(video, output_type="pt")
        # [B, T, C, H, W] -> [B, C, T, H, W]
        return video.permute(0, 2, 1, 3, 4)
