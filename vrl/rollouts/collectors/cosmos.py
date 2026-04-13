"""Cosmos Predict2-2B Diffusers-based collector for RL training.

Uses HuggingFace ``diffusers.Cosmos2VideoToWorldPipeline`` with
``sde_step_with_logprob`` for per-step log-probability tracking.

Targets ``nvidia/Cosmos-Predict2-2B-Video2World`` for single-GPU training.

Follows the same architecture as ``WanDiffusersCollector``:
  Collector → FlowMatchingEvaluator → GRPO → OnlineTrainer
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vrl.rollouts.types import ExperienceBatch

if TYPE_CHECKING:
    from vrl.rewards.base import RewardFunction

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CosmosDiffusersCollectorConfig:
    """Configuration for CosmosDiffusersCollector."""

    num_steps: int = 35
    guidance_scale: float = 7.0
    height: int = 704
    width: int = 1280
    num_frames: int = 93  # default for Cosmos Predict2 (81 gen + 12 cond)
    max_sequence_length: int = 512
    fps: int = 16

    # CFG during sampling
    cfg: bool = True

    # KL reward — subtract kl_reward * kl from rewards before advantages.
    kl_reward: float = 0.0

    # SDE window — only inject SDE noise for steps within the window.
    sde_window_size: int = 0
    sde_window_range: tuple[int, int] = (0, 10)

    # Same latent — reuse the same noise for samples sharing a prompt.
    same_latent: bool = False


class CosmosDiffusersCollector:
    """Collect rollouts from a Cosmos2VideoToWorldPipeline with per-step log-probs.

    Uses the HuggingFace ``diffusers.Cosmos2VideoToWorldPipeline`` with a
    custom denoise loop that tracks per-step log-probabilities via
    ``sde_step_with_logprob``.

    Implements both ``collect()`` (rollout) and ``forward_step()`` (single-timestep
    forward for training evaluator).
    """

    def __init__(
        self,
        pipeline: Any,  # diffusers.Cosmos2VideoToWorldPipeline
        reward_fn: Any,  # RewardFunction instance
        config: CosmosDiffusersCollectorConfig | None = None,
        reference_image: Any = None,  # PIL.Image for Video2World conditioning
    ) -> None:
        self.pipeline = pipeline
        self.reward_fn = reward_fn
        self.config = config or CosmosDiffusersCollectorConfig()
        self.reference_image = reference_image

    def _get_sde_window(self) -> tuple[int, int] | None:
        """Compute random SDE window for this collection."""
        cfg = self.config
        if cfg.sde_window_size <= 0:
            return None
        lo, hi = cfg.sde_window_range
        start = random.randint(lo, max(lo, hi - cfg.sde_window_size))
        end = start + cfg.sde_window_size
        return (start, end)

    async def collect(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> ExperienceBatch:
        """Collect Cosmos Predict2 rollouts with per-step log-probabilities.

        Steps:
        1. Encode text (prompt + negative prompt)
        2. Prepare latents with conditioning (reference image → VAE encode)
        3. Custom denoise loop with sde_step_with_logprob
        4. Decode VAE -> video
        5. Reward scoring
        6. Stack into ExperienceBatch
        """
        import torch

        from vrl.algorithms.types import Rollout, Trajectory
        from vrl.rollouts.evaluators.diffusion.flow_matching import sde_step_with_logprob

        pipe = self.pipeline
        cfg = self.config
        device = pipe.device

        # 1. Encode text
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompts,
            negative_prompt=[""] * len(prompts),
            do_classifier_free_guidance=cfg.cfg and cfg.guidance_scale > 1.0,
            num_videos_per_prompt=1,
            max_sequence_length=cfg.max_sequence_length,
            device=device,
        )
        transformer_dtype = pipe.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        batch_size = len(prompts)

        # 2. Prepare scheduler + latents (Cosmos2 Video2World needs reference image)
        pipe.scheduler.set_timesteps(cfg.num_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        do_cfg = cfg.cfg and cfg.guidance_scale > 1.0
        num_channels_latents = pipe.transformer.config.in_channels

        # Preprocess reference image for Video2World
        reference_image = kwargs.get("reference_image", self.reference_image)
        if reference_image is not None:
            video_input = pipe.video_processor.preprocess_video(
                reference_image, height=cfg.height, width=cfg.width,
            ).to(device, dtype=pipe.vae.dtype)
        else:
            # No reference → create a blank conditioning frame
            video_input = torch.zeros(
                batch_size, 3, 1, cfg.height, cfg.width,
                device=device, dtype=pipe.vae.dtype,
            )

        # Cosmos2 prepare_latents returns a 6-tuple:
        # (latents, init_latents, cond_indicator, uncond_indicator, cond_mask, uncond_mask)
        latents_result = pipe.prepare_latents(
            video=video_input,
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=cfg.height,
            width=cfg.width,
            num_frames=cfg.num_frames,
            do_classifier_free_guidance=do_cfg,
            dtype=torch.float32,
            device=device,
            generator=None,
            latents=None,
        )
        latents = latents_result[0]
        init_latents = latents_result[1]
        cond_indicator = latents_result[2]
        uncond_indicator = latents_result[3]
        cond_mask = latents_result[4]
        uncond_mask = latents_result[5]

        # SDE window
        sde_window = self._get_sde_window()

        # Same-latent generator
        if cfg.same_latent:
            latent_generator = torch.Generator(device=device)
            latent_generator.manual_seed(hash(prompts[0]) % (2**32))
        else:
            latent_generator = None

        # Sigma values for conditioning noise (from pipeline defaults)
        sigma_data = pipe.scheduler.config.sigma_data  # 1.0
        sigma_conditioning = 0.0001  # matches pipeline default

        # 3. Custom denoise loop with log-prob tracking
        all_observations = []
        all_actions = []
        all_log_probs = []
        all_kls = []
        all_timestep_values = []

        with torch.amp.autocast("cuda", dtype=transformer_dtype):
            with torch.no_grad():
                for step_idx, t in enumerate(timesteps):
                    latents_ori = latents.clone()

                    # Build conditioning latent (reference frames blended with noise)
                    # Following Cosmos2VideoToWorldPipeline.__call__ denoising loop
                    sigma_t = t / pipe.scheduler.config.num_train_timesteps
                    cond_latent = (
                        init_latents * (1 - sigma_conditioning)
                        + sigma_conditioning * torch.randn_like(init_latents)
                    )
                    # Blend: conditioning frames from init_latents, generation frames from latents
                    cond_latent_input = cond_indicator * cond_latent + (1 - cond_indicator) * latents.to(transformer_dtype)
                    cond_timestep = cond_indicator * sigma_conditioning + (1 - cond_indicator) * t

                    timestep_batch = cond_timestep.expand(batch_size, -1) if cond_timestep.ndim > 0 else t.expand(batch_size)

                    # Forward pass: cond
                    noise_pred = pipe.transformer(
                        hidden_states=cond_latent_input.to(transformer_dtype),
                        timestep=t.expand(batch_size),
                        encoder_hidden_states=prompt_embeds,
                        fps=cfg.fps,
                        condition_mask=cond_mask,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred.to(prompt_embeds.dtype)

                    # CFG: uncond pass
                    if do_cfg:
                        uncond_latent_input = uncond_indicator * cond_latent + (1 - uncond_indicator) * latents.to(transformer_dtype)
                        noise_uncond = pipe.transformer(
                            hidden_states=uncond_latent_input.to(transformer_dtype),
                            timestep=t.expand(batch_size),
                            encoder_hidden_states=negative_prompt_embeds,
                            fps=cfg.fps,
                            condition_mask=uncond_mask,
                            return_dict=False,
                        )[0]
                        noise_pred = noise_uncond + cfg.guidance_scale * (
                            noise_pred - noise_uncond
                        )

                    # Check SDE window
                    in_sde_window = sde_window is None or (
                        sde_window[0] <= step_idx < sde_window[1]
                    )

                    # SDE step with log-prob
                    sde_result = sde_step_with_logprob(
                        pipe.scheduler,
                        noise_pred.float(),
                        t.unsqueeze(0),
                        latents.float(),
                        generator=latent_generator if in_sde_window else None,
                        deterministic=not in_sde_window,
                        return_dt=cfg.kl_reward > 0,
                    )
                    prev_latents = sde_result.prev_sample
                    latents = prev_latents

                    all_observations.append(latents_ori.detach())
                    all_actions.append(prev_latents.detach())
                    all_log_probs.append(sde_result.log_prob.detach())
                    all_timestep_values.append(t.detach())

                    # Per-step KL tracking for kl_reward
                    if cfg.kl_reward > 0:
                        all_kls.append(sde_result.log_prob.detach().abs())
                    else:
                        all_kls.append(
                            torch.zeros(batch_size, device=device)
                        )

        # Stack: [T, B, ...] -> [B, T, ...]
        observations = torch.stack(all_observations, dim=1)
        actions = torch.stack(all_actions, dim=1)
        log_probs = torch.stack(all_log_probs, dim=1)
        timesteps_tensor = torch.stack(
            [tv.expand(batch_size) for tv in all_timestep_values], dim=1
        )
        kl_tensor = torch.stack(all_kls, dim=1)

        # 4. Decode VAE -> video (Cosmos2 normalization includes sigma_data)
        latents_for_decode = latents.to(pipe.vae.dtype)
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents_for_decode.device, latents_for_decode.dtype)
        )
        latents_std = (
            torch.tensor(pipe.vae.config.latents_std)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents_for_decode.device, latents_for_decode.dtype)
        )
        # Cosmos2 decode: z_raw = z_norm * std / sigma_data + mean
        latents_for_decode = latents_for_decode * latents_std / sigma_data + latents_mean
        video = pipe.vae.decode(latents_for_decode, return_dict=False)[0]
        # Postprocess to [0, 1] range
        video = pipe.video_processor.postprocess_video(video, output_type="pt")
        # video: [B, T, C, H, W] after postprocess — transpose to [B, C, T, H, W]
        video = video.permute(0, 2, 1, 3, 4)

        # 5. Score with reward function
        rewards_list = []
        for i in range(batch_size):
            dummy_trajectory = Trajectory(
                prompt=prompts[i],
                seed=0,
                steps=[],
                output=video[i],
            )
            dummy_rollout = Rollout(
                request=None,
                trajectory=dummy_trajectory,
            )
            r = await self.reward_fn.score(dummy_rollout)
            rewards_list.append(r)

        # 6. Subtract kl_reward * kl from rewards
        rewards_raw = torch.tensor(rewards_list, dtype=torch.float32, device=device)
        if cfg.kl_reward > 0:
            kl_total = kl_tensor.sum(dim=1)
            rewards_adjusted = rewards_raw - cfg.kl_reward * kl_total
        else:
            rewards_adjusted = rewards_raw

        # 7. Build ExperienceBatch
        dones = torch.ones(batch_size, dtype=torch.bool, device=device)
        group_ids = torch.zeros(batch_size, dtype=torch.long, device=device)

        return ExperienceBatch(
            observations=observations,
            actions=actions,
            rewards=rewards_adjusted,
            dones=dones,
            group_ids=group_ids,
            extras={
                "log_probs": log_probs,
                "timesteps": timesteps_tensor,
                "kl": kl_tensor,
                "reward_before_kl": rewards_raw,
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "guidance_scale": cfg.guidance_scale,
                "cfg": cfg.cfg,
                "fps": cfg.fps,
                "cond_mask": cond_mask,
                "uncond_mask": uncond_mask,
                "cond_indicator": cond_indicator,
                "uncond_indicator": uncond_indicator,
                "init_latents": init_latents,
            },
            videos=video,
            prompts=prompts,
        )

    # ------------------------------------------------------------------
    # forward_step — used by Evaluator during training
    # ------------------------------------------------------------------

    def forward_step(
        self,
        model: Any,
        batch: ExperienceBatch,
        timestep_idx: int,
    ) -> dict[str, Any]:
        """Cosmos Predict2 forward: single transformer + optional CFG.

        Used by the evaluator to compute fresh log-probs under the
        current policy during training. Includes Cosmos-specific kwargs
        (fps, condition_mask) for correct transformer forward.
        """
        import torch

        timesteps = batch.extras["timesteps"]
        t = timesteps[:, timestep_idx] if timesteps.ndim > 1 else timesteps

        prompt_embeds = batch.extras["prompt_embeds"]
        negative_prompt_embeds = batch.extras["negative_prompt_embeds"]
        guidance_scale = batch.extras["guidance_scale"]
        do_cfg = batch.extras["cfg"] and guidance_scale > 1.0
        fps = batch.extras["fps"]
        cond_mask = batch.extras["cond_mask"]
        uncond_mask = batch.extras["uncond_mask"]
        cond_indicator = batch.extras["cond_indicator"]
        uncond_indicator = batch.extras["uncond_indicator"]
        init_latents = batch.extras["init_latents"]

        # Prepare latents
        latents = batch.observations[:, timestep_idx]

        # Build conditioning latent (reference frames blended)
        sigma_conditioning = 0.0001
        cond_latent = (
            init_latents * (1 - sigma_conditioning)
            + sigma_conditioning * torch.randn_like(init_latents)
        )
        cond_latent_input = cond_indicator * cond_latent + (1 - cond_indicator) * latents.to(prompt_embeds.dtype)

        # Forward pass: cond
        noise_pred_cond = model(
            hidden_states=cond_latent_input.to(prompt_embeds.dtype),
            timestep=t,
            encoder_hidden_states=prompt_embeds,
            fps=fps,
            condition_mask=cond_mask,
            return_dict=False,
        )[0]
        noise_pred_cond = noise_pred_cond.to(prompt_embeds.dtype)

        if do_cfg:
            uncond_latent_input = uncond_indicator * cond_latent + (1 - uncond_indicator) * latents.to(prompt_embeds.dtype)
            noise_pred_uncond = model(
                hidden_states=uncond_latent_input.to(prompt_embeds.dtype),
                timestep=t,
                encoder_hidden_states=negative_prompt_embeds,
                fps=fps,
                condition_mask=uncond_mask,
                return_dict=False,
            )[0]
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
        else:
            noise_pred_uncond = torch.zeros_like(noise_pred_cond)
            noise_pred = noise_pred_cond

        return {
            "noise_pred": noise_pred,
            "noise_pred_cond": noise_pred_cond,
            "noise_pred_uncond": noise_pred_uncond,
        }
