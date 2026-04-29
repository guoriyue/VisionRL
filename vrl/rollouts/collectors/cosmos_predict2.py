"""Cosmos Predict2 collector for RL training (DiffusionPolicy contract).

Collector → FlowMatchingEvaluator → GRPO → OnlineTrainer

Drives the new single-protocol :class:`DiffusionPolicy` API:

    encode_prompt → prepare_sampling → forward_step ×N → decode_latents

The collector owns the SDE step / log-prob recording / reward scoring;
the adapter owns the transformer forward and the family-specific
conditioning bundle.
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
class CosmosPredict2CollectorConfig:
    """Configuration for CosmosPredict2Collector."""

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


class CosmosPredict2Collector:
    """Collect rollouts from Cosmos Predict2 V2W with per-step log-probs.

    Calls the adapter's ``encode_prompt`` / ``prepare_sampling`` /
    ``forward_step`` / ``decode_latents`` directly. The collector only
    owns the SDE-step loop, reward scoring, and ``ExperienceBatch``
    assembly.
    """

    def __init__(
        self,
        model: Any,  # CosmosPredict2Policy
        reward_fn: Any,  # RewardFunction instance
        config: CosmosPredict2CollectorConfig | None = None,
        reference_image: Any = None,  # PIL.Image for Video2World conditioning
    ) -> None:
        self.model = model
        self.reward_fn = reward_fn
        self.config = config or CosmosPredict2CollectorConfig()
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
        1. Encode prompt via adapter (single forward pass for the whole
           prompt; CFG controlled by ``guidance_scale``).
        2. Build the per-request opaque sampling state via
           ``prepare_sampling`` (sets up latents + V2W conditioning).
        3. Custom SDE loop calling ``forward_step`` per step + the shared
           ``sde_step_with_logprob`` to record log-probs.
        4. Decode video via ``decode_latents``.
        5. Reward scoring + ``ExperienceBatch`` assembly.
        """
        import torch

        from vrl.algorithms.types import Rollout, Trajectory
        from vrl.models.diffusion import VideoGenerationRequest
        from vrl.rollouts.evaluators.diffusion.flow_matching import sde_step_with_logprob

        cfg = self.config
        batch_size = len(prompts)

        seed = kwargs.get("seed", None)
        request = VideoGenerationRequest(
            prompt=prompts[0] if len(prompts) == 1 else prompts[0],
            num_steps=cfg.num_steps,
            guidance_scale=cfg.guidance_scale,
            height=cfg.height,
            width=cfg.width,
            frame_count=cfg.num_frames,
            fps=cfg.fps,
            seed=seed,
        )

        reference_image = kwargs.get("reference_image", self.reference_image)

        # 1. Encode prompt via the adapter
        encoded = self.model.encode_prompt(
            prompts[0],
            request.negative_prompt or None,
            max_sequence_length=cfg.max_sequence_length,
            guidance_scale=cfg.guidance_scale,
            reference_image=reference_image,
        )

        # 2. Build the sampling state (latents, V2W conditioning bundle, ...)
        sampling_state = self.model.prepare_sampling(
            request, encoded, reference_image=reference_image,
        )

        sde_window = self._get_sde_window()

        device = sampling_state.latents.device
        if seed is not None:
            sde_generator = torch.Generator(device=device)
            sde_generator.manual_seed(seed)
        elif cfg.same_latent:
            sde_generator = torch.Generator(device=device)
            sde_generator.manual_seed(hash(prompts[0]) % (2**32))
        else:
            sde_generator = None

        # 3. Custom denoise loop with log-prob tracking
        all_observations = []
        all_actions = []
        all_log_probs = []
        all_kls = []
        all_timestep_values = []

        # Derive transformer dtype from the encoded prompt (returned by
        # the adapter) instead of poking into SamplingState; collectors
        # treat SamplingState as opaque past ``latents``/``timesteps``/
        # ``scheduler``.
        transformer_dtype = encoded["prompt_embeds"].dtype
        total_steps = len(sampling_state.timesteps)

        with torch.amp.autocast("cuda", dtype=transformer_dtype):
            with torch.no_grad():
                for step_idx in range(total_steps):
                    latents_ori = sampling_state.latents.clone()
                    t = sampling_state.timesteps[step_idx]

                    fwd = self.model.forward_step(sampling_state, step_idx)
                    noise_pred = fwd["noise_pred"]

                    in_sde_window = sde_window is None or (
                        sde_window[0] <= step_idx < sde_window[1]
                    )

                    sde_result = sde_step_with_logprob(
                        sampling_state.scheduler,
                        noise_pred.float(),
                        t.unsqueeze(0),
                        sampling_state.latents.float(),
                        generator=sde_generator if in_sde_window else None,
                        deterministic=not in_sde_window,
                        return_dt=cfg.kl_reward > 0,
                    )
                    prev_latents = sde_result.prev_sample
                    sampling_state.latents = prev_latents

                    all_observations.append(latents_ori.detach())
                    all_actions.append(prev_latents.detach())
                    all_log_probs.append(sde_result.log_prob.detach())
                    all_timestep_values.append(t.detach())

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

        # 4. Decode latents to video [B, C, T, H, W]
        video = self.model.decode_latents(sampling_state.latents)

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

        rewards_raw = torch.tensor(rewards_list, dtype=torch.float32, device=device)
        if cfg.kl_reward > 0:
            kl_total = kl_tensor.sum(dim=1)
            rewards_adjusted = rewards_raw - cfg.kl_reward * kl_total
        else:
            rewards_adjusted = rewards_raw

        dones = torch.ones(batch_size, dtype=torch.bool, device=device)
        group_ids = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Boundary helpers — adapter projects SamplingState into the
        # collector-visible dicts so the collector never reads private
        # state fields directly.
        training_extras = self.model.export_training_extras(sampling_state)
        rollout_context = self.model.export_batch_context(sampling_state)

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
                **training_extras,
            },
            context=rollout_context,
            videos=video,
            prompts=prompts,
        )

