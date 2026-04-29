"""Wan2.1 T2V collector for RL training (1.3B and 14B variants).

Collector -> FlowMatchingEvaluator -> GRPO -> OnlineTrainer

The collector drives the new ``DiffusionPolicy`` contract directly:

    encode_prompt -> prepare_sampling -> forward_step xN (with SDE step) -> decode_latents

It owns the SDE-step loop, the per-step log-prob bookkeeping, the reward
scoring, and the ``ExperienceBatch`` assembly. The eval/training path
re-uses the adapter's ``forward_step`` with a ``model=`` override so there
is exactly one transformer-forward implementation per family.
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vrl.rollouts.types import ExperienceBatch

if TYPE_CHECKING:
    from vrl.rewards.base import RewardFunction

logger = logging.getLogger(__name__)

# Populated by collect() when VRL_PROFILE_COLLECT=1. OnlineTrainer merges
# these into phase_times so --profile output includes sub-phase breakdowns.
_LAST_COLLECT_PHASES: dict[str, float] = {}


def _sync_time() -> float:
    """CUDA-synced wall time; only use when profiling."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


@dataclass(slots=True)
class Wan_2_1CollectorConfig:
    """Configuration for Wan_2_1Collector."""

    num_steps: int = 20
    guidance_scale: float = 4.5
    height: int = 240
    width: int = 416
    num_frames: int = 33
    max_sequence_length: int = 512

    # CFG during sampling
    cfg: bool = True

    # KL reward — subtract kl_reward * kl from rewards before advantages.
    kl_reward: float = 0.0

    # SDE window — only inject SDE noise for steps within the window.
    sde_window_size: int = 0
    sde_window_range: tuple[int, int] = (0, 10)

    # Same latent — reuse the same noise for samples sharing a prompt.
    same_latent: bool = False


class Wan_2_1Collector:
    """Collect rollouts from Wan 1.3B with per-step log-probabilities.

    Delegates the per-step transformer forward to the model family's
    ``DiffusionPolicy`` (``encode_prompt``, ``prepare_sampling``,
    ``forward_step``, ``decode_latents``). The collector only owns the
    SDE-step loop, reward scoring, and ``ExperienceBatch`` assembly.
    """

    def __init__(
        self,
        model: Any,  # WanT2VDiffusersPolicy
        reward_fn: Any,  # RewardFunction instance
        config: Wan_2_1CollectorConfig | None = None,
    ) -> None:
        self.model = model
        self.reward_fn = reward_fn
        self.config = config or Wan_2_1CollectorConfig()

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
        """Collect Wan Diffusers rollouts with per-step log-probabilities.

        Steps:
        1. Encode prompt via adapter (once, even for group sampling)
        2. Expand embeds to ``group_size`` when > 1 (group-batched sampling)
        3. prepare_sampling via adapter (latents + scheduler + per-family state)
        4. Custom SDE loop: adapter.forward_step per step + sde_step_with_logprob
        5. Decode via adapter.decode_latents
        6. Reward scoring
        7. Stack into ExperienceBatch

        group_size: when > 1, expands a single prompt into ``group_size``
        parallel rollouts within one denoise loop. GRPO's per-group
        advantage normalization uses these samples as a group.
        """
        import torch

        from vrl.algorithms.types import Rollout, Trajectory
        from vrl.models.diffusion import VideoGenerationRequest
        from vrl.rollouts.evaluators.diffusion.flow_matching import sde_step_with_logprob

        cfg = self.config

        # Extract structured kwargs (from PromptExample via OnlineTrainer)
        target_text = kwargs.get("target_text", "")
        references = kwargs.get("references", [])
        task_type = kwargs.get("task_type", "text_to_video")
        request_overrides = kwargs.get("request_overrides", {})
        sample_metadata = kwargs.get("sample_metadata", {})
        seed = kwargs.get("seed", None)
        group_size = int(kwargs.get("group_size", 1))

        if group_size > 1 and len(prompts) != 1:
            raise ValueError(
                f"group_size={group_size} requires exactly one prompt; got {len(prompts)}"
            )

        # Build request — apply overrides from PromptExample
        req_kwargs: dict[str, Any] = {
            "prompt": prompts[0],
            "num_steps": cfg.num_steps,
            "guidance_scale": cfg.guidance_scale,
            "height": cfg.height,
            "width": cfg.width,
            "frame_count": cfg.num_frames,
            "extra": {"max_sequence_length": cfg.max_sequence_length},
        }
        if seed is not None:
            req_kwargs["seed"] = seed
        req_kwargs.update(request_overrides)
        request = VideoGenerationRequest(**req_kwargs)

        _prof = os.environ.get("VRL_PROFILE_COLLECT") == "1"
        _phases: dict[str, float] = {}
        _t = _sync_time() if _prof else 0.0

        # 1. Encode prompt via adapter (once per prompt)
        encoded = self.model.encode_prompt(
            request.prompt,
            request.negative_prompt or None,
            max_sequence_length=cfg.max_sequence_length,
            guidance_scale=cfg.guidance_scale,
        )
        if _prof:
            _now = _sync_time()
            _phases["collect.encode_prompt"] = _now - _t
            _t = _now

        # 2. Expand embeds to group_size for parallel group sampling
        if group_size > 1:
            pe = encoded["prompt_embeds"]
            repeat_shape = (group_size,) + (1,) * (pe.ndim - 1)
            encoded["prompt_embeds"] = pe.repeat(*repeat_shape)
            neg = encoded.get("negative_prompt_embeds")
            if neg is not None:
                encoded["negative_prompt_embeds"] = neg.repeat(*repeat_shape)

        # 3. prepare_sampling via adapter (batch_size derived from prompt_embeds)
        sampling_state = self.model.prepare_sampling(request, encoded)
        # Use actual batch size from sampling state (reflects group expansion)
        batch_size = sampling_state.latents.shape[0]

        # SDE window
        sde_window = self._get_sde_window()

        # SDE noise generator — deterministic when seed is provided or same_latent
        device = sampling_state.latents.device
        if seed is not None:
            sde_generator = torch.Generator(device=device)
            sde_generator.manual_seed(seed)
        elif cfg.same_latent:
            sde_generator = torch.Generator(device=device)
            sde_generator.manual_seed(hash(prompts[0]) % (2**32))
        else:
            sde_generator = None

        # 4. Custom denoise loop with log-prob tracking
        all_observations = []
        all_actions = []
        all_log_probs = []
        all_kls = []
        all_timestep_values = []

        # Autocast dtype: pull from the wrapped pipeline's transformer so
        # the collector never introspects WanT2VSamplingState's private
        # embed tensors.
        transformer_dtype = self.model.pipeline.transformer.dtype
        total_steps = len(sampling_state.timesteps)

        with torch.amp.autocast("cuda", dtype=transformer_dtype):
            with torch.no_grad():
                for step_idx in range(total_steps):
                    latents_ori = sampling_state.latents.clone()
                    t = sampling_state.timesteps[step_idx]

                    # Forward pass via adapter
                    fwd = self.model.forward_step(sampling_state, step_idx)
                    noise_pred = fwd["noise_pred"]

                    # Check SDE window
                    in_sde_window = sde_window is None or (
                        sde_window[0] <= step_idx < sde_window[1]
                    )

                    # SDE step with log-prob (caller owns scheduler step)
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

                    # Per-step KL tracking for kl_reward
                    if cfg.kl_reward > 0:
                        all_kls.append(sde_result.log_prob.detach().abs())
                    else:
                        all_kls.append(
                            torch.zeros(batch_size, device=device)
                        )

        if _prof:
            _now = _sync_time()
            _phases["collect.denoise_loop"] = _now - _t
            _t = _now

        # Stack: [T, B, ...] -> [B, T, ...]
        observations = torch.stack(all_observations, dim=1)
        actions = torch.stack(all_actions, dim=1)
        log_probs = torch.stack(all_log_probs, dim=1)
        timesteps_tensor = torch.stack(
            [tv.expand(batch_size) for tv in all_timestep_values], dim=1
        )
        kl_tensor = torch.stack(all_kls, dim=1)

        # 5. Decode via adapter
        video = self.model.decode_latents(sampling_state.latents)
        if _prof:
            _now = _sync_time()
            _phases["collect.vae_decode"] = _now - _t
            _t = _now

        # 6. Score with reward function
        # Build rollout metadata from structured kwargs so reward functions
        # (e.g. OCRReward) can access target_text, references, etc.
        rollout_metadata: dict[str, Any] = dict(sample_metadata)
        if target_text:
            rollout_metadata["target_text"] = target_text
        if references:
            rollout_metadata["references"] = references
        rollout_metadata["task_type"] = task_type

        # With group sampling, prompts has length 1 but batch_size == group_size.
        # Each sample shares the source prompt string for reward metadata.
        effective_prompts = (
            prompts * batch_size if len(prompts) == 1 else prompts
        )

        rewards_list = []
        for i in range(batch_size):
            dummy_trajectory = Trajectory(
                prompt=effective_prompts[i],
                seed=0,
                steps=[],
                output=video[i],
            )
            dummy_rollout = Rollout(
                request=None,
                trajectory=dummy_trajectory,
                metadata=rollout_metadata,
            )
            r = await self.reward_fn.score(dummy_rollout)
            rewards_list.append(r)
        if _prof:
            _now = _sync_time()
            _phases["collect.reward_score"] = _now - _t
            _t = _now

        # 7. Subtract kl_reward * kl from rewards
        rewards_raw = torch.tensor(rewards_list, dtype=torch.float32, device=device)
        if cfg.kl_reward > 0:
            kl_total = kl_tensor.sum(dim=1)
            rewards_adjusted = rewards_raw - cfg.kl_reward * kl_total
        else:
            rewards_adjusted = rewards_raw

        # 8. Build ExperienceBatch
        dones = torch.ones(batch_size, dtype=torch.bool, device=device)
        group_ids = torch.zeros(batch_size, dtype=torch.long, device=device)

        if _prof:
            _LAST_COLLECT_PHASES.clear()
            _LAST_COLLECT_PHASES.update(_phases)

        # Project private SamplingState through the adapter boundary helpers.
        training_extras = self.model.export_training_extras(sampling_state)
        rollout_context = self.model.export_batch_context(sampling_state)

        extras: dict[str, Any] = {
            "log_probs": log_probs,
            "timesteps": timesteps_tensor,
            "kl": kl_tensor,
            "reward_before_kl": rewards_raw,
        }
        extras.update(training_extras)

        context: dict[str, Any] = dict(rollout_context)

        return ExperienceBatch(
            observations=observations,
            actions=actions,
            rewards=rewards_adjusted,
            dones=dones,
            group_ids=group_ids,
            extras=extras,
            context=context,
            videos=video,
            prompts=effective_prompts,
        )

