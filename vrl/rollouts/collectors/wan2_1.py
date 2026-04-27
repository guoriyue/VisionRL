"""Wan2.1 T2V collector for RL training (1.3B and 14B variants).

Collector → FlowMatchingEvaluator → GRPO → OnlineTrainer
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
class Wan21CollectorConfig:
    """Configuration for Wan21Collector."""

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


class Wan21Collector:
    """Collect rollouts from Wan 1.3B with per-step log-probabilities.

    Delegates model-specific forward passes to the model family's
    ``denoise_init`` / ``predict_noise`` / ``decode_vae`` methods.
    The collector only owns the SDE-step loop, reward scoring,
    and ``ExperienceBatch`` assembly.
    """

    def __init__(
        self,
        model: Any,  # DiffusersWanT2VModel
        reward_fn: Any,  # RewardFunction instance
        config: Wan21CollectorConfig | None = None,
    ) -> None:
        self.model = model
        self.reward_fn = reward_fn
        self.config = config or Wan21CollectorConfig()

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
        1. Encode text via model family (once, even for group sampling)
        2. Expand embeds to ``group_size`` when > 1 (group-batched sampling)
        3. denoise_init via model family (prepares latents + scheduler)
        4. Custom SDE loop: model.predict_noise per step + sde_step_with_logprob
        5. Decode VAE via model family
        6. Reward scoring
        7. Stack into ExperienceBatch

        group_size: when > 1, expands a single prompt into ``group_size``
        parallel rollouts within one denoise loop. GRPO's per-group
        advantage normalization uses these samples as a group.
        """
        import torch

        from vrl.algorithms.types import Rollout, Trajectory
        from vrl.models.base import VideoGenerationRequest
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

        # 1. Encode text via model family (once per prompt)
        state: dict[str, Any] = {}
        encode_result = await self.model.encode_text(request, state)
        state.update(encode_result.state_updates)
        if _prof:
            _now = _sync_time()
            _phases["collect.encode_text"] = _now - _t
            _t = _now

        # 2. Expand embeds to group_size for parallel group sampling
        if group_size > 1:
            pe = state["prompt_embeds"]
            repeat_shape = (group_size,) + (1,) * (pe.ndim - 1)
            state["prompt_embeds"] = pe.repeat(*repeat_shape)
            neg = state.get("negative_prompt_embeds")
            if neg is not None:
                state["negative_prompt_embeds"] = neg.repeat(*repeat_shape)

        # 3. denoise_init via model family (batch_size derived from prompt_embeds)
        denoise_loop = await self.model.denoise_init(request, state)
        ms = denoise_loop.model_state
        # Use actual batch size from model state (reflects group expansion)
        batch_size = ms.latents.shape[0]

        # SDE window
        sde_window = self._get_sde_window()

        # SDE noise generator — deterministic when seed is provided or same_latent
        device = ms.latents.device
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

        do_cfg = cfg.cfg and cfg.guidance_scale > 1.0
        transformer_dtype = ms.prompt_embeds.dtype

        with torch.amp.autocast("cuda", dtype=transformer_dtype):
            with torch.no_grad():
                for step_idx in range(denoise_loop.total_steps):
                    latents_ori = ms.latents.clone()
                    t = ms.timesteps[step_idx]

                    # Forward pass via model family
                    fwd = await self.model.predict_noise(denoise_loop, step_idx)
                    noise_pred = fwd["noise_pred"]

                    # Check SDE window
                    in_sde_window = sde_window is None or (
                        sde_window[0] <= step_idx < sde_window[1]
                    )

                    # SDE step with log-prob
                    sde_result = sde_step_with_logprob(
                        ms.scheduler,
                        noise_pred.float(),
                        t.unsqueeze(0),
                        ms.latents.float(),
                        generator=sde_generator if in_sde_window else None,
                        deterministic=not in_sde_window,
                        return_dt=cfg.kl_reward > 0,
                    )
                    prev_latents = sde_result.prev_sample
                    ms.latents = prev_latents

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

                    denoise_loop.current_step = step_idx + 1

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

        # 4. Decode VAE via model family
        decode_state: dict[str, Any] = {"latents": ms.latents}
        decode_result = await self.model.decode_vae(request, decode_state)
        video = decode_result.state_updates["video"]
        if _prof:
            _now = _sync_time()
            _phases["collect.vae_decode"] = _now - _t
            _t = _now

        # 5. Score with reward function
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

        if _prof:
            _LAST_COLLECT_PHASES.clear()
            _LAST_COLLECT_PHASES.update(_phases)

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
                "prompt_embeds": ms.prompt_embeds,
                "negative_prompt_embeds": ms.negative_prompt_embeds,
            },
            context={
                "guidance_scale": ms.guidance_scale,
                "cfg": do_cfg,
                "model_family": "wan-diffusers-t2v",
            },
            videos=video,
            prompts=effective_prompts,
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
        """Wan Diffusers forward: single transformer + optional CFG.

        Used by the evaluator to compute fresh log-probs under the
        current policy during training.  Delegates to the model family's
        ``_predict_noise_with_model``.
        """
        from vrl.engine.model_executor.execution_state import DenoiseLoopState
        from vrl.models.families.diffusers_state import DiffusersDenoiseState

        # Read per-sample tensors from extras
        timesteps = batch.extras["timesteps"]
        prompt_embeds = batch.extras["prompt_embeds"]
        negative_prompt_embeds = batch.extras["negative_prompt_embeds"]
        latents = batch.observations[:, timestep_idx]

        # Read shared metadata from context
        ctx = batch.context
        guidance_scale = ctx["guidance_scale"]
        do_cfg = ctx["cfg"]

        t = timesteps[:, timestep_idx] if timesteps.ndim > 1 else timesteps

        # Reconstruct DiffusersDenoiseState for the model family
        ms = DiffusersDenoiseState(
            latents=latents,
            timesteps=t,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            do_cfg=do_cfg and guidance_scale > 1.0,
            model_family="wan-diffusers-t2v",
        )

        ds = DenoiseLoopState(current_step=0, total_steps=1, model_state=ms)
        return self.model._predict_noise_with_model(model, ds, step_idx=0)
