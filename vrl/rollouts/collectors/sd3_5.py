"""SD3.5-Medium collector for Flow-GRPO image RL training.

Collector → FlowMatchingEvaluator → GRPO → OnlineTrainer

Mirrors ``wan_2_1.py`` but for image generation:
- 4D latents [B, C, H, W] (no temporal dim)
- ``sample_batch_size`` controls rollout-side micro-batching: when
  ``group_size > sample_batch_size``, the group is split into
  ``ceil(group_size / sample_batch_size)`` micro-rollouts run sequentially,
  trading wall-time for VRAM. Per-prompt advantage normalization sees the
  full ``group_size`` set, matching paper's distributed G=24 setup on a
  single GPU.
- SDE step uses ``sde_type="cps"`` with paper noise_level a=0.7 (Eq.9).

Adapter contract (post-refactor):
``self.model`` is a ``SD3_5Policy`` exposing five methods —
``encode_prompt``, ``prepare_sampling``, ``forward_step``,
``decode_latents`` (and the inherited ``inference``). The collector owns
the SDE step / log-prob bookkeeping; the adapter owns one transformer
forward per step.
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

_LAST_COLLECT_PHASES: dict[str, float] = {}


def _sync_time() -> float:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


@dataclass(slots=True)
class SD3_5CollectorConfig:
    """Configuration for SD3_5Collector."""

    # Diffusion sampling — paper B.3: sampling T=10, eval T=40, resolution 512
    num_steps: int = 10
    guidance_scale: float = 4.5
    height: int = 512
    width: int = 512
    max_sequence_length: int = 256

    # Flow-GRPO Eq.9 noise level a (paper B.3: a=0.7)
    noise_level: float = 0.7

    # CFG during sampling
    cfg: bool = True

    # Rollout-side micro-batching: when group_size > sample_batch_size,
    # split group into ceil(group/sample_batch_size) sequential micro-rollouts.
    # Trades wall-time for VRAM so single-GPU can match paper's G=24.
    sample_batch_size: int = 8

    # KL reward — subtract kl_reward * kl from rewards before advantages.
    kl_reward: float = 0.0

    # SDE window — only inject SDE noise for steps within the window.
    sde_window_size: int = 0
    sde_window_range: tuple[int, int] = (0, 10)

    # Same latent — reuse the same noise for samples sharing a prompt.
    same_latent: bool = False


class SD3_5Collector:
    """Collect rollouts from SD3.5-M with per-step log-probabilities.

    Group rollouts of size ``group_size`` are split into micro-rollouts
    of size ``sample_batch_size``; each micro-rollout runs an independent
    denoise loop and the trajectories are concatenated on the batch dim
    into a single ``ExperienceBatch`` so per-prompt advantage normalization
    sees the full group.
    """

    def __init__(
        self,
        model: Any,  # SD3_5Policy
        reward_fn: Any,  # RewardFunction instance
        config: SD3_5CollectorConfig | None = None,
    ) -> None:
        self.model = model
        self.reward_fn = reward_fn
        self.config = config or SD3_5CollectorConfig()

    def _get_sde_window(self) -> tuple[int, int] | None:
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
        """Collect SD3 rollouts with per-step log-probabilities.

        Steps:
        1. Encode prompt via adapter (once per prompt)
        2. Split group_size into micro-batches of sample_batch_size
        3. For each micro-batch: prepare_sampling → SDE loop → decode_latents
        4. Reward scoring on all samples
        5. Concatenate micro-batch trajectories into one ExperienceBatch
        """
        import torch

        from vrl.algorithms.types import Rollout, Trajectory
        from vrl.models.diffusion import VideoGenerationRequest
        from vrl.rollouts.evaluators.diffusion.flow_matching import sde_step_with_logprob

        cfg = self.config

        target_text = kwargs.get("target_text", "")
        references = kwargs.get("references", [])
        task_type = kwargs.get("task_type", "text_to_image")
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
            "frame_count": 1,  # SD3 is image-only
            "extra": {"max_sequence_length": cfg.max_sequence_length},
        }
        if seed is not None:
            req_kwargs["seed"] = seed
        req_kwargs.update(request_overrides)
        request = VideoGenerationRequest(**req_kwargs)

        _prof = os.environ.get("VRL_PROFILE_COLLECT") == "1"
        _phases: dict[str, float] = {}
        _t = _sync_time() if _prof else 0.0

        # 1. Encode prompt once (re-used across micro-batches)
        encoded = self.model.encode_prompt(
            prompts[0],
            request.negative_prompt or None,
            max_sequence_length=cfg.max_sequence_length,
            guidance_scale=cfg.guidance_scale,
        )
        if _prof:
            _now = _sync_time()
            _phases["collect.encode_prompt"] = _now - _t
            _t = _now

        # 2. Plan micro-batches — split group_size into chunks of sample_batch_size
        sample_bs = max(1, cfg.sample_batch_size)
        if group_size <= sample_bs:
            chunk_sizes = [group_size]
        else:
            n_full = group_size // sample_bs
            remainder = group_size - n_full * sample_bs
            chunk_sizes = [sample_bs] * n_full
            if remainder > 0:
                chunk_sizes.append(remainder)

        # 3. Run micro-rollouts; collect per-chunk tensors then concat
        all_obs_chunks: list[Any] = []
        all_act_chunks: list[Any] = []
        all_lp_chunks: list[Any] = []
        all_t_chunks: list[Any] = []
        all_kl_chunks: list[Any] = []
        all_video_chunks: list[Any] = []
        # Per-chunk training-extras dicts (prompt_embeds etc.) — populated
        # via adapter.export_training_extras and concat'd along dim=0 after
        # the loop. Keeps the collector opaque to SD3SamplingState fields.
        all_extras_chunks: list[dict[str, Any]] = []
        rollout_context: dict[str, Any] | None = None
        do_cfg = cfg.cfg and cfg.guidance_scale > 1.0
        sde_window = self._get_sde_window()

        chunk_offset = 0
        for chunk_idx, chunk_g in enumerate(chunk_sizes):
            # Build per-chunk encoded dict by repeating embeds chunk_g times
            pe = encoded["prompt_embeds"]
            pp = encoded["pooled_prompt_embeds"]
            repeat_seq = (chunk_g,) + (1,) * (pe.ndim - 1)
            repeat_pool = (chunk_g,) + (1,) * (pp.ndim - 1)
            chunk_encoded: dict[str, Any] = {
                "prompt_embeds": pe.repeat(*repeat_seq),
                "pooled_prompt_embeds": pp.repeat(*repeat_pool),
            }
            neg = encoded.get("negative_prompt_embeds")
            neg_pool = encoded.get("negative_pooled_prompt_embeds")
            if neg is not None:
                chunk_encoded["negative_prompt_embeds"] = neg.repeat(*repeat_seq)
            if neg_pool is not None:
                chunk_encoded["negative_pooled_prompt_embeds"] = neg_pool.repeat(
                    *repeat_pool,
                )

            # prepare_sampling for this chunk
            state = self.model.prepare_sampling(request, chunk_encoded)
            chunk_batch = state.latents.shape[0]
            device = state.latents.device

            # Per-chunk SDE generator (deterministic when seed is provided)
            if seed is not None:
                gen = torch.Generator(device=device)
                gen.manual_seed(seed + chunk_offset)
            elif cfg.same_latent:
                gen = torch.Generator(device=device)
                gen.manual_seed(
                    (hash(prompts[0]) + chunk_offset) % (2**32)
                )
            else:
                gen = None

            obs_steps: list[Any] = []
            act_steps: list[Any] = []
            lp_steps: list[Any] = []
            kl_steps: list[Any] = []
            t_steps: list[Any] = []
            transformer_dtype = chunk_encoded["prompt_embeds"].dtype

            with torch.amp.autocast("cuda", dtype=transformer_dtype):
                with torch.no_grad():
                    for step_idx in range(len(state.timesteps)):
                        latents_ori = state.latents.clone()
                        t = state.timesteps[step_idx]

                        fwd = self.model.forward_step(state, step_idx)
                        noise_pred = fwd["noise_pred"]

                        in_sde_window = sde_window is None or (
                            sde_window[0] <= step_idx < sde_window[1]
                        )

                        sde_result = sde_step_with_logprob(
                            state.scheduler,
                            noise_pred.float(),
                            t.unsqueeze(0),
                            state.latents.float(),
                            generator=gen if in_sde_window else None,
                            deterministic=not in_sde_window,
                            return_dt=cfg.kl_reward > 0,
                            noise_level=cfg.noise_level,
                            sde_type="cps",
                        )
                        prev_latents = sde_result.prev_sample
                        state.latents = prev_latents

                        obs_steps.append(latents_ori.detach())
                        act_steps.append(prev_latents.detach())
                        lp_steps.append(sde_result.log_prob.detach())
                        t_steps.append(t.detach())

                        if cfg.kl_reward > 0:
                            kl_steps.append(sde_result.log_prob.detach().abs())
                        else:
                            kl_steps.append(
                                torch.zeros(chunk_batch, device=device)
                            )

            # Stack chunk-step tensors → [chunk_batch, T, ...]
            obs_chunk = torch.stack(obs_steps, dim=1)
            act_chunk = torch.stack(act_steps, dim=1)
            lp_chunk = torch.stack(lp_steps, dim=1)
            t_chunk = torch.stack(
                [tv.expand(chunk_batch) for tv in t_steps], dim=1
            )
            kl_chunk = torch.stack(kl_steps, dim=1)

            # Decode latents for this chunk
            video_chunk = self.model.decode_latents(state.latents)

            all_obs_chunks.append(obs_chunk)
            all_act_chunks.append(act_chunk)
            all_lp_chunks.append(lp_chunk)
            all_t_chunks.append(t_chunk)
            all_kl_chunks.append(kl_chunk)
            all_video_chunks.append(video_chunk)

            # Pull per-chunk training extras from the adapter (opaque to
            # SamplingState internals); concat after the loop on dim=0.
            all_extras_chunks.append(self.model.export_training_extras(state))

            # Capture rollout-level context once — guidance_scale / do_cfg
            # are shared across chunks, so any chunk's state suffices.
            if rollout_context is None:
                rollout_context = self.model.export_batch_context(state)

            chunk_offset += chunk_g

        if _prof:
            _now = _sync_time()
            _phases["collect.denoise_loop_all_chunks"] = _now - _t
            _t = _now

        # 4. Concat all chunks on batch dim
        observations = torch.cat(all_obs_chunks, dim=0)
        actions = torch.cat(all_act_chunks, dim=0)
        log_probs = torch.cat(all_lp_chunks, dim=0)
        timesteps_tensor = torch.cat(all_t_chunks, dim=0)
        kl_tensor = torch.cat(all_kl_chunks, dim=0)
        video = torch.cat(all_video_chunks, dim=0)

        batch_size = observations.shape[0]
        device = observations.device

        # 5. Score with reward function
        rollout_metadata: dict[str, Any] = dict(sample_metadata)
        if target_text:
            rollout_metadata["target_text"] = target_text
        if references:
            rollout_metadata["references"] = references
        rollout_metadata["task_type"] = task_type

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
        rewards_raw = torch.tensor(
            rewards_list, dtype=torch.float32, device=device,
        )
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

        # Concat per-chunk training extras along the batch dim (None entries
        # — e.g. negative_*_embeds when CFG is off — propagate as None).
        training_extras: dict[str, Any] = {}
        if all_extras_chunks:
            for key in all_extras_chunks[0].keys():
                vals = [c[key] for c in all_extras_chunks]
                if any(v is None for v in vals):
                    training_extras[key] = None
                else:
                    training_extras[key] = torch.cat(vals, dim=0)

        extras: dict[str, Any] = {
            "log_probs": log_probs,
            "timesteps": timesteps_tensor,
            "kl": kl_tensor,
            "reward_before_kl": rewards_raw,
        }
        extras.update(training_extras)

        # rollout_context is populated by the first chunk; fall back to a
        # synthesized minimal dict if no chunks ran (defensive — chunk_sizes
        # is non-empty in practice).
        if rollout_context is None:
            rollout_context = {
                "guidance_scale": cfg.guidance_scale,
                "cfg": do_cfg,
                "model_family": self.model.family,
            }

        return ExperienceBatch(
            observations=observations,
            actions=actions,
            rewards=rewards_adjusted,
            dones=dones,
            group_ids=group_ids,
            extras=extras,
            context=rollout_context,
            videos=video,
            prompts=effective_prompts,
        )

