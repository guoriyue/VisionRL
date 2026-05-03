"""Shared fused-stage helpers for diffusion family executors."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch

from vrl.algorithms.flow_matching import sde_step_with_logprob
from vrl.engine.generation.protocols import PipelineChunkResult
from vrl.engine.generation.types import (
    GenerationMetrics,
    GenerationRequest,
    GenerationSampleSpec,
    OutputBatch,
    RolloutDenoisingEnv,
    RolloutDitTrajectory,
    RolloutTrajectoryData,
)


@dataclass(frozen=True, slots=True)
class DiffusionDenoiseConfig:
    """Runtime knobs for one diffusion micro-batch denoise loop."""

    sample_start: int
    seed: int | None
    same_latent: bool
    sde_window: tuple[int, int] | None
    return_kl: bool
    noise_level: float = 1.0
    sde_type: str = "sde"


@dataclass(slots=True)
class DiffusionChunkResult(PipelineChunkResult):
    """Output of one fused diffusion micro-batch."""

    observations: Any
    actions: Any
    log_probs: Any
    timesteps: Any
    kl: Any
    video: Any
    training_extras: dict[str, Any]
    context: dict[str, Any]


def select_sde_window(
    sde_window_size: int,
    sde_window_range: tuple[int, int] | list[int],
) -> tuple[int, int] | None:
    """Pick the stochastic denoise-step window for a request."""

    if sde_window_size <= 0:
        return None
    lo, hi = int(sde_window_range[0]), int(sde_window_range[1])
    start = random.randint(lo, max(lo, hi - sde_window_size))
    return (start, start + sde_window_size)


def repeat_tensor_batch(value: Any, count: int) -> Any:
    """Repeat a tensor whose first dimension is a singleton batch."""

    if count < 1:
        raise ValueError("count must be >= 1")
    if not isinstance(value, torch.Tensor):
        return value
    if count == 1:
        return value
    repeat_shape = (count,) + (1,) * (value.ndim - 1)
    return value.repeat(*repeat_shape)


def run_diffusion_denoise_chunk(
    *,
    policy: Any,
    request: Any,
    encoded: dict[str, Any],
    config: DiffusionDenoiseConfig,
    prepare_kwargs: dict[str, Any] | None = None,
) -> DiffusionChunkResult:
    """Run one fused diffusion micro-batch: prepare -> denoise -> decode."""

    state = policy.prepare_sampling(request, encoded, **(prepare_kwargs or {}))
    chunk_batch = state.latents.shape[0]
    device = state.latents.device
    generator = _build_generator(
        device=device,
        sample_start=config.sample_start,
        seed=config.seed,
        same_latent=config.same_latent,
    )

    obs_steps: list[Any] = []
    act_steps: list[Any] = []
    lp_steps: list[Any] = []
    kl_steps: list[Any] = []
    t_steps: list[Any] = []

    prompt_embeds = encoded.get("prompt_embeds")
    transformer_dtype = (
        prompt_embeds.dtype
        if isinstance(prompt_embeds, torch.Tensor)
        else state.latents.dtype
    )
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=transformer_dtype)
        if torch.cuda.is_available()
        else _NullCtx()
    )

    with autocast_ctx, torch.no_grad():
        for step_idx in range(len(state.timesteps)):
            latents_ori = state.latents.clone()
            timestep = state.timesteps[step_idx]
            fwd = policy.forward_step(state, step_idx)
            noise_pred = fwd["noise_pred"]

            in_sde_window = config.sde_window is None or (
                config.sde_window[0] <= step_idx < config.sde_window[1]
            )
            sde_result = sde_step_with_logprob(
                state.scheduler,
                noise_pred.float(),
                timestep.unsqueeze(0),
                state.latents.float(),
                generator=generator if in_sde_window else None,
                deterministic=not in_sde_window,
                return_dt=config.return_kl,
                noise_level=config.noise_level,
                sde_type=config.sde_type,
            )
            prev_latents = sde_result.prev_sample
            state.latents = prev_latents

            obs_steps.append(latents_ori.detach())
            act_steps.append(prev_latents.detach())
            lp_steps.append(sde_result.log_prob.detach())
            t_steps.append(timestep.detach())
            if config.return_kl:
                kl_steps.append(sde_result.log_prob.detach().abs())
            else:
                kl_steps.append(torch.zeros(chunk_batch, device=device))

    observations = torch.stack(obs_steps, dim=1)
    actions = torch.stack(act_steps, dim=1)
    log_probs = torch.stack(lp_steps, dim=1)
    timesteps = torch.stack(
        [timestep.expand(chunk_batch) for timestep in t_steps],
        dim=1,
    )
    kl = torch.stack(kl_steps, dim=1)
    video = policy.decode_latents(state.latents)

    return DiffusionChunkResult(
        observations=observations,
        actions=actions,
        log_probs=log_probs,
        timesteps=timesteps,
        kl=kl,
        video=video,
        training_extras=policy.export_training_extras(state),
        context=policy.export_batch_context(state),
    )


def build_diffusion_output_batch(
    *,
    request: GenerationRequest,
    sample_specs: list[GenerationSampleSpec],
    prompts: list[str],
    chunks: list[DiffusionChunkResult],
    num_steps: int,
) -> OutputBatch:
    """Pack diffusion chunks into the canonical engine OutputBatch."""

    if not chunks:
        raise ValueError("chunks must be non-empty")

    observations = torch.cat([chunk.observations for chunk in chunks], dim=0)
    actions = torch.cat([chunk.actions for chunk in chunks], dim=0)
    log_probs = torch.cat([chunk.log_probs for chunk in chunks], dim=0)
    timesteps_tensor = torch.cat([chunk.timesteps for chunk in chunks], dim=0)
    kl_tensor = torch.cat([chunk.kl for chunk in chunks], dim=0)
    video = torch.cat([chunk.video for chunk in chunks], dim=0)
    training_extras = _concat_training_extras(
        [chunk.training_extras for chunk in chunks]
    )
    rollout_context = chunks[0].context
    if not rollout_context:
        raise ValueError("DiffusionChunkResult.context must be non-empty")

    denoising_env = RolloutDenoisingEnv(
        extra={
            "actions": actions,
            "kl": kl_tensor,
            "training_extras": training_extras,
            "context": rollout_context,
            "videos": video,
        },
    )
    dit_trajectory = RolloutDitTrajectory(
        latents=observations,
        timesteps=timesteps_tensor,
    )
    rollout_trajectory_data = RolloutTrajectoryData(
        rollout_log_probs=log_probs,
        denoising_env=denoising_env,
        dit_trajectory=dit_trajectory,
    )
    peak_mem_mb = peak_memory_mb()
    metrics = GenerationMetrics(
        num_prompts=len(prompts),
        num_samples=len(sample_specs),
        num_steps=num_steps,
        micro_batches=len(chunks),
        peak_memory_mb=peak_mem_mb,
    )

    return OutputBatch(
        request_id=request.request_id,
        family=request.family,
        task=request.task,
        prompts=prompts,
        sample_specs=sample_specs,
        output=video,
        rollout_trajectory_data=rollout_trajectory_data,
        metrics=metrics,
        peak_memory_mb=peak_mem_mb or 0.0,
    )


def peak_memory_mb() -> float | None:
    """Return CUDA peak memory if available."""

    if not torch.cuda.is_available():
        return None
    try:
        peak_bytes = torch.cuda.max_memory_allocated()
    except Exception:
        return None
    return peak_bytes / (1024 * 1024)


def _build_generator(
    *,
    device: Any,
    sample_start: int,
    seed: int | None,
    same_latent: bool,
) -> torch.Generator | None:
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + sample_start)
        return generator
    if same_latent:
        raise ValueError("same_latent=True requires an explicit sampling.seed")
    return None


def _concat_training_extras(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    training_extras: dict[str, Any] = {}
    if not chunks:
        return training_extras
    for key in chunks[0]:
        vals = [chunk[key] for chunk in chunks]
        if any(value is None for value in vals):
            training_extras[key] = None
        elif all(isinstance(value, torch.Tensor) for value in vals):
            training_extras[key] = torch.cat(vals, dim=0)
        else:
            training_extras[key] = vals[0]
    return training_extras


class _NullCtx:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None


__all__ = [
    "DiffusionChunkResult",
    "DiffusionDenoiseConfig",
    "build_diffusion_output_batch",
    "peak_memory_mb",
    "repeat_tensor_batch",
    "run_diffusion_denoise_chunk",
    "select_sde_window",
]
