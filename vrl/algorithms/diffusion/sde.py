"""Diffusion SDE math shared by generation and training replay."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SDEStepResult:
    """Result of a single SDE denoising step with log-probability."""

    prev_sample: Any
    log_prob: Any
    prev_sample_mean: Any
    std_dev_t: Any
    dt: Any | None = None


def sde_step_with_logprob(
    scheduler: Any,
    model_output: Any,
    timestep: Any,
    sample: Any,
    prev_sample: Any | None = None,
    generator: Any | None = None,
    deterministic: bool = False,
    return_dt: bool = False,
    noise_level: float = 1.0,
    sde_type: str = "sde",
) -> SDEStepResult:
    """Compute one diffusion SDE step and its per-sample log-probability."""
    import torch
    from diffusers.utils.torch_utils import randn_tensor

    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    step_index = [scheduler.index_for_timestep(t) for t in timestep]
    prev_step_index = [s + 1 for s in step_index]

    scheduler.sigmas = scheduler.sigmas.to(sample.device)
    ndim = sample.ndim
    view_shape = (-1,) + (1,) * (ndim - 1)

    sigma = scheduler.sigmas[step_index].view(*view_shape)
    sigma_prev = scheduler.sigmas[prev_step_index].view(*view_shape)
    sigma_max = scheduler.sigmas[1].item()
    sigma_min = scheduler.sigmas[-1].item()
    dt = sigma_prev - sigma

    if sde_type == "cps":
        std_dev_t = sigma_prev * math.sin(noise_level * math.pi / 2)
        pred_original_sample = sample - sigma * model_output
        noise_estimate = sample + model_output * (1 - sigma)
        prev_sample_mean = (
            pred_original_sample * (1 - sigma_prev)
            + noise_estimate * torch.sqrt(sigma_prev**2 - std_dev_t**2)
        )

        if prev_sample is not None and generator is not None:
            raise ValueError("Cannot pass both generator and prev_sample.")

        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * variance_noise

        if deterministic:
            prev_sample = sample + dt * model_output

        log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    else:
        std_dev_t = sigma_min + (sigma_max - sigma_min) * sigma

        if noise_level != 1.0:
            std_dev_t = torch.sqrt(
                sigma
                / (
                    1
                    - torch.where(
                        sigma == 1,
                        torch.tensor(sigma_max, device=sigma.device),
                        sigma,
                    )
                )
            ) * noise_level

        prev_sample_mean = (
            sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
            + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
        )

        if prev_sample is not None and generator is not None:
            raise ValueError("Cannot pass both generator and prev_sample.")

        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = (
                prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise
            )

        if deterministic:
            prev_sample = sample + dt * model_output

        noise_scale = std_dev_t * torch.sqrt(-1 * dt)
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * noise_scale**2)
            - torch.log(noise_scale)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    sqrt_neg_dt = torch.sqrt(-1 * dt) if return_dt else None
    return SDEStepResult(
        prev_sample=prev_sample,
        log_prob=log_prob,
        prev_sample_mean=prev_sample_mean,
        std_dev_t=std_dev_t,
        dt=sqrt_neg_dt,
    )


def compute_kl_divergence(
    prev_sample_mean: Any,
    prev_sample_mean_ref: Any,
    std_dev_t: Any,
    dt: Any | None = None,
) -> Any:
    """KL divergence between current and reference model in latent space."""
    denom = 2 * std_dev_t**2
    if dt is not None:
        denom = 2 * (std_dev_t * dt) ** 2
    return ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(
        dim=tuple(range(1, prev_sample_mean.ndim))
    ) / denom.squeeze()


__all__ = [
    "SDEStepResult",
    "compute_kl_divergence",
    "sde_step_with_logprob",
]
