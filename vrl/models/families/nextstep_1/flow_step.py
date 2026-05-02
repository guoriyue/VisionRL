"""Per-token flow-matching sample-with-logprob for NextStep-1.

NextStep-1 generates each image token via a small flow-matching MLP head
conditioned on the LLM's hidden state. For RL we need both the sampled
token and a Gaussian log-probability so we can form the GRPO ratio.

This mirrors ``vrl.rollouts.evaluators.diffusion.flow_matching`` but at
the per-token granularity: each AR step does its own K-step flow ODE
internally, then injects one Gaussian noise at the final step. The
log-probability is the standard isotropic-Gaussian density of that noise.

Why not full path-integral log-probability?
-------------------------------------------
NextStep-1.1's RL post-training (per the model card) treats each
generated token as Gaussian-distributed around the deterministic-flow
mean — exactly what flow_grpo's ``sde_step_with_logprob`` does for
diffusion. We follow the same convention.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class FlowStepResult:
    """Output of one AR step's flow-matching sample-with-logprob."""

    token: torch.Tensor          # [B, D_token]    — sampled continuous token
    log_prob: torch.Tensor       # [B]             — Gaussian log p(token | mean)
    mean: torch.Tensor           # [B, D_token]    — deterministic flow ODE solution
    std: torch.Tensor            # [B] or scalar   — Gaussian std at sampling step
    initial_noise: torch.Tensor  # [B, D_token]    — x_0 prior used by replay


def flow_sample_with_logprob(
    image_head: Any,
    cond: torch.Tensor,         # [B, D_hidden] — LLM last hidden state
    *,
    num_flow_steps: int = 20,
    noise_level: float = 1.0,
    cfg_uncond: torch.Tensor | None = None,
    cfg_scale: float = 1.0,
    generator: torch.Generator | None = None,
    initial_noise: torch.Tensor | None = None,
    velocity_fn: Callable[..., torch.Tensor] | None = None,
) -> FlowStepResult:
    """Sample one continuous token from the flow head and return its log-prob.

    Algorithm (matching the diffusion side's SDE-with-logprob convention):

        1. Initialise ``x_0 ~ N(0, I)`` (or use the head's prescribed prior).
        2. Run ``num_flow_steps - 1`` deterministic Euler steps on the
           flow ODE: ``x_{k+1} = x_k + dt * v(x_k, t_k, cond)``.
        3. At the final step, inject Gaussian noise of scale ``std``:
           ``token = x_{K-1} + dt * v(x_{K-1}, t_{K-1}, cond) + std * eps``.
        4. ``log_prob = -0.5 * ||eps||^2 / std^2 - D_token * log(std)
                       - 0.5 * D_token * log(2 pi)``.

    Args:
        image_head: NextStep-1's flow-matching MLP (``model.image_head``).
            Must expose either:
              - ``image_head.velocity(x, t, cond)`` returning ``[B, D]`` v(x,t),
                OR
              - ``image_head(x, t, cond)`` (forward) returning v(x,t).
            We call ``velocity_fn`` if provided, else fall back to forward.
        cond: ``[B, D_hidden]`` LLM hidden state at this AR position.
        num_flow_steps: Number of Euler steps inside the flow ODE.
        noise_level: Scales the final-step Gaussian std (analogue of the
            ``a`` knob in flow_grpo's SDE-from-ODE conversion). 0 → fully
            deterministic (zero log-prob mass), 1 → unit-variance noise.
        cfg_uncond: ``[B, D_hidden]`` unconditional hidden state for CFG.
            When provided, velocity is computed as
                v_guided = (1 + s) * v(x, cond) - s * v(x, uncond)
            with ``s = cfg_scale``. Skip if cfg_scale ≈ 1.
        cfg_scale: CFG strength on the velocity.
        generator: Optional torch.Generator for reproducibility.
        initial_noise: Optional explicit ``x_0`` prior. When provided, this
            exact tensor is used as the deterministic flow prefix and returned
            as ``FlowStepResult.initial_noise`` for replay.
        velocity_fn: Optional override for how to call image_head. If None,
            we try ``image_head.velocity(...)`` then ``image_head(...)``.

    Returns:
        ``FlowStepResult`` with the sampled token and its scalar log-prob.

    NOTE
    ----
    The exact velocity-call signature depends on NextStep-1's upstream
    implementation. Until we have ``stepfun-ai/NextStep-1`` installed we
    cannot bind this — see the ``# TODO(nextstep-binding)`` markers.
    """
    B, D = cond.shape[0], None  # D inferred from x_0 below
    device = cond.device
    dtype = cond.dtype

    # NextStep-1's FlowMatchingHead stores the token latent dim as
    # ``image_head.input_dim`` = num_channels * patch_size².
    token_dim = getattr(image_head, "input_dim", None)
    if token_dim is None:
        raise RuntimeError(
            "flow_sample_with_logprob: image_head has no ``input_dim`` "
            "attribute — upstream API may have changed. Inspect "
            "FlowMatchingHead.__init__ in modeling_nextstep.py."
        )
    D = int(token_dim)

    # x_0 ~ N(0, I) — flow-matching prior is standard normal. This is the
    # replay artifact saved by NextStepPolicy; when supplied explicitly, do
    # not sample another prior inside this helper.
    if initial_noise is None:
        x = torch.randn(B, D, device=device, dtype=dtype, generator=generator)
    else:
        if initial_noise.shape != (B, D):
            raise ValueError(
                "initial_noise must have shape "
                f"{(B, D)}, got {tuple(initial_noise.shape)}"
            )
        x = initial_noise.to(device=device, dtype=dtype)
    x0 = x

    # Linear time grid t in [0, 1]
    t_grid = torch.linspace(0.0, 1.0, num_flow_steps + 1, device=device, dtype=dtype)
    dt = 1.0 / num_flow_steps

    def _velocity(xk: torch.Tensor, tk: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Upstream FlowMatchingHead exposes its velocity predictor as
        # ``image_head.net`` (a SimpleMLPAdaLN). Caller may override.
        if velocity_fn is not None:
            return velocity_fn(xk, tk, c)
        return image_head.net(xk, tk, c)

    # K-1 deterministic Euler steps
    for k in range(num_flow_steps - 1):
        tk = t_grid[k].expand(B)
        v_cond = _velocity(x, tk, cond)
        if cfg_uncond is not None and abs(cfg_scale - 1.0) > 1e-6:
            v_uncond = _velocity(x, tk, cfg_uncond)
            v = (1.0 + cfg_scale) * v_cond - cfg_scale * v_uncond
        else:
            v = v_cond
        x = x + dt * v

    # Final step with Gaussian noise injection (the source of the log-prob)
    tk = t_grid[num_flow_steps - 1].expand(B)
    v_cond = _velocity(x, tk, cond)
    if cfg_uncond is not None and abs(cfg_scale - 1.0) > 1e-6:
        v_uncond = _velocity(x, tk, cfg_uncond)
        v = (1.0 + cfg_scale) * v_cond - cfg_scale * v_uncond
    else:
        v = v_cond

    mean = x + dt * v

    # SDE-from-ODE: std = noise_level * sqrt(dt) (matches flow_grpo's
    # final-step parameterisation; sigma_min ≪ sigma_max in flat schedule)
    std_scalar = noise_level * math.sqrt(dt)
    std = torch.full((B,), std_scalar, device=device, dtype=dtype)

    eps = torch.randn(
        mean.shape,
        device=mean.device,
        dtype=mean.dtype,
        generator=generator,
    )
    token = mean + std_scalar * eps

    # Isotropic Gaussian log-prob, summed across token dim, then mean per
    # sample. We use SUM (not mean) over D so that ratios of fresh/old
    # log-probs stay correctly scaled across batch entries.
    sq_err = ((token.detach() - mean) ** 2).sum(dim=-1)  # [B]
    log_prob = (
        -sq_err / (2.0 * std_scalar ** 2)
        - float(D) * math.log(max(std_scalar, 1e-12))
        - 0.5 * float(D) * math.log(2.0 * math.pi)
    )

    return FlowStepResult(
        token=token,
        log_prob=log_prob,
        mean=mean,
        std=std,
        initial_noise=x0,
    )


def flow_logprob_at(
    image_head: Any,
    cond: torch.Tensor,         # [B, D_hidden]
    target_token: torch.Tensor, # [B, D_token]
    saved_noise: torch.Tensor | None = None,
    *,
    num_flow_steps: int = 20,
    noise_level: float = 1.0,
    cfg_uncond: torch.Tensor | None = None,
    cfg_scale: float = 1.0,
    velocity_fn: Callable[..., torch.Tensor] | None = None,
) -> torch.Tensor:
    """Recompute log-prob of a previously-sampled continuous token.

    The replay assumes the deterministic flow ODE prefix is **the same
    seed-determined trajectory** as collection time — which we can only
    guarantee if we either (a) save the prior x_0 and any per-step noise
    as ``saved_noise``, or (b) accept a Monte-Carlo approximation by
    re-running with a fresh prior.

    Strategy implemented here: re-run the deterministic prefix with the
    *current* policy's velocity field, take its terminal mean ``mu``, and
    compute log p(target_token | mu, std). This matches what
    ``sde_step_with_logprob`` does in the diffusion path — the "old" and
    "fresh" log-probs differ only because the velocity field has been
    updated by SGD, which is exactly what GRPO's ratio is supposed to
    capture.

    Returns:
        ``[B]`` log-probabilities with grad flowing through ``image_head``.
    """
    B, D = target_token.shape
    device = target_token.device
    dtype = target_token.dtype

    if saved_noise is not None:
        # Reproducible mode: caller has stashed x_0 and used it everywhere.
        x = saved_noise.to(device=device, dtype=dtype)
    else:
        # Fallback mode: fresh x_0 ~ N(0, I). Biased, but matches what the
        # SD3/Wan flow-matching path does today (they also don't replay
        # collection-time noise verbatim).
        x = torch.randn(B, D, device=device, dtype=dtype)

    t_grid = torch.linspace(0.0, 1.0, num_flow_steps + 1, device=device, dtype=dtype)
    dt = 1.0 / num_flow_steps

    def _velocity(xk: torch.Tensor, tk: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if velocity_fn is not None:
            return velocity_fn(xk, tk, c)
        if hasattr(image_head, "net"):
            return image_head.net(xk, tk, c)
        if hasattr(image_head, "velocity"):
            return image_head.velocity(xk, tk, c)  # type: ignore[no-any-return]
        return image_head(xk, tk, c)

    for k in range(num_flow_steps - 1):
        tk = t_grid[k].expand(B)
        v_cond = _velocity(x, tk, cond)
        if cfg_uncond is not None and abs(cfg_scale - 1.0) > 1e-6:
            v_uncond = _velocity(x, tk, cfg_uncond)
            v = (1.0 + cfg_scale) * v_cond - cfg_scale * v_uncond
        else:
            v = v_cond
        x = x + dt * v

    tk = t_grid[num_flow_steps - 1].expand(B)
    v_cond = _velocity(x, tk, cond)
    if cfg_uncond is not None and abs(cfg_scale - 1.0) > 1e-6:
        v_uncond = _velocity(x, tk, cfg_uncond)
        v = (1.0 + cfg_scale) * v_cond - cfg_scale * v_uncond
    else:
        v = v_cond

    mean = x + dt * v
    std_scalar = noise_level * math.sqrt(dt)

    sq_err = ((target_token - mean) ** 2).sum(dim=-1)
    log_prob = (
        -sq_err / (2.0 * std_scalar ** 2)
        - float(D) * math.log(max(std_scalar, 1e-12))
        - 0.5 * float(D) * math.log(2.0 * math.pi)
    )
    return log_prob
