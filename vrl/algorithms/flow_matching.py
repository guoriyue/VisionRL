"""Backward-compatibility shim for diffusion SDE math."""

from vrl.algorithms.diffusion.sde import (  # noqa: F401
    SDEStepResult,
    compute_kl_divergence,
    sde_step_with_logprob,
)
