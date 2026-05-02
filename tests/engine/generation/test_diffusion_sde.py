"""Tests for canonical diffusion SDE math exports."""

from __future__ import annotations


def test_diffusion_sde_is_canonical_export() -> None:
    from vrl.rollouts.diffusion.sde import (
        SDEStepResult,
        compute_kl_divergence,
        sde_step_with_logprob,
    )
    from vrl.rollouts.evaluators.diffusion import flow_matching

    assert flow_matching.SDEStepResult is SDEStepResult
    assert flow_matching.compute_kl_divergence is compute_kl_divergence
    assert flow_matching.sde_step_with_logprob is sde_step_with_logprob
