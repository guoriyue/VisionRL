"""Tests for canonical flow-matching math exports."""

from __future__ import annotations


def test_flow_matching_is_canonical_math_export() -> None:
    from vrl.algorithms import flow_matching
    from vrl.algorithms.flow_matching import (
        SDEStepResult,
        compute_kl_divergence,
        sde_step_with_logprob,
    )

    assert flow_matching.SDEStepResult is SDEStepResult
    assert flow_matching.compute_kl_divergence is compute_kl_divergence
    assert flow_matching.sde_step_with_logprob is sde_step_with_logprob


def test_diffusion_evaluator_package_exports_evaluator_only() -> None:
    from vrl.rollouts.evaluators.diffusion import FlowMatchingEvaluator

    assert FlowMatchingEvaluator.__name__ == "FlowMatchingEvaluator"
