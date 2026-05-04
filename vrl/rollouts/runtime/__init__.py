"""Rollout runtime backend wiring."""

from vrl.rollouts.runtime.backend import (
    DRIVER_CUDA_OWNERSHIP_ERROR,
    build_rollout_backend_from_cfg,
    validate_rollout_backend_config,
)
from vrl.rollouts.runtime.config import RolloutBackendConfig
from vrl.rollouts.runtime.launch_inputs import (
    RolloutRuntimeInputs,
    build_rollout_runtime_inputs,
)

__all__ = [
    "DRIVER_CUDA_OWNERSHIP_ERROR",
    "RolloutBackendConfig",
    "RolloutRuntimeInputs",
    "build_rollout_backend_from_cfg",
    "build_rollout_runtime_inputs",
    "validate_rollout_backend_config",
]
