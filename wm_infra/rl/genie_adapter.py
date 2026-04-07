"""Compatibility shim for consumer-side Genie RL adapters."""

from wm_infra.consumers.rl.genie_adapter import GenieRLSpec, GenieTokenReward, GenieWorldModelAdapter

__all__ = [
    "GenieRLSpec",
    "GenieTokenReward",
    "GenieWorldModelAdapter",
]
