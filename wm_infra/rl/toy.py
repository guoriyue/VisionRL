"""Compatibility shim for consumer-side toy world models."""

from wm_infra.consumers.rl.toy import ToyContinuousWorldModel, ToyLineWorldModel, ToyLineWorldSpec, ToyWorldSpec

__all__ = [
    "ToyContinuousWorldModel",
    "ToyLineWorldModel",
    "ToyLineWorldSpec",
    "ToyWorldSpec",
]
