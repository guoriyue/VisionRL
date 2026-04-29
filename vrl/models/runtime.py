"""Runtime build spec and runtime bundle (CONTRACT.md, SPRINT_model_refactor.md §5.1, §5.3.E).

These two dataclasses are the only sanctioned interface between training scripts
and family-adjacent builders. Scripts must not import diffusers / native /
official backend classes directly; builders consume a ``RuntimeBuildSpec`` and
return a ``RuntimeBundle``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vrl.models.diffusion import DiffusionPolicy


@dataclass
class RuntimeBuildSpec:
    """Runtime-only slice of the whole RL config.

    Builders take this, not the whole RL cfg. Reward / algorithm / trainer /
    dataset / logging cadence are explicitly out of scope.
    """

    model_name_or_path: str
    device: Any
    dtype: Any
    backend_preference: tuple[str, ...] = ("diffusers",)
    task_variant: str | None = None
    mixed_precision: str | None = None
    use_lora: bool = False
    lora_path: str | None = None
    lora_config: dict[str, Any] | None = None
    offload_config: dict[str, Any] | None = None
    scheduler_config: dict[str, Any] | None = None
    native_backend_config: dict[str, Any] | None = None
    diffusers_backend_config: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeBundle:
    """Sole output of a family builder. Consumed by scripts / collectors / trainers.

    ``backend_handle`` carries the raw backend object (e.g. diffusers pipeline)
    and must be treated as builder-internal — scripts and trainers must not
    reach into it. Use ``policy`` for inference, ``trainable_modules``
    for optimizer wiring, ``scheduler`` for evaluator construction.
    """

    policy: DiffusionPolicy
    trainable_modules: dict[str, Any]
    scheduler: Any
    backend_kind: str
    backend_handle: Any
    ref_modules: dict[str, Any] | None = None
    runtime_caps: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
