"""Explicit rollout collector registry."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Literal

from vrl.engine import RolloutBackend
from vrl.rollouts.collector.core import RolloutCollector
from vrl.rollouts.collector.requests import (
    RolloutEngineRequestBuilder,
    RolloutRequestBuilder,
)
from vrl.rollouts.collector.rewards import RewardScorer
from vrl.rollouts.families.specs import FAMILY_REGISTRY, get_rollout_family_entry
from vrl.rollouts.packers.ar_continuous import ARContinuousRolloutPacker
from vrl.rollouts.packers.ar_discrete import ARDiscreteRolloutPacker
from vrl.rollouts.packers.diffusion import DiffusionRolloutPacker

CollectorKind = Literal["diffusion", "ar_discrete", "ar_continuous"]

LAST_COLLECT_PHASES: dict[str, float] = {}


@dataclass(frozen=True, slots=True)
class CollectorRegistryEntry:
    """Declarative binding from explicit family name to collector components."""

    family: str
    task: str
    kind: CollectorKind
    config_cls: type
    executor_cls: type
    request_prefix: str | None = None
    default_task_type: str | None = None
    error_prefix: str | None = None
    sampling_fields: tuple[str, ...] = ()
    return_artifacts: tuple[str, ...] = ()
    metadata_key: str | None = None


def _import_from_path(path: str) -> Any:
    module_path, attr = path.split(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr)


COLLECTOR_REGISTRY: dict[str, CollectorRegistryEntry] = {
    family: CollectorRegistryEntry(
        family=entry.family,
        task=entry.task,
        kind=entry.collector.kind,
        config_cls=_import_from_path(entry.collector.config_cls),
        executor_cls=_import_from_path(entry.executor_cls),
        request_prefix=entry.collector.request_prefix,
        default_task_type=entry.collector.default_task_type,
        error_prefix=entry.collector.error_prefix,
        sampling_fields=entry.collector.sampling_fields,
        return_artifacts=entry.collector.return_artifacts,
        metadata_key=entry.collector.metadata_key,
    )
    for family, entry in FAMILY_REGISTRY.items()
}


def build_rollout_collector(
    family: str,
    *,
    model: Any | None,
    reward_fn: Any | None,
    config: Any | None = None,
    runtime: RolloutBackend | None = None,
    reference_image: Any = None,
) -> RolloutCollector:
    """Build a rollout collector from an explicit family registry key."""

    family_entry = get_rollout_family_entry(family)
    entry = _entry_for(family_entry.family)
    collector_config = _resolve_config(entry, config)
    request_builder = _build_request_builder(entry, collector_config)
    packer = _build_packer(entry)
    executor_kwargs = _build_executor_kwargs(entry, collector_config, reference_image)

    return RolloutCollector(
        model=model,
        config=collector_config,
        family=entry.family,
        task=entry.task,
        executor_cls=entry.executor_cls,
        request_builder=request_builder,
        packer=packer,
        reward_scorer=RewardScorer(reward_fn),
        default_group_size=_default_group_size(entry, collector_config),
        runtime=runtime,
        executor_kwargs=executor_kwargs,
        phase_sink=LAST_COLLECT_PHASES,
    )


def collector_config_cls(family: str) -> type:
    """Return the config schema for an explicit family registry key."""

    return _entry_for(get_rollout_family_entry(family).family).config_cls


def _entry_for(family: str) -> CollectorRegistryEntry:
    try:
        return COLLECTOR_REGISTRY[family]
    except KeyError as exc:
        raise NotImplementedError(
            f"no rollout collector registered for family={family!r}; "
            f"registered={sorted(COLLECTOR_REGISTRY)}",
        ) from exc


def _resolve_config(entry: CollectorRegistryEntry, config: Any | None) -> Any:
    if config is None:
        return entry.config_cls()
    if not isinstance(config, entry.config_cls):
        raise TypeError(
            f"{entry.family} collector requires {entry.config_cls.__name__}, "
            f"got {type(config).__name__}",
        )
    return config


def _build_request_builder(
    entry: CollectorRegistryEntry,
    config: Any,
) -> RolloutRequestBuilder:
    if entry.kind in {"diffusion", "ar_discrete", "ar_continuous"}:
        if entry.request_prefix is None:
            raise ValueError(f"{entry.family} collector registry entry is incomplete")
        return RolloutEngineRequestBuilder(
            family=entry.family,
            task=entry.task,
            request_prefix=entry.request_prefix,
            config=config,
            sampling_fields=entry.sampling_fields,
            return_artifacts=entry.return_artifacts,
            default_task_type=entry.default_task_type,
            metadata_key=entry.metadata_key,
        )
    raise AssertionError(f"unhandled collector kind: {entry.kind}")


def _build_packer(entry: CollectorRegistryEntry) -> Any:
    if entry.kind == "diffusion":
        if entry.error_prefix is None:
            raise ValueError(f"{entry.family} diffusion registry entry is incomplete")
        return DiffusionRolloutPacker(error_prefix=entry.error_prefix)
    if entry.kind == "ar_discrete":
        return ARDiscreteRolloutPacker()
    if entry.kind == "ar_continuous":
        return ARContinuousRolloutPacker()
    raise AssertionError(f"unhandled collector kind: {entry.kind}")


def _build_executor_kwargs(
    entry: CollectorRegistryEntry,
    config: Any,
    reference_image: Any,
) -> dict[str, Any]:
    if entry.kind != "diffusion":
        return {}
    kwargs: dict[str, Any] = {"sample_batch_size": config.sample_batch_size}
    if reference_image is not None:
        kwargs["reference_image"] = reference_image
    return kwargs


def _default_group_size(entry: CollectorRegistryEntry, config: Any) -> int:
    if entry.kind == "diffusion":
        return 1
    return int(config.n_samples_per_prompt)


__all__ = [
    "COLLECTOR_REGISTRY",
    "LAST_COLLECT_PHASES",
    "CollectorRegistryEntry",
    "build_rollout_collector",
    "collector_config_cls",
]
