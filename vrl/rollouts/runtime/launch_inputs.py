"""Rollout runtime input builders for distributed generation backends."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from vrl.engine.core.runtime_spec import GenerationRuntimeSpec
from vrl.engine.gather import ChunkGatherer
from vrl.rollouts.families.specs import (
    RolloutFamilyEntry,
    get_rollout_family_entry,
)
from vrl.rollouts.runtime.config import RolloutBackendConfig


@dataclass(frozen=True, slots=True)
class RolloutRuntimeInputs:
    """Serializable worker spec plus driver-side pure gatherer."""

    runtime_spec: GenerationRuntimeSpec
    gatherer: ChunkGatherer


def build_rollout_runtime_inputs(
    cfg: Any,
    family: str,
    *,
    weight_dtype: Any,
    executor_kwargs: Mapping[str, Any] | None = None,
    policy_version: int = 0,
) -> RolloutRuntimeInputs:
    """Build Ray rollout launcher inputs from a family registry entry."""

    rollout_config = RolloutBackendConfig.from_cfg(cfg)

    entry = get_rollout_family_entry(family)
    rollout_device = "cuda" if rollout_config.gpus_per_worker > 0 else "cpu"
    runtime_build_spec = _call_runtime_spec_extractor(
        entry,
        cfg,
        rollout_device,
        _dtype_to_string(weight_dtype),
    )
    resolved_executor_kwargs = _build_executor_kwargs(entry, cfg)
    resolved_executor_kwargs.update(dict(executor_kwargs or {}))

    return RolloutRuntimeInputs(
        runtime_spec=GenerationRuntimeSpec(
            family=entry.family,
            task=entry.task,
            build_spec=_runtime_build_spec_payload(runtime_build_spec),
            executor_kwargs=resolved_executor_kwargs,
            policy_version=policy_version,
            runtime_builder=entry.runtime_builder,
            executor_cls=entry.executor_cls,
        ),
        gatherer=_build_gatherer(entry),
    )


def _call_runtime_spec_extractor(
    entry: RolloutFamilyEntry,
    cfg: Any,
    device: str,
    weight_dtype: str,
) -> Any:
    extractor = _import_from_path(entry.runtime_spec_extractor)
    return extractor(cfg, device, weight_dtype)


def _build_gatherer(entry: RolloutFamilyEntry) -> ChunkGatherer:
    gatherer_cls = _import_from_path(entry.gatherer.import_path)
    return gatherer_cls(**entry.gatherer.kwargs)


def _build_executor_kwargs(entry: RolloutFamilyEntry, cfg: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    metadata = entry.executor_kwargs
    if metadata.include_sample_batch_size:
        sample_batch_size = _cfg_path(cfg, "rollout.sample_batch_size", None)
        if sample_batch_size is not None:
            kwargs["sample_batch_size"] = int(sample_batch_size)
    if metadata.include_reference_image:
        reference_image = _cfg_path(cfg, "model.reference_image", None)
        if reference_image:
            kwargs["reference_image"] = str(reference_image)
    return kwargs


def _runtime_build_spec_payload(spec: Any) -> dict[str, Any]:
    payload = asdict(spec)
    payload["device"] = _device_to_string(payload["device"])
    payload["dtype"] = _dtype_to_string(payload["dtype"])
    return payload


def _import_from_path(path: str) -> Any:
    module_path, attr = path.split(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _device_to_string(value: Any) -> str:
    return str(value)


def _dtype_to_string(value: Any) -> str:
    text = str(value)
    return text.removeprefix("torch.")


_MISSING = object()


def _cfg_path(cfg: Any, path: str, default: Any) -> Any:
    node = cfg
    for key in path.split("."):
        node = _cfg_get(node, key, _MISSING)
        if node is _MISSING:
            return default
    return node


def _cfg_get(node: Any, key: str, default: Any) -> Any:
    if node is None:
        return default
    getter = getattr(node, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    try:
        return node[key]
    except (KeyError, IndexError, TypeError):
        pass
    return getattr(node, key, default)


__all__ = ["RolloutRuntimeInputs", "build_rollout_runtime_inputs"]
