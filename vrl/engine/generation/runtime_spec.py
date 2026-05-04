"""Serializable generation runtime construction spec."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

_FORBIDDEN_LIVE_OBJECT_KEYS = frozenset({"executor", "policy", "pipeline"})
_SCALAR_TYPES = (str, int, float, bool, type(None))


@dataclass(frozen=True, slots=True)
class GenerationRuntimeSpec:
    """Worker-side executor construction contract for generation runtimes.

    The spec intentionally carries import paths and serializable config only.
    Distributed runtimes own model/executor construction; callers must not pass
    live executor, policy, or pipeline objects through this boundary.
    """

    family: str | None = None
    task: str | None = None
    model_config: dict[str, Any] = field(default_factory=dict)
    executor_kwargs: dict[str, Any] = field(default_factory=dict)
    policy_version: int | None = None
    runtime_builder: str | None = None
    executor_cls: str | None = None
    build_spec: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_config",
            _normalize_config_mapping(self.model_config, "model_config"),
        )
        object.__setattr__(
            self,
            "executor_kwargs",
            _normalize_config_mapping(self.executor_kwargs, "executor_kwargs"),
        )
        if self.build_spec is not None:
            object.__setattr__(
                self,
                "build_spec",
                _normalize_config_mapping(self.build_spec, "build_spec"),
            )
        if self.policy_version is not None:
            object.__setattr__(self, "policy_version", int(self.policy_version))

        for attr in ("runtime_builder", "executor_cls"):
            value = getattr(self, attr)
            if value is not None and not isinstance(value, str):
                raise TypeError(
                    f"{attr} must be an import path string, got {type(value).__name__}"
                )
        _validate_builder_mode(self)

        _assert_pickle_serializable(self)

    @classmethod
    def from_value(
        cls,
        value: GenerationRuntimeSpec | Mapping[str, Any],
    ) -> GenerationRuntimeSpec:
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls.from_dict(value)
        raise TypeError(
            "runtime_spec must be a GenerationRuntimeSpec or dict, "
            f"got {type(value).__name__}",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GenerationRuntimeSpec:
        forbidden = sorted(_FORBIDDEN_LIVE_OBJECT_KEYS.intersection(value))
        if forbidden:
            keys = ", ".join(repr(key) for key in forbidden)
            raise ValueError(
                "GenerationRuntimeSpec cannot include live object keys: "
                f"{keys}. Use runtime_builder and executor_cls import paths instead.",
            )

        field_names = {field.name for field in fields(cls)}
        unknown = sorted(set(value).difference(field_names))
        if unknown:
            keys = ", ".join(repr(key) for key in unknown)
            raise ValueError(
                "unsupported GenerationRuntimeSpec keys: "
                f"{keys}. Put model payload under model_config or executor_kwargs.",
            )

        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "task": self.task,
            "model_config": dict(self.model_config),
            "executor_kwargs": dict(self.executor_kwargs),
            "policy_version": self.policy_version,
            "runtime_builder": self.runtime_builder,
            "executor_cls": self.executor_cls,
            "build_spec": None if self.build_spec is None else dict(self.build_spec),
        }

    def build_spec_payload(self) -> dict[str, Any]:
        if self.build_spec is not None:
            return dict(self.build_spec)
        return dict(self.model_config)


def _normalize_config_mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a dict, got {type(value).__name__}")
    normalized = dict(value)
    _validate_serializable_config(normalized, path)
    return normalized


def _validate_builder_mode(spec: GenerationRuntimeSpec) -> None:
    if spec.runtime_builder is None or spec.executor_cls is None:
        raise ValueError(
            "GenerationRuntimeSpec requires runtime_builder and executor_cls import paths",
        )


def _validate_serializable_config(value: Any, path: str) -> None:
    if isinstance(value, _SCALAR_TYPES):
        return

    if _is_torch_tensor(value):
        raise TypeError(f"{path} must not contain live torch.Tensor objects")
    if _is_torch_module(value):
        raise TypeError(f"{path} must not contain live torch.nn.Module objects")
    if _is_diffusers_pipeline(value):
        raise TypeError(f"{path} must not contain live diffusers pipeline objects")
    if callable(value):
        raise TypeError(f"{path} must not contain live callable objects")

    if isinstance(value, Mapping):
        for key, inner in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings, got {type(key).__name__}")
            _validate_serializable_config(inner, f"{path}.{key}")
        return

    if isinstance(value, list):
        for index, inner in enumerate(value):
            _validate_serializable_config(inner, f"{path}[{index}]")
        return

    if isinstance(value, tuple):
        for index, inner in enumerate(value):
            _validate_serializable_config(inner, f"{path}[{index}]")
        return

    raise TypeError(
        f"{path} must contain only primitive config values, lists, tuples, and dicts; "
        f"got {type(value).__name__}",
    )


def _assert_pickle_serializable(spec: GenerationRuntimeSpec) -> None:
    try:
        pickle.dumps(spec)
    except Exception as exc:
        raise TypeError("GenerationRuntimeSpec must be pickle-serializable") from exc


def _is_torch_tensor(value: Any) -> bool:
    return _type_is_from_module(value, "torch") and value.__class__.__name__ == "Tensor"


def _is_torch_module(value: Any) -> bool:
    module_cls = getattr(getattr(value, "__class__", None), "__mro__", ())
    return any(
        cls.__module__ == "torch.nn.modules.module" and cls.__name__ == "Module"
        for cls in module_cls
    )


def _is_diffusers_pipeline(value: Any) -> bool:
    value_type = type(value)
    return value_type.__module__.startswith("diffusers.") and "Pipeline" in value_type.__name__


def _type_is_from_module(value: Any, module_prefix: str) -> bool:
    return type(value).__module__.split(".", 1)[0] == module_prefix


__all__ = ["GenerationRuntimeSpec"]
