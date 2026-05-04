"""Rollout backend selection and validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from vrl.engine.core.runtime import RolloutBackend
from vrl.rollouts.runtime.config import RolloutBackendConfig

DRIVER_CUDA_OWNERSHIP_ERROR = (
    "Driver loaded rollout policy on CUDA. "
    "For Ray backend, set model.device=cpu so Ray actors load CUDA copies, "
    "or set distributed.rollout.allow_driver_gpu_overlap=true only for explicit "
    "colocate experiments."
)

_MISSING = object()


def validate_rollout_backend_config(
    cfg: Any,
    *,
    driver_bundle: Any | None = None,
    driver_policy: Any | None = None,
    trainable_modules: Mapping[str, Any] | Iterable[Any] | None = None,
) -> RolloutBackendConfig:
    """Validate rollout backend config before a collector touches the runtime."""
    config = RolloutBackendConfig.from_cfg(cfg)

    if not config.allow_driver_gpu_overlap and _driver_rollout_policy_on_cuda(
        driver_bundle=driver_bundle,
        driver_policy=driver_policy,
        trainable_modules=trainable_modules,
    ):
        raise ValueError(DRIVER_CUDA_OWNERSHIP_ERROR)

    return config


def build_rollout_backend_from_cfg(
    cfg: Any,
    *,
    driver_bundle: Any | None = None,
    driver_policy: Any | None = None,
    trainable_modules: Mapping[str, Any] | Iterable[Any] | None = None,
    runtime_spec: Any | None = None,
    gatherer: Any | None = None,
) -> RolloutBackend:
    """Build the Ray rollout backend selected by config.

    Ray launch requires a serializable ``runtime_spec`` plus pure chunk
    ``gatherer`` so the launcher can create rollout workers without receiving
    live model objects.
    """
    config = validate_rollout_backend_config(
        cfg,
        driver_bundle=driver_bundle,
        driver_policy=driver_policy,
        trainable_modules=trainable_modules,
    )

    runtime_spec = (
        runtime_spec
        if runtime_spec is not None
        else _cfg_path(
            cfg,
            "distributed.rollout.runtime_spec",
            None,
        )
    )
    if runtime_spec is not None and gatherer is not None:
        from vrl.distributed.ray.rollout.launcher import RayRolloutLauncher

        return RayRolloutLauncher().launch(config.to_dict(), runtime_spec, gatherer)

    raise ValueError(
        "Ray-only rollout backend requires runtime_spec plus gatherer so "
        "RayRolloutLauncher can construct workers through the "
        "runtime_builder+executor_cls path.",
    )


def _driver_rollout_policy_on_cuda(
    *,
    driver_bundle: Any | None,
    driver_policy: Any | None,
    trainable_modules: Mapping[str, Any] | Iterable[Any] | None,
) -> bool:
    policy = driver_policy
    if policy is None and driver_bundle is not None:
        policy = getattr(driver_bundle, "policy", None)

    has_policy_device, device = _get_device(policy)
    if has_policy_device:
        return _is_cuda_device(device)

    modules = trainable_modules
    if modules is None and driver_bundle is not None:
        modules = getattr(driver_bundle, "trainable_modules", None)

    return any(_is_cuda_device(device) for device in _iter_parameter_devices(modules))


def _get_device(obj: Any) -> tuple[bool, Any]:
    if obj is None:
        return False, None
    try:
        device = obj.device
    except Exception:
        return False, None
    if device is None:
        return False, None
    return True, device


def _iter_parameter_devices(obj: Any, seen: set[int] | None = None) -> Iterable[Any]:
    if obj is None or isinstance(obj, (str, bytes)):
        return
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return
    seen.add(obj_id)

    if isinstance(obj, Mapping):
        for value in obj.values():
            yield from _iter_parameter_devices(value, seen)
        return

    has_device, device = _get_device(obj)
    if has_device:
        yield device
        return

    parameters = getattr(obj, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            device = getattr(parameter, "device", None)
            if device is not None:
                yield device
        return

    if isinstance(obj, Iterable):
        for value in obj:
            yield from _iter_parameter_devices(value, seen)


def _is_cuda_device(device: Any) -> bool:
    device_type = getattr(device, "type", None)
    if device_type is not None:
        return str(device_type).lower() == "cuda"
    return str(device).lower().startswith("cuda")


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


__all__ = [
    "DRIVER_CUDA_OWNERSHIP_ERROR",
    "build_rollout_backend_from_cfg",
    "validate_rollout_backend_config",
]
