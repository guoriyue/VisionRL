"""Factory and validation helpers for collector-facing rollout backends."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from vrl.distributed.ray.config import DistributedRolloutConfig
from vrl.engine.generation.runtime import RolloutBackend

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
) -> DistributedRolloutConfig:
    """Validate rollout backend config before a collector touches the runtime."""
    config = DistributedRolloutConfig.from_cfg(cfg)

    if config.backend == "local" and config.num_workers != 1:
        raise ValueError(
            "local rollout backend requires distributed.rollout.num_workers=1; "
            f"got {config.num_workers}",
        )

    if (
        config.backend == "ray"
        and not config.allow_driver_gpu_overlap
        and _driver_rollout_policy_on_cuda(
            driver_bundle=driver_bundle,
            driver_policy=driver_policy,
            trainable_modules=trainable_modules,
        )
    ):
        raise ValueError(DRIVER_CUDA_OWNERSHIP_ERROR)

    return config


def build_rollout_backend_from_cfg(
    cfg: Any,
    *,
    runtime: RolloutBackend | None = None,
    local_runtime_builder: Callable[[], RolloutBackend] | None = None,
    driver_bundle: Any | None = None,
    driver_policy: Any | None = None,
    trainable_modules: Mapping[str, Any] | Iterable[Any] | None = None,
    runtime_spec: Any | None = None,
    gatherer: Any | None = None,
) -> RolloutBackend:
    """Build or return the rollout backend selected by config.

    Ray callers may inject an already-built runtime. If no runtime is injected,
    local builds use ``local_runtime_builder`` and Ray builds require a
    serializable ``runtime_spec`` plus pure chunk ``gatherer`` so the launcher
    can create Ray rollout workers without receiving live model objects.
    """
    config = validate_rollout_backend_config(
        cfg,
        driver_bundle=driver_bundle,
        driver_policy=driver_policy,
        trainable_modules=trainable_modules,
    )

    if runtime is not None:
        return _require_rollout_backend(runtime)

    if config.backend == "local":
        if local_runtime_builder is None:
            raise ValueError(
                "local rollout backend requires local_runtime_builder when no runtime is injected",
            )
        return _require_rollout_backend(local_runtime_builder())

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

        return RayRolloutLauncher().launch(config, runtime_spec, gatherer)

    raise ValueError(
        "Ray rollout backend requires an injected runtime, or runtime_spec plus "
        "gatherer so RayRolloutLauncher can construct workers through the "
        "single runtime_builder+executor_cls path.",
    )


def build_local_generation_runtime(
    *,
    model: Any,
    family: str,
    task: str,
    cfg: Any,
    executor_kwargs: Mapping[str, Any] | None = None,
) -> RolloutBackend:
    """Build the local in-process GenerationRuntime for one family executor."""
    if model is None:
        raise RuntimeError(
            "local generation runtime requires a live model; inject a runtime "
            "for distributed rollout",
        )

    from vrl.engine import EngineLoop, Scheduler
    from vrl.engine.generation.registry import FamilyPipelineRegistry
    from vrl.engine.generation.runtime import (
        GenerationBatchPlanner,
        GenerationModelRunner,
        GenerationRuntime,
    )
    from vrl.engine.generation.worker import GenerationWorker

    executor_cls = _executor_cls_for(family, task)
    registry = FamilyPipelineRegistry()
    registry.register(executor_cls(model, **dict(executor_kwargs or {})))
    worker = GenerationWorker(registry)
    runner = GenerationModelRunner(worker, execute_in_thread=False)
    engine_loop = EngineLoop(
        scheduler=Scheduler(
            batch_planner=GenerationBatchPlanner(
                max_batch_size=int(_cfg_get(cfg, "max_batch_requests", 1)),
            ),
        ),
        model_runner=runner,
    )
    return GenerationRuntime(engine_loop)


def _require_rollout_backend(runtime: Any) -> RolloutBackend:
    if not callable(getattr(runtime, "generate", None)):
        raise TypeError(
            "rollout runtime must implement async generate(request) -> OutputBatch",
        )
    return runtime


def _executor_cls_for(family: str, task: str) -> type:
    if family == "sd3_5" and task == "t2i":
        from vrl.models.families.sd3_5.executor import SD3_5PipelineExecutor

        return SD3_5PipelineExecutor
    if family == "wan_2_1" and task == "t2v":
        from vrl.models.families.wan_2_1.executor import Wan_2_1PipelineExecutor

        return Wan_2_1PipelineExecutor
    if family == "cosmos" and task == "v2w":
        from vrl.models.families.cosmos.executor import CosmosPipelineExecutor

        return CosmosPipelineExecutor
    if family == "janus_pro" and task == "ar_t2i":
        from vrl.models.families.janus_pro.executor import JanusProPipelineExecutor

        return JanusProPipelineExecutor
    if family == "nextstep_1" and task == "ar_t2i":
        from vrl.models.families.nextstep_1.executor import NextStep1PipelineExecutor

        return NextStep1PipelineExecutor
    raise NotImplementedError(f"no local generation executor for {family}/{task}")


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
    "build_local_generation_runtime",
    "build_rollout_backend_from_cfg",
    "validate_rollout_backend_config",
]
