"""Ray rollout worker for chunk-level generation execution."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from vrl.distributed.ray.dependencies import current_gpu_ids, current_node_ip
from vrl.distributed.ray.module_loading import import_from_path
from vrl.distributed.ray.rollout.types import RayChunkResult
from vrl.engine.core.protocols import ChunkedFamilyPipelineExecutor
from vrl.engine.core.runtime_spec import GenerationRuntimeSpec
from vrl.engine.core.types import GenerationRequest
from vrl.engine.gather import require_chunked_executor
from vrl.engine.microbatching import MicroBatchPlan


class RayRolloutWorker:
    """Own one rollout executor inside a Ray actor process."""

    def __init__(
        self,
        worker_id: str,
        runtime_spec: GenerationRuntimeSpec | Mapping[str, Any],
    ) -> None:
        self.worker_id = worker_id
        self.runtime_spec = _normalize_runtime_spec(runtime_spec)
        self.family = self.runtime_spec.family
        self.executor: ChunkedFamilyPipelineExecutor | None = None
        self._policy_version: int | None = self.runtime_spec.policy_version

    def load_policy(self) -> None:
        """Build the family executor from the serialized runtime spec."""

        if self.executor is not None:
            return
        self.executor = _build_executor(self.runtime_spec)

    def release_policy(self) -> None:
        """Drop loaded model state so the actor releases CUDA memory before exit."""

        self.executor = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def update_weights(self, state_ref: Any, policy_version: int) -> None:
        """Update rollout weights, then record the active policy version."""

        self.load_policy()
        policy = getattr(self.executor, "model", None)
        if state_ref is not None:
            if policy is None:
                raise RuntimeError(
                    f"{type(self.executor).__name__} must expose model for weight sync",
                )
            apply_state = getattr(policy, "load_trainable_state", None)
            if not callable(apply_state):
                raise RuntimeError(
                    f"{type(policy).__name__} must implement load_trainable_state()",
                )
            apply_state(state_ref)
        self._policy_version = int(policy_version)

    def current_policy_version(self) -> int | None:
        return self._policy_version

    def worker_metadata(self) -> dict[str, Any]:
        try:
            node_ip = current_node_ip()
            gpu_ids = current_gpu_ids()
        except Exception:
            node_ip = "local"
            gpu_ids = []
        return {
            "worker_id": self.worker_id,
            "node_ip": node_ip,
            "gpu_ids": gpu_ids,
            "policy_version": self._policy_version,
        }

    def execute_chunk(
        self,
        request: GenerationRequest,
        chunk: MicroBatchPlan,
    ) -> RayChunkResult:
        self.load_policy()
        expected_version = request.policy_version
        if expected_version is not None and self._policy_version != expected_version:
            return RayChunkResult(
                request_id=request.request_id,
                worker_id=self.worker_id,
                chunk=chunk,
                output=None,
                metrics=self.worker_metadata(),
                policy_version=self._policy_version,
                error=(
                    "policy_version mismatch: "
                    f"expected={expected_version}, actual={self._policy_version}"
                ),
            )
        try:
            assert self.executor is not None
            output = self.executor.forward_chunk(request, chunk)
            return RayChunkResult(
                request_id=request.request_id,
                worker_id=self.worker_id,
                chunk=chunk,
                output=_to_cpu(output),
                metrics=self.worker_metadata(),
                policy_version=self._policy_version,
            )
        except Exception as exc:
            return RayChunkResult(
                request_id=request.request_id,
                worker_id=self.worker_id,
                chunk=chunk,
                output=None,
                metrics=self.worker_metadata(),
                policy_version=self._policy_version,
                error=str(exc),
            )


def _normalize_runtime_spec(
    runtime_spec: GenerationRuntimeSpec | Mapping[str, Any],
) -> GenerationRuntimeSpec:
    spec = GenerationRuntimeSpec.from_value(runtime_spec)
    if spec.family is None:
        raise ValueError("GenerationRuntimeSpec.family is required")
    return spec


def _build_executor(runtime_spec: GenerationRuntimeSpec) -> ChunkedFamilyPipelineExecutor:
    builder_path = runtime_spec.runtime_builder
    executor_path = runtime_spec.executor_cls
    if builder_path is None or executor_path is None:
        raise ValueError(
            "GenerationRuntimeSpec requires runtime_builder and executor_cls import paths",
        )

    from vrl.models.runtime import RuntimeBuildSpec

    build_runtime_bundle = import_from_path(str(builder_path))
    executor_cls = import_from_path(str(executor_path))
    bundle = build_runtime_bundle(
        RuntimeBuildSpec(
            **_normalize_runtime_build_spec_payload(
                runtime_spec.build_spec_payload(),
            )
        ),
    )
    built = executor_cls(bundle.policy, **dict(runtime_spec.executor_kwargs))
    return require_chunked_executor(built)


def _normalize_runtime_build_spec_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    device = normalized.get("device")
    if isinstance(device, str):
        import torch

        normalized["device"] = torch.device(device)
    dtype = normalized.get("dtype")
    if isinstance(dtype, str):
        normalized["dtype"] = _torch_dtype_from_string(dtype)
    return normalized


def _torch_dtype_from_string(value: str) -> Any:
    import torch

    key = value.removeprefix("torch.").lower()
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(f"unsupported torch dtype string in runtime_spec: {value!r}") from exc


def _to_cpu(value: Any) -> Any:
    if _is_tensor(value):
        return value.detach().cpu()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        payload = {
            field.name: _to_cpu(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
        return type(value)(**payload)
    if isinstance(value, dict):
        return {key: _to_cpu(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_to_cpu(inner) for inner in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(inner) for inner in value)
    return value


def _is_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "cpu")


__all__ = ["RayRolloutWorker"]
