"""Ray rollout worker for chunk-level generation execution."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from vrl.distributed.ray.actor import RayActorBase
from vrl.distributed.ray.spec import RolloutRuntimeSpec
from vrl.distributed.ray.types import RayChunkResult
from vrl.distributed.ray.utils import import_from_path
from vrl.engine.generation.gather import require_chunked_executor
from vrl.engine.generation.types import GenerationRequest
from vrl.executors.base import ChunkedFamilyPipelineExecutor
from vrl.executors.microbatching import MicroBatchPlan


class RayRolloutWorker(RayActorBase):
    """Own one rollout executor inside a Ray actor process."""

    def __init__(
        self,
        worker_id: str,
        family: str,
        runtime_spec: RolloutRuntimeSpec | Mapping[str, Any],
    ) -> None:
        self.worker_id = worker_id
        self.runtime_spec = _normalize_runtime_spec(runtime_spec, family=family)
        self.family = self.runtime_spec.family or family
        self.executor: ChunkedFamilyPipelineExecutor | None = None
        self._policy_version: int | None = self.runtime_spec.policy_version

    def load_policy(self) -> None:
        """Build the family executor from the serialized runtime spec."""

        if self.executor is not None:
            return
        self.executor = _build_executor(self.runtime_spec)

    def update_weights(self, state_ref: Any, policy_version: int) -> None:
        """Update rollout weights, then record the active policy version."""

        self.load_policy()
        policy = getattr(self.executor, "model", None)
        if policy is not None:
            apply_state = getattr(policy, "load_trainable_state", None)
            if callable(apply_state) and state_ref is not None:
                apply_state(state_ref)
        self._policy_version = int(policy_version)

    def current_policy_version(self) -> int | None:
        return self._policy_version

    def worker_metadata(self) -> dict[str, Any]:
        try:
            node_ip = self.get_node_ip()
            gpu_ids = self.get_gpu_ids()
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
    runtime_spec: RolloutRuntimeSpec | Mapping[str, Any],
    *,
    family: str,
) -> RolloutRuntimeSpec:
    spec = RolloutRuntimeSpec.from_value(runtime_spec)
    if spec.family is not None and spec.family != family:
        raise ValueError(
            "RayRolloutWorker family mismatch: "
            f"worker family={family!r}, runtime_spec.family={spec.family!r}",
        )
    if spec.family is None:
        spec = replace(spec, family=family)
    return spec


def _build_executor(runtime_spec: RolloutRuntimeSpec) -> ChunkedFamilyPipelineExecutor:
    factory_path = runtime_spec.executor_factory
    if factory_path is not None:
        factory = import_from_path(str(factory_path))
        built = factory(runtime_spec)
        return require_chunked_executor(built)

    builder_path = runtime_spec.runtime_builder
    executor_path = runtime_spec.executor_cls
    if builder_path is None or executor_path is None:
        raise ValueError(
            "RayRolloutWorker requires runtime_spec with 'executor_factory' "
            "or ('runtime_builder' and 'executor_cls') import paths",
        )

    from vrl.models.runtime import RuntimeBuildSpec

    build_runtime_bundle = import_from_path(str(builder_path))
    executor_cls = import_from_path(str(executor_path))
    bundle = build_runtime_bundle(RuntimeBuildSpec(**runtime_spec.build_spec_payload()))
    built = executor_cls(bundle.policy, **dict(runtime_spec.executor_kwargs))
    return require_chunked_executor(built)


def _to_cpu(value: Any) -> Any:
    if _is_tensor(value):
        return value.detach().cpu()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        payload = {
            field.name: _to_cpu(getattr(value, field.name))
            for field in dataclasses.fields(value)
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
