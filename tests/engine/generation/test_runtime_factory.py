"""Tests for rollout runtime factory fail-fast behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from omegaconf import OmegaConf

from vrl.engine.generation import OutputBatch
from vrl.rollouts.backend import (
    DRIVER_CUDA_OWNERSHIP_ERROR,
    build_rollout_backend_from_cfg,
    validate_rollout_backend_config,
)


class _FakeRuntime:
    async def generate(self, request: Any) -> OutputBatch:
        return OutputBatch(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=list(request.prompts),
            sample_specs=[],
            output=None,
        )


class _CudaPolicy:
    device = "cuda:0"


class _CpuPolicy:
    device = "cpu"


class _FakeParameter:
    def __init__(self, device: str) -> None:
        self.device = device


class _FakeModule:
    def __init__(self, device: str) -> None:
        self._parameters = [_FakeParameter(device)]

    def parameters(self) -> list[_FakeParameter]:
        return self._parameters


@dataclass
class _Bundle:
    policy: Any
    trainable_modules: dict[str, Any]


def _cfg(*, backend: str = "local", num_workers: int = 1, overlap: bool = False):
    return OmegaConf.create(
        {
            "distributed": {
                "backend": backend,
                "rollout": {
                    "num_workers": num_workers,
                    "allow_driver_gpu_overlap": overlap,
                },
            },
        },
    )


def test_local_backend_builds_injected_local_runtime() -> None:
    runtime = _FakeRuntime()
    calls = 0

    def build_local() -> _FakeRuntime:
        nonlocal calls
        calls += 1
        return runtime

    backend = build_rollout_backend_from_cfg(
        _cfg(),
        local_runtime_builder=build_local,
    )

    assert backend is runtime
    assert calls == 1


def test_local_backend_rejects_multiple_workers_before_building_runtime() -> None:
    calls = 0

    def build_local() -> _FakeRuntime:
        nonlocal calls
        calls += 1
        return _FakeRuntime()

    with pytest.raises(ValueError, match="num_workers=1"):
        build_rollout_backend_from_cfg(
            _cfg(num_workers=2),
            local_runtime_builder=build_local,
        )

    assert calls == 0


def test_ray_backend_without_runtime_spec_does_not_build_local_runtime() -> None:
    calls = 0

    def build_local() -> _FakeRuntime:
        nonlocal calls
        calls += 1
        return _FakeRuntime()

    with pytest.raises(ValueError, match="requires an injected runtime"):
        build_rollout_backend_from_cfg(
            _cfg(backend="ray"),
            local_runtime_builder=build_local,
            driver_policy=_CpuPolicy(),
        )

    assert calls == 0


def test_ray_backend_runtime_spec_without_gatherer_fails_clearly() -> None:
    with pytest.raises(ValueError, match="runtime_spec plus gatherer"):
        build_rollout_backend_from_cfg(
            _cfg(backend="ray"),
            runtime_spec={
                "family": "fake",
                "task": "t2i",
                "runtime_builder": "tests.fake:build_runtime",
                "executor_cls": "tests.fake:Executor",
            },
            driver_policy=_CpuPolicy(),
        )


def test_ray_backend_rejects_driver_cuda_policy_without_overlap() -> None:
    with pytest.raises(ValueError, match="Driver loaded rollout policy on CUDA"):
        validate_rollout_backend_config(
            _cfg(backend="ray"),
            driver_policy=_CudaPolicy(),
        )

    assert "model.device=cpu" in DRIVER_CUDA_OWNERSHIP_ERROR


def test_ray_backend_accepts_explicit_runtime_without_building_local_runtime() -> None:
    runtime = _FakeRuntime()

    def build_local() -> _FakeRuntime:
        raise AssertionError("Ray backend must not build local GenerationRuntime")

    backend = build_rollout_backend_from_cfg(
        _cfg(backend="ray"),
        runtime=runtime,
        local_runtime_builder=build_local,
        driver_policy=_CpuPolicy(),
    )

    assert backend is runtime


def test_ray_backend_detects_cuda_trainable_module_when_policy_has_no_device() -> None:
    bundle = _Bundle(
        policy=object(),
        trainable_modules={"transformer": _FakeModule("cuda:1")},
    )

    with pytest.raises(ValueError, match="Driver loaded rollout policy on CUDA"):
        validate_rollout_backend_config(
            _cfg(backend="ray"),
            driver_bundle=bundle,
        )


def test_ray_backend_allows_driver_cuda_policy_with_explicit_overlap() -> None:
    config = validate_rollout_backend_config(
        _cfg(backend="ray", overlap=True),
        driver_policy=_CudaPolicy(),
    )

    assert config.backend == "ray"
    assert config.allow_driver_gpu_overlap is True
