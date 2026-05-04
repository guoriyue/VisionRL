"""Tests for rollout runtime factory fail-fast behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from omegaconf import OmegaConf

from vrl.engine import OutputBatch
from vrl.engine.core.runtime_spec import GenerationRuntimeSpec
from vrl.rollouts.runtime.backend import (
    DRIVER_CUDA_OWNERSHIP_ERROR,
    build_rollout_backend_from_cfg,
    validate_rollout_backend_config,
)
from vrl.rollouts.runtime.config import RolloutBackendConfig


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


class _FakeGatherer:
    def gather_chunks(self, request: Any, sample_specs: Any, chunks: Any) -> OutputBatch:
        del sample_specs, chunks
        return OutputBatch(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=list(request.prompts),
            sample_specs=[],
            output=None,
        )


def _runtime_spec() -> GenerationRuntimeSpec:
    return GenerationRuntimeSpec(
        family="fake",
        task="t2i",
        runtime_builder="tests.fake:build_runtime",
        executor_cls="tests.fake:Executor",
    )


def _cfg(*, backend: str = "ray", num_workers: int = 1, overlap: bool = False):
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


def test_rollout_backend_config_from_cfg_requires_explicit_backend() -> None:
    with pytest.raises(ValueError, match=r"distributed\.backend or backend"):
        RolloutBackendConfig.from_cfg({})


def test_rollout_backend_config_from_cfg_rejects_non_ray_backend() -> None:
    with pytest.raises(ValueError, match="backend must be 'ray'"):
        RolloutBackendConfig.from_cfg(_cfg(backend="local"))


@pytest.mark.parametrize(
    ("runtime_spec", "gatherer"),
    [
        pytest.param(None, _FakeGatherer(), id="missing-runtime-spec"),
        pytest.param(_runtime_spec(), None, id="missing-gatherer"),
        pytest.param(None, None, id="missing-both"),
    ],
)
def test_ray_backend_requires_runtime_spec_and_gatherer(
    runtime_spec: Any,
    gatherer: Any,
) -> None:
    with pytest.raises(ValueError, match="runtime_spec plus gatherer"):
        build_rollout_backend_from_cfg(
            _cfg(),
            runtime_spec=runtime_spec,
            gatherer=gatherer,
            driver_policy=_CpuPolicy(),
        )


def test_ray_backend_rejects_driver_cuda_policy_without_overlap() -> None:
    with pytest.raises(ValueError, match="Driver loaded rollout policy on CUDA"):
        validate_rollout_backend_config(
            _cfg(),
            driver_policy=_CudaPolicy(),
        )

    assert "model.device=cpu" in DRIVER_CUDA_OWNERSHIP_ERROR


def test_ray_backend_detects_cuda_trainable_module_when_policy_has_no_device() -> None:
    bundle = _Bundle(
        policy=object(),
        trainable_modules={"transformer": _FakeModule("cuda:1")},
    )

    with pytest.raises(ValueError, match="Driver loaded rollout policy on CUDA"):
        validate_rollout_backend_config(
            _cfg(),
            driver_bundle=bundle,
        )


def test_ray_backend_allows_driver_cuda_policy_with_explicit_overlap() -> None:
    config = validate_rollout_backend_config(
        _cfg(overlap=True),
        driver_policy=_CudaPolicy(),
    )

    assert config.backend == "ray"
    assert config.allow_driver_gpu_overlap is True
