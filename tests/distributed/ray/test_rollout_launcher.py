"""Tests for the Ray rollout launcher."""

from __future__ import annotations

import asyncio
import pickle
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vrl.distributed.ray import (
    DistributedRolloutConfig,
    RayRolloutLauncher,
    RayRolloutWorker,
    RolloutRuntimeSpec,
)
from vrl.engine.generation import (
    GenerationRequest,
    OutputBatch,
    WorkloadSignature,
    build_rollout_backend_from_cfg,
)
from vrl.executors import ChunkedFamilyPipelineExecutor, PipelineChunkResult
from vrl.executors.microbatching import MicroBatchPlan


@dataclass(slots=True)
class _LaunchChunk(PipelineChunkResult):
    executor_instance_id: str
    prompt_index: int
    sample_start: int
    sample_count: int


class _LauncherFakeExecutor(ChunkedFamilyPipelineExecutor):
    family = "fake"
    task = "t2i"

    def __init__(self) -> None:
        self.executor_instance_id = uuid.uuid4().hex

    def workload_signature(self, request: GenerationRequest) -> WorkloadSignature:
        return WorkloadSignature.from_request(request)

    def forward(
        self,
        request: GenerationRequest,
        sample_specs: list[Any],
    ) -> OutputBatch:
        chunks = [
            self.forward_chunk(
                request,
                MicroBatchPlan(
                    prompt_index=spec.prompt_index,
                    prompt=spec.prompt,
                    sample_start=spec.sample_index,
                    sample_count=1,
                ),
            )
            for spec in sample_specs
        ]
        return self.gather_chunks(request, sample_specs, chunks)

    def forward_chunk(
        self,
        request: GenerationRequest,
        chunk: MicroBatchPlan,
    ) -> _LaunchChunk:
        del request
        return _LaunchChunk(
            executor_instance_id=self.executor_instance_id,
            prompt_index=chunk.prompt_index,
            sample_start=chunk.sample_start,
            sample_count=chunk.sample_count,
        )

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_specs: Sequence[Any],
        chunks: Sequence[_LaunchChunk],
    ) -> OutputBatch:
        return _LauncherGatherer().gather_chunks(request, sample_specs, chunks)


class _LauncherGatherer:
    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_specs: Sequence[Any],
        chunks: Sequence[_LaunchChunk],
    ) -> OutputBatch:
        return OutputBatch(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=list(request.prompts),
            sample_specs=list(sample_specs),
            output=[
                {
                    "executor_instance_id": chunk.executor_instance_id,
                    "prompt_index": chunk.prompt_index,
                    "sample_start": chunk.sample_start,
                    "sample_count": chunk.sample_count,
                }
                for chunk in chunks
            ],
        )


def make_launcher_fake_executor(
    runtime_spec: RolloutRuntimeSpec,
) -> _LauncherFakeExecutor:
    assert runtime_spec.family == "fake"
    assert runtime_spec.executor_kwargs == {"sample_batch_size": 2}
    return _LauncherFakeExecutor()


def make_launcher_runtime_bundle(build_spec: Any) -> Any:
    assert str(build_spec.device) == "cuda"
    assert build_spec.dtype is torch.bfloat16
    return SimpleNamespace(policy={"device": build_spec.device, "dtype": build_spec.dtype})


class LauncherExecutorFromPolicy(_LauncherFakeExecutor):
    def __init__(self, policy: Any, **kwargs: Any) -> None:
        super().__init__()
        self.policy = policy
        self.kwargs = kwargs


@pytest.fixture()
def ray_local():
    ray = pytest.importorskip("ray")
    ray.init(
        local_mode=True,
        num_cpus=4,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
    )
    try:
        yield ray
    finally:
        ray.shutdown()


def _runtime_spec(*, policy_version: int | None = 3) -> RolloutRuntimeSpec:
    return RolloutRuntimeSpec(
        family="fake",
        task="t2i",
        model_config={"model_name_or_path": "fake-model", "dtype": "float32"},
        executor_kwargs={"sample_batch_size": 2},
        policy_version=policy_version,
        executor_factory=(
            "tests.distributed.ray.test_rollout_launcher:"
            "make_launcher_fake_executor"
        ),
    )


def _request(*, policy_version: int | None = 3) -> GenerationRequest:
    return GenerationRequest(
        request_id="req",
        family="fake",
        task="t2i",
        prompts=["p0", "p1"],
        samples_per_prompt=4,
        sampling={"sample_batch_size": 2},
        policy_version=policy_version,
    )


def _config(*, num_workers: int = 2) -> DistributedRolloutConfig:
    return DistributedRolloutConfig(
        backend="ray",
        num_workers=num_workers,
        gpus_per_worker=0.0,
        cpus_per_worker=1.0,
        max_inflight_chunks_per_worker=1,
    )


def test_ray_rollout_launcher_single_worker_smoke(ray_local) -> None:
    launcher = RayRolloutLauncher(init_ray=False)
    runtime = launcher.launch(_config(num_workers=1), _runtime_spec(), _LauncherGatherer())
    try:
        output = asyncio.run(runtime.generate(_request()))
    finally:
        asyncio.run(runtime.shutdown())

    assert [
        (row["prompt_index"], row["sample_start"], row["sample_count"])
        for row in output.output
    ] == [(0, 0, 2), (0, 2, 2), (1, 0, 2), (1, 2, 2)]
    assert len({row["executor_instance_id"] for row in output.output}) == 1


def test_ray_rollout_launcher_multi_worker_assigns_chunks_to_distinct_replicas(
    ray_local,
) -> None:
    runtime = RayRolloutLauncher(init_ray=False).launch(
        _config(num_workers=2),
        _runtime_spec(policy_version=5),
        _LauncherGatherer(),
    )
    try:
        workers = runtime.executor.workers
        assert [worker.worker_id for worker in workers] == ["rollout-0", "rollout-1"]
        assert all(worker.actor is not None for worker in workers)
        assert all(worker.node_id for worker in workers)
        assert all(isinstance(worker.gpu_ids, tuple) for worker in workers)

        assignments = runtime.executor.planner.plan(_request(policy_version=5), workers)
        assert [assignment.worker_id for assignment in assignments] == [
            "rollout-0",
            "rollout-1",
            "rollout-0",
            "rollout-1",
        ]

        output = asyncio.run(runtime.generate(_request(policy_version=5)))
    finally:
        asyncio.run(runtime.shutdown())

    assert output.error is None
    by_executor: dict[str, list[tuple[int, int]]] = {}
    for row in output.output:
        by_executor.setdefault(row["executor_instance_id"], []).append(
            (row["prompt_index"], row["sample_start"]),
        )
    assert len(by_executor) == 2
    assert {tuple(chunks) for chunks in by_executor.values()} == {
        ((0, 0), (1, 0)),
        ((0, 2), (1, 2)),
    }


def test_runtime_factory_launches_ray_runtime_when_spec_and_gatherer_are_provided(
    ray_local,
) -> None:
    runtime = build_rollout_backend_from_cfg(
        {
            "distributed": {
                "backend": "ray",
                "rollout": {
                    "num_workers": 1,
                    "gpus_per_worker": 0.0,
                    "cpus_per_worker": 1.0,
                    "executor_factory": "legacy.factory:path",
                },
            },
        },
        runtime_spec=_runtime_spec(policy_version=None),
        gatherer=_LauncherGatherer(),
    )
    try:
        output = asyncio.run(runtime.generate(_request(policy_version=None)))
    finally:
        asyncio.run(runtime.shutdown())

    assert [
        (row["prompt_index"], row["sample_start"], row["sample_count"])
        for row in output.output
    ] == [(0, 0, 2), (0, 2, 2), (1, 0, 2), (1, 2, 2)]


def test_ray_rollout_launcher_runtime_spec_rejects_live_objects() -> None:
    spec = _runtime_spec(policy_version=7)
    assert pickle.loads(pickle.dumps(spec)) == spec

    for key in ("executor", "policy", "pipeline"):
        with pytest.raises(ValueError, match=key):
            RayRolloutLauncher().launch(
                _config(num_workers=1),
                {"family": "fake", "task": "t2i", key: object()},
                _LauncherGatherer(),
            )

    with pytest.raises(TypeError, match=r"torch\.Tensor"):
        RayRolloutLauncher().launch(
            _config(num_workers=1),
            {
                "family": "fake",
                "task": "t2i",
                "model_config": {"weights": torch.zeros(1)},
                "executor_factory": (
                    "tests.distributed.ray.test_rollout_launcher:"
                    "make_launcher_fake_executor"
                ),
            },
            _LauncherGatherer(),
        )


def test_ray_rollout_worker_runtime_builder_normalizes_device_and_dtype() -> None:
    worker = RayRolloutWorker(
        worker_id="w0",
        family="fake",
        runtime_spec=RolloutRuntimeSpec(
            family="fake",
            task="t2i",
            build_spec={
                "model_name_or_path": "fake-model",
                "device": "cuda",
                "dtype": "bfloat16",
            },
            runtime_builder=(
                "tests.distributed.ray.test_rollout_launcher:"
                "make_launcher_runtime_bundle"
            ),
            executor_cls=(
                "tests.distributed.ray.test_rollout_launcher:"
                "LauncherExecutorFromPolicy"
            ),
        ),
    )

    worker.load_policy()

    assert isinstance(worker.executor, LauncherExecutorFromPolicy)
    assert str(worker.executor.policy["device"]) == "cuda"
    assert worker.executor.policy["dtype"] is torch.bfloat16


def test_ray_rollout_launcher_requires_ray(monkeypatch) -> None:
    import vrl.distributed.ray.launcher as launcher_mod

    def _missing_ray() -> Any:
        raise ImportError("Ray distributed rollout support requires `ray`.")

    monkeypatch.setattr(launcher_mod, "require_ray", _missing_ray)

    with pytest.raises(ImportError, match="Ray distributed rollout support"):
        RayRolloutLauncher().launch(
            _config(num_workers=1),
            _runtime_spec(),
            _LauncherGatherer(),
        )
