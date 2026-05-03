"""Tests for distributed large rollout execution primitives."""

from __future__ import annotations

import asyncio
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vrl.distributed.ray import (
    DistributedExecutionPlanner,
    DistributedRolloutConfig,
    DistributedRolloutExecutor,
    RayDistributedRuntime,
    RayRolloutWeightSync,
    RayRolloutWorker,
    RayTrainActor,
    RayTrainGroup,
    RayWorkerHandle,
    RolloutRuntimeSpec,
)
from vrl.engine.generation import (
    ChunkedFamilyPipelineExecutor,
    GenerationRequest,
    OutputBatch,
    PipelineChunkResult,
    WorkloadSignature,
)
from vrl.engine.generation.microbatching import MicroBatchPlan


@dataclass(slots=True)
class _TensorChunk(PipelineChunkResult):
    value: torch.Tensor
    prompt_index: int
    sample_start: int
    sample_count: int


class _FakeChunkedExecutor(ChunkedFamilyPipelineExecutor):
    family = "fake"
    task = "t2i"

    def __init__(self, policy: Any | None = None, **kwargs: Any) -> None:
        self.policy = policy
        self.kwargs = dict(kwargs)

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
    ) -> _TensorChunk:
        del request
        return _TensorChunk(
            value=torch.full((chunk.sample_count,), float(chunk.prompt_index)),
            prompt_index=chunk.prompt_index,
            sample_start=chunk.sample_start,
            sample_count=chunk.sample_count,
        )

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_specs: list[Any],
        chunks: list[_TensorChunk],
    ) -> OutputBatch:
        return _FakeChunkGatherer().gather_chunks(request, sample_specs, chunks)


class _FakePolicyWithTrainableState:
    def __init__(self) -> None:
        self.loaded_states: list[dict[str, Any]] = []

    def load_trainable_state(self, state_dict: dict[str, Any]) -> None:
        self.loaded_states.append(dict(state_dict))


class _FakeExecutorWithPolicy(_FakeChunkedExecutor):
    def __init__(self, policy: Any | None = None, **kwargs: Any) -> None:
        super().__init__(policy, **kwargs)
        self.model = policy or _FakePolicyWithTrainableState()


class _FakeChunkGatherer:
    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_specs: Sequence[Any],
        chunks: Sequence[_TensorChunk],
    ) -> OutputBatch:
        return OutputBatch(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=list(request.prompts),
            sample_specs=list(sample_specs),
            output=[
                (chunk.prompt_index, chunk.sample_start, chunk.sample_count) for chunk in chunks
            ],
        )


_FAKE_RUNTIME_BUILDER = (
    "tests.distributed.ray.test_large_rollout_execution:make_fake_runtime_bundle"
)
_FAKE_EXECUTOR_CLS = "tests.distributed.ray.test_large_rollout_execution:_FakeChunkedExecutor"
_FAKE_POLICY_EXECUTOR_CLS = (
    "tests.distributed.ray.test_large_rollout_execution:_FakeExecutorWithPolicy"
)


def make_fake_runtime_bundle(build_spec: Any) -> Any:
    assert build_spec.model_name_or_path == "fake-model"
    assert str(build_spec.device) == "cpu"
    assert build_spec.dtype is torch.float32
    return SimpleNamespace(policy=_FakePolicyWithTrainableState())


def _runtime_spec(*, policy_version: int | None = 3) -> RolloutRuntimeSpec:
    return RolloutRuntimeSpec(
        family="fake",
        task="t2i",
        build_spec={
            "model_name_or_path": "fake-model",
            "device": "cpu",
            "dtype": "float32",
        },
        executor_kwargs={"sample_batch_size": 2},
        policy_version=policy_version,
        runtime_builder=_FAKE_RUNTIME_BUILDER,
        executor_cls=_FAKE_EXECUTOR_CLS,
    )


def _runtime_spec_with_policy(*, policy_version: int | None = 3) -> RolloutRuntimeSpec:
    return RolloutRuntimeSpec(
        family="fake",
        task="t2i",
        build_spec={
            "model_name_or_path": "fake-model",
            "device": "cpu",
            "dtype": "float32",
        },
        policy_version=policy_version,
        runtime_builder=_FAKE_RUNTIME_BUILDER,
        executor_cls=_FAKE_POLICY_EXECUTOR_CLS,
    )


def _runtime_spec_dict(*, policy_version: int | None = 3) -> dict[str, Any]:
    return _runtime_spec(policy_version=policy_version).to_dict()


class _FakeActor:
    def __init__(self) -> None:
        self.version: int | None = None
        self.state_ref: Any = None

    def update_weights(self, state_ref: Any, policy_version: int) -> None:
        self.state_ref = state_ref
        self.version = policy_version


class _FakeTrainer:
    def __init__(self) -> None:
        self.seen: list[list[Any]] = []

    async def step(self, prompts: list[Any]) -> dict[str, Any]:
        self.seen.append(list(prompts))
        return {"num_prompts": len(prompts)}


def make_fake_trainer(train_config: Any) -> _FakeTrainer:
    assert train_config["marker"] == "inside-actor"
    return _FakeTrainer()


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


def test_distributed_rollout_config_validation() -> None:
    config = DistributedRolloutConfig(
        backend="ray",
        num_workers=2,
        gpus_per_worker=0.0,
        cpus_per_worker=1.0,
    )

    assert config.backend == "ray"
    assert config.num_workers == 2
    assert config.gpus_per_worker == 0.0


def test_distributed_execution_planner_round_robins_chunks() -> None:
    workers = [
        RayWorkerHandle(worker_id="w0", node_id="n0", gpu_ids=(0,)),
        RayWorkerHandle(worker_id="w1", node_id="n1", gpu_ids=(1,)),
    ]

    assignments = DistributedExecutionPlanner().plan(_request(), workers)

    assert [assignment.worker_id for assignment in assignments] == [
        "w0",
        "w1",
        "w0",
        "w1",
    ]
    assert [
        (assignment.chunk.prompt_index, assignment.chunk.sample_start)
        for assignment in assignments
    ] == [(0, 0), (0, 2), (1, 0), (1, 2)]


def test_rollout_runtime_spec_is_pickle_serializable() -> None:
    spec = _runtime_spec(policy_version=3)

    restored = pickle.loads(pickle.dumps(spec))

    assert restored == spec
    assert restored.to_dict() == spec.to_dict()


@pytest.mark.parametrize("key", ["executor", "policy", "pipeline"])
def test_rollout_runtime_spec_rejects_live_object_keys(key: str) -> None:
    with pytest.raises(ValueError, match=key):
        RayRolloutWorker(
            worker_id="w0",
            family="fake",
            runtime_spec={
                key: _FakeChunkedExecutor(),
                "policy_version": 3,
                "runtime_builder": _FAKE_RUNTIME_BUILDER,
                "executor_cls": _FAKE_EXECUTOR_CLS,
            },
        )


def test_rollout_runtime_spec_rejects_live_tensor_payload() -> None:
    with pytest.raises(TypeError, match=r"torch\.Tensor"):
        RolloutRuntimeSpec(
            family="fake",
            task="t2i",
            model_config={"weights": torch.zeros(1)},
            runtime_builder=_FAKE_RUNTIME_BUILDER,
            executor_cls=_FAKE_EXECUTOR_CLS,
        )


def test_rollout_runtime_spec_rejects_executor_factory_key() -> None:
    with pytest.raises(ValueError, match=r"unsupported.*executor_factory"):
        RolloutRuntimeSpec.from_dict(
            {
                "family": "fake",
                "task": "t2i",
                "executor_factory": "tests.fake:factory",
                "runtime_builder": "tests.fake:build_runtime",
                "executor_cls": "tests.fake:Executor",
            },
        )


def test_rollout_runtime_spec_rejects_missing_builder_mode() -> None:
    with pytest.raises(ValueError, match="runtime_builder and executor_cls"):
        RolloutRuntimeSpec(family="fake", task="t2i")


@pytest.mark.parametrize(
    ("runtime_builder", "executor_cls", "match"),
    [
        (None, "tests.fake:Executor", "runtime_builder and executor_cls"),
        ("tests.fake:build_runtime", None, "runtime_builder and executor_cls"),
    ],
)
def test_rollout_runtime_spec_requires_complete_builder_pair(
    runtime_builder: str | None,
    executor_cls: str | None,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        RolloutRuntimeSpec(
            family="fake",
            task="t2i",
            runtime_builder=runtime_builder,
            executor_cls=executor_cls,
        )


def test_ray_rollout_worker_returns_cpu_only_chunk_result() -> None:
    worker = RayRolloutWorker(
        worker_id="w0",
        family="fake",
        runtime_spec=_runtime_spec(policy_version=3),
    )
    result = worker.execute_chunk(
        _request(policy_version=3),
        MicroBatchPlan(prompt_index=0, prompt="p0", sample_start=0, sample_count=2),
    )

    assert result.error is None
    assert result.policy_version == 3
    assert isinstance(result.output, _TensorChunk)
    assert result.output.value.device.type == "cpu"


def test_ray_rollout_worker_rejects_stale_policy_version() -> None:
    worker = RayRolloutWorker(
        worker_id="w0",
        family="fake",
        runtime_spec=_runtime_spec_dict(policy_version=1),
    )

    result = worker.execute_chunk(
        _request(policy_version=2),
        MicroBatchPlan(prompt_index=0, prompt="p0", sample_start=0, sample_count=2),
    )

    assert result.output is None
    assert "policy_version mismatch" in (result.error or "")


@pytest.mark.parametrize(
    "state",
    [
        {"weight": torch.ones(1)},
        {"transformer.weight": torch.ones(1)},
    ],
)
def test_ray_rollout_worker_updates_policy_trainable_state(state: dict[str, Any]) -> None:
    worker = RayRolloutWorker(
        worker_id="w0",
        family="fake",
        runtime_spec=_runtime_spec_with_policy(policy_version=1),
    )

    worker.update_weights(state, policy_version=9)

    assert worker.current_policy_version() == 9
    assert isinstance(worker.executor, _FakeExecutorWithPolicy)
    assert worker.executor.model.loaded_states == [state]


def test_ray_rollout_worker_requires_model_for_weight_sync() -> None:
    worker = RayRolloutWorker(
        worker_id="w0",
        family="fake",
        runtime_spec=_runtime_spec(policy_version=1),
    )

    with pytest.raises(RuntimeError, match="must expose model for weight sync"):
        worker.update_weights({"transformer.weight": torch.ones(1)}, policy_version=9)


def test_distributed_rollout_executor_gathers_direct_actor_results() -> None:
    actors = [
        RayRolloutWorker("w0", "fake", _runtime_spec(policy_version=3)),
        RayRolloutWorker("w1", "fake", _runtime_spec(policy_version=3)),
    ]
    workers = [
        RayWorkerHandle(worker_id="w0", node_id="n0", actor=actors[0]),
        RayWorkerHandle(worker_id="w1", node_id="n1", actor=actors[1]),
    ]
    executor = DistributedRolloutExecutor(
        DistributedExecutionPlanner(),
        workers,
        _FakeChunkGatherer(),
    )
    assert not isinstance(executor.gatherer, ChunkedFamilyPipelineExecutor)

    output = asyncio.run(executor.execute(_request(policy_version=3)))

    assert output.output == [(0, 0, 2), (0, 2, 2), (1, 0, 2), (1, 2, 2)]


def test_distributed_rollout_executor_rejects_invalid_inflight_limit() -> None:
    actor = RayRolloutWorker("w0", "fake", _runtime_spec(policy_version=3))
    workers = [RayWorkerHandle(worker_id="w0", node_id="n0", actor=actor)]

    with pytest.raises(ValueError, match="max_inflight_chunks_per_worker"):
        DistributedRolloutExecutor(
            DistributedExecutionPlanner(),
            workers,
            _FakeChunkGatherer(),
            max_inflight_chunks_per_worker=0,
        )


def test_ray_distributed_runtime_delegates_to_executor() -> None:
    actors = [
        RayRolloutWorker("w0", "fake", _runtime_spec(policy_version=None)),
    ]
    workers = [RayWorkerHandle(worker_id="w0", node_id="n0", actor=actors[0])]
    executor = DistributedRolloutExecutor(
        DistributedExecutionPlanner(),
        workers,
        _FakeChunkGatherer(),
    )
    runtime = RayDistributedRuntime(executor)

    output = asyncio.run(runtime.generate(_request(policy_version=None)))

    assert output.request_id == "req"
    assert output.output == [(0, 0, 2), (0, 2, 2), (1, 0, 2), (1, 2, 2)]


def test_ray_distributed_runtime_fills_current_policy_version() -> None:
    actor = RayRolloutWorker(
        "w0",
        "fake",
        _runtime_spec(policy_version=5),
    )
    workers = [RayWorkerHandle(worker_id="w0", node_id="n0", actor=actor)]
    executor = DistributedRolloutExecutor(
        DistributedExecutionPlanner(),
        workers,
        _FakeChunkGatherer(),
    )
    runtime = RayDistributedRuntime(executor)
    runtime.current_policy_version = 5

    output = asyncio.run(runtime.generate(_request(policy_version=None)))

    assert output.error is None
    assert output.output == [(0, 0, 2), (0, 2, 2), (1, 0, 2), (1, 2, 2)]


def test_ray_rollout_weight_sync_updates_direct_workers() -> None:
    actors = [_FakeActor(), _FakeActor()]
    workers = [
        RayWorkerHandle(worker_id="w0", node_id="n0", actor=actors[0]),
        RayWorkerHandle(worker_id="w1", node_id="n1", actor=actors[1]),
    ]

    state = {"transformer.weight": torch.ones(1)}

    asyncio.run(RayRolloutWeightSync(workers).push_to_rollout_workers(state, 7))

    assert [actor.version for actor in actors] == [7, 7]
    assert [actor.state_ref for actor in actors] == [state, state]


def test_ray_train_actor_builds_trainer_inside_actor() -> None:
    actor = RayTrainActor(
        {
            "marker": "inside-actor",
            "trainer_factory": make_fake_trainer,
        },
    )

    result = actor.train_step(["p0", "p1"])

    assert result == {"num_prompts": 2}
    assert actor.metadata()["trainer_loaded"] is True


def test_ray_train_group_delegates_to_primary_actor() -> None:
    actor = RayTrainActor(
        {
            "marker": "inside-actor",
            "trainer_factory": make_fake_trainer,
        },
    )
    group = RayTrainGroup([actor])

    result = group.train_step(["p0"])

    assert result == {"num_prompts": 1}
