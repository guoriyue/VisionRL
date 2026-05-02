"""Tests for distributed large rollout execution primitives."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

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
)
from vrl.engine.generation import GenerationRequest, OutputBatch, WorkloadSignature
from vrl.executors.planning import MicroBatchPlan


@dataclass(slots=True)
class _TensorChunk:
    value: torch.Tensor
    prompt_index: int
    sample_start: int
    sample_count: int


class _FakeChunkedExecutor:
    family = "fake"
    task = "t2i"

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
        return OutputBatch(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=list(request.prompts),
            sample_specs=sample_specs,
            output=[
                (chunk.prompt_index, chunk.sample_start, chunk.sample_count)
                for chunk in chunks
            ],
        )


class _FakeActor:
    def __init__(self) -> None:
        self.version: int | None = None

    def update_weights(self, state_ref: Any, policy_version: int) -> None:
        del state_ref
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


def test_distributed_rollout_config_validation_and_legacy_mapping() -> None:
    from vrl.distributed.ray import RayConfig

    config = DistributedRolloutConfig.from_legacy(
        RayConfig(
            enable=True,
            num_rollout_workers=2,
            gpus_per_rollout_worker=0.0,
            cpus_per_rollout_worker=1.0,
        ),
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


def test_ray_rollout_worker_returns_cpu_only_chunk_result() -> None:
    worker = RayRolloutWorker(
        worker_id="w0",
        family="fake",
        runtime_spec={
            "executor": _FakeChunkedExecutor(),
            "policy_version": 3,
        },
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
        runtime_spec={
            "executor": _FakeChunkedExecutor(),
            "policy_version": 1,
        },
    )

    result = worker.execute_chunk(
        _request(policy_version=2),
        MicroBatchPlan(prompt_index=0, prompt="p0", sample_start=0, sample_count=2),
    )

    assert result.output is None
    assert "policy_version mismatch" in (result.error or "")


def test_distributed_rollout_executor_gathers_direct_actor_results() -> None:
    actors = [
        RayRolloutWorker("w0", "fake", {"executor": _FakeChunkedExecutor(), "policy_version": 3}),
        RayRolloutWorker("w1", "fake", {"executor": _FakeChunkedExecutor(), "policy_version": 3}),
    ]
    workers = [
        RayWorkerHandle(worker_id="w0", node_id="n0", actor=actors[0]),
        RayWorkerHandle(worker_id="w1", node_id="n1", actor=actors[1]),
    ]
    executor = DistributedRolloutExecutor(
        DistributedExecutionPlanner(),
        workers,
        _FakeChunkedExecutor(),
    )

    output = asyncio.run(executor.execute(_request(policy_version=3)))

    assert output.output == [(0, 0, 2), (0, 2, 2), (1, 0, 2), (1, 2, 2)]


def test_ray_distributed_runtime_delegates_to_executor() -> None:
    actors = [
        RayRolloutWorker("w0", "fake", {"executor": _FakeChunkedExecutor()}),
    ]
    workers = [RayWorkerHandle(worker_id="w0", node_id="n0", actor=actors[0])]
    executor = DistributedRolloutExecutor(
        DistributedExecutionPlanner(),
        workers,
        _FakeChunkedExecutor(),
    )
    runtime = RayDistributedRuntime(executor)

    output = asyncio.run(runtime.generate(_request(policy_version=None)))

    assert output.request_id == "req"
    assert output.output == [(0, 0, 2), (0, 2, 2), (1, 0, 2), (1, 2, 2)]


def test_ray_distributed_runtime_fills_current_policy_version() -> None:
    actor = RayRolloutWorker(
        "w0",
        "fake",
        {"executor": _FakeChunkedExecutor(), "policy_version": 5},
    )
    workers = [RayWorkerHandle(worker_id="w0", node_id="n0", actor=actor)]
    executor = DistributedRolloutExecutor(
        DistributedExecutionPlanner(),
        workers,
        _FakeChunkedExecutor(),
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

    asyncio.run(RayRolloutWeightSync(workers).push_to_rollout_workers({}, 7))

    assert [actor.version for actor in actors] == [7, 7]


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
