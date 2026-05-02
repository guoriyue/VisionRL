"""Tests for SGLang-style generation runtime adapters."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from vrl.engine import ContinuousBatchPlanner, EngineLoop, Scheduler
from vrl.engine.generation import (
    FamilyPipelineRegistry,
    GenerationBatchPlanner,
    GenerationIdFactory,
    GenerationModelRunner,
    GenerationRequest,
    GenerationRuntime,
    GenerationWorker,
    OutputBatch,
    RolloutDitTrajectory,
    RolloutTrajectoryData,
    WorkloadSignature,
)
from vrl.engine.types import SchedulerOutput, SchedulerRequest
from vrl.executors import BatchedFamilyPipelineExecutor, FamilyPipelineExecutor
from vrl.executors.batching import (
    forward_batch_by_merging_prompts,
)


class _FakeExecutor(BatchedFamilyPipelineExecutor):
    family = "fake"
    task = "t2i"

    def __init__(self) -> None:
        self.forward_calls = 0
        self.forward_batch_calls = 0

    def workload_signature(
        self,
        request: GenerationRequest,
    ) -> WorkloadSignature:
        return WorkloadSignature.from_request(request)

    def forward(
        self,
        request: GenerationRequest,
        sample_specs: list[Any],
    ) -> OutputBatch:
        self.forward_calls += 1
        return OutputBatch(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=list(request.prompts),
            sample_specs=sample_specs,
            output={"num_samples": len(sample_specs)},
            rollout_trajectory_data=RolloutTrajectoryData(
                rollout_log_probs=[0.0 for _ in sample_specs],
                dit_trajectory=RolloutDitTrajectory(
                    latents=["latent" for _ in sample_specs],
                    timesteps=list(range(int(request.sampling.get("num_steps", 1)))),
                ),
            ),
        )

    def forward_batch(
        self,
        requests: list[GenerationRequest],
        sample_specs_by_request: dict[str, list[Any]],
    ) -> dict[str, OutputBatch]:
        self.forward_batch_calls += 1
        self.forward_calls += 1
        return {
            request.request_id: OutputBatch(
                request_id=request.request_id,
                family=request.family,
                task=request.task,
                prompts=list(request.prompts),
                sample_specs=sample_specs_by_request[request.request_id],
                output={
                    "num_samples": len(sample_specs_by_request[request.request_id])
                },
            )
            for request in requests
        }


class _ListOutputExecutor:
    family = "fake"
    task = "t2i"

    def forward(
        self,
        request: GenerationRequest,
        sample_specs: list[Any],
    ) -> OutputBatch:
        return OutputBatch(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=list(request.prompts),
            sample_specs=sample_specs,
            output=[spec.sample_id for spec in sample_specs],
            extra={
                "per_sample": [spec.prompt for spec in sample_specs],
                "context": {"shared": True},
            },
        )


def _request(
    request_id: str = "req-1",
    *,
    height: int = 512,
    width: int = 512,
    num_steps: int = 10,
    seed: int | None = 7,
) -> GenerationRequest:
    sampling = {
        "height": height,
        "width": width,
        "num_steps": num_steps,
    }
    if seed is not None:
        sampling["seed"] = seed
    return GenerationRequest(
        request_id=request_id,
        family="fake",
        task="t2i",
        prompts=["a test prompt"],
        samples_per_prompt=2,
        sampling=sampling,
        return_artifacts={"output", "rollout_trajectory_data"},
        metadata={"dataset": "unit"},
    )


def _worker() -> GenerationWorker:
    registry = FamilyPipelineRegistry()
    executor: FamilyPipelineExecutor = _FakeExecutor()
    registry.register(executor)
    return GenerationWorker(registry)


def test_generation_request_validation() -> None:
    with pytest.raises(ValueError, match="prompts"):
        GenerationRequest(
            request_id="req",
            family="fake",
            task="t2i",
            prompts=[],
            samples_per_prompt=1,
        )

    with pytest.raises(ValueError, match="samples_per_prompt"):
        GenerationRequest(
            request_id="req",
            family="fake",
            task="t2i",
            prompts=["x"],
            samples_per_prompt=0,
        )

    with pytest.raises(ValueError, match="policy_version"):
        GenerationRequest(
            request_id="req",
            family="fake",
            task="t2i",
            prompts=["x"],
            samples_per_prompt=1,
            policy_version=-1,
        )


def test_generation_id_factory_is_deterministic() -> None:
    request = _request()
    specs = GenerationIdFactory().build_sample_specs(request)

    assert [spec.sample_id for spec in specs] == [
        "req-1:prompt:0:sample:0",
        "req-1:prompt:0:sample:1",
    ]
    assert [spec.seed for spec in specs] == [7, 8]
    assert {spec.group_id for spec in specs} == {"req-1:prompt:0"}


def test_generation_model_runner_converts_scheduler_io() -> None:
    request = _request()
    scheduler_request = SchedulerRequest(
        request_id=request.request_id,
        data=request,
    )
    scheduler_output = SchedulerOutput(
        requests=[scheduler_request],
        batch_data=[request],
        step_id=1,
    )
    runner = GenerationModelRunner(_worker(), execute_in_thread=False)

    output = runner.execute(scheduler_output)

    assert output.req_ids == ["req-1"]
    request_output = output.outputs["req-1"]
    assert request_output.finished is True
    assert request_output.finish_reason == "completed"
    assert isinstance(request_output.data, OutputBatch)
    assert request_output.data.output == {"num_samples": 2}


def test_generation_worker_batches_identical_sampling_once() -> None:
    registry = FamilyPipelineRegistry()
    executor = _FakeExecutor()
    registry.register(executor)
    worker = GenerationWorker(registry)

    outputs = worker.execute([_request("a", seed=None), _request("b", seed=None)])

    assert sorted(outputs) == ["a", "b"]
    assert executor.forward_batch_calls == 1
    assert executor.forward_calls == 1
    assert outputs["a"].output == {"num_samples": 2}
    assert outputs["b"].output == {"num_samples": 2}


def test_generation_worker_does_not_batch_different_sampling() -> None:
    registry = FamilyPipelineRegistry()
    executor = _FakeExecutor()
    registry.register(executor)
    worker = GenerationWorker(registry)

    outputs = worker.execute([
        _request("a", seed=None),
        _request("b", height=768, seed=None),
    ])

    assert sorted(outputs) == ["a", "b"]
    assert executor.forward_batch_calls == 0
    assert executor.forward_calls == 2


def test_generation_worker_does_not_batch_different_policy_versions() -> None:
    registry = FamilyPipelineRegistry()
    executor = _FakeExecutor()
    registry.register(executor)
    worker = GenerationWorker(registry)

    req_a = _request("a", seed=None)
    req_b = _request("b", seed=None)
    req_a.policy_version = 1
    req_b.policy_version = 2
    outputs = worker.execute([req_a, req_b])

    assert sorted(outputs) == ["a", "b"]
    assert executor.forward_batch_calls == 0
    assert executor.forward_calls == 2


def test_generation_worker_does_not_batch_seeded_requests() -> None:
    registry = FamilyPipelineRegistry()
    executor = _FakeExecutor()
    registry.register(executor)
    worker = GenerationWorker(registry)

    outputs = worker.execute([_request("a", seed=7), _request("b", seed=7)])

    assert sorted(outputs) == ["a", "b"]
    assert executor.forward_batch_calls == 0
    assert executor.forward_calls == 2


def test_forward_batch_by_merging_prompts_splits_outputs_by_request() -> None:
    req_a = _request("a")
    req_b = _request("b")
    id_factory = GenerationIdFactory()
    specs_by_request = {
        "a": id_factory.build_sample_specs(req_a),
        "b": id_factory.build_sample_specs(req_b),
    }

    outputs = forward_batch_by_merging_prompts(
        _ListOutputExecutor(),
        [req_a, req_b],
        specs_by_request,
    )

    assert outputs["a"].output == [
        "a:prompt:0:sample:0",
        "a:prompt:0:sample:1",
    ]
    assert outputs["b"].output == [
        "b:prompt:0:sample:0",
        "b:prompt:0:sample:1",
    ]
    assert outputs["a"].extra["per_sample"] == ["a test prompt", "a test prompt"]
    assert outputs["a"].extra["context"] == {"shared": True}


@pytest.mark.asyncio
async def test_generation_runtime_uses_existing_engine_loop() -> None:
    worker = _worker()
    runner = GenerationModelRunner(worker, execute_in_thread=False)
    engine_loop = EngineLoop(
        scheduler=Scheduler(
            batch_planner=ContinuousBatchPlanner(max_batch_size=1),
        ),
        model_runner=runner,
    )
    runtime = GenerationRuntime(engine_loop)

    try:
        output = await asyncio.wait_for(runtime.generate(_request()), timeout=2.0)
    finally:
        await runtime.shutdown()

    assert output.request_id == "req-1"
    assert output.output == {"num_samples": 2}
    assert output.rollout_trajectory_data is not None
    assert output.rollout_trajectory_data.rollout_log_probs == [0.0, 0.0]


def test_generation_batch_planner_groups_same_signature() -> None:
    planner = GenerationBatchPlanner(max_batch_size=3)
    req_a = SchedulerRequest("a", _request("a"))
    req_b = SchedulerRequest("b", _request("b"))
    req_c = SchedulerRequest("c", _request("c", height=768))

    selected = planner.select_requests([req_a, req_b, req_c], [])
    batch = planner.build_batch(selected)

    assert [request.request_id for request in selected] == ["a", "b"]
    assert [request.request_id for request in batch] == ["a", "b"]
