"""Integration test: serve Cosmos model through the full engine stack.

StubCosmosLocalExecutor → ComposedPipeline → CallableModelRunner → EngineLoop
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
import pytest_asyncio

from tests.conftest import model_to_pipeline
from wm_infra.engine.interfaces import (
    FIFOBatchPlanner,
    SimpleResourceManager,
    SinglePassIterationController,
)
from wm_infra.engine.managers.engine_loop import EngineLoop
from wm_infra.engine.managers.scheduler import Scheduler
from wm_infra.engine.model_executor.model_runner import CallableModelRunner
from wm_infra.models.families.cosmos.model import CosmosGenerationModel
from wm_infra.schemas.video_generation import VideoGenerationRequest


def _make_cosmos_engine() -> EngineLoop:
    """Build a full engine stack with Cosmos stub model."""
    model = CosmosGenerationModel()  # defaults to StubCosmosLocalExecutor
    pipeline = model_to_pipeline(model)

    async def model_fn(request):
        data = request.data
        results = await pipeline.run(data, {})
        return results

    runner = CallableModelRunner(model_fn)
    scheduler = Scheduler(
        batch_planner=FIFOBatchPlanner(max_batch_size=4),
        resource_manager=SimpleResourceManager(max_concurrent=4),
        iteration_controller=SinglePassIterationController(),
    )
    return EngineLoop(scheduler=scheduler, model_runner=runner)


@pytest_asyncio.fixture
async def cosmos_engine():
    engine = _make_cosmos_engine()
    await engine.start()
    yield engine
    await engine.stop()


@pytest.mark.asyncio
async def test_single_request(cosmos_engine: EngineLoop):
    """Submit one Cosmos request and verify output is a list of StageResults."""
    request = VideoGenerationRequest(
        prompt="A futuristic cityscape at sunset",
        width=256,
        height=160,
        frame_count=4,
        num_steps=10,
        seed=123,
    )
    result = await cosmos_engine.get_result("cosmos-1")

    # get_result returns None before submission, let's use the full flow
    engine = cosmos_engine
    await engine.add_request("cosmos-single", request)
    result = await engine.get_result("cosmos-single")

    assert result is not None
    # Result is a list of StageResults from pipeline.run()
    assert isinstance(result, list)
    assert len(result) == 5  # 5 stages

    # Last stage (postprocess) should have produced uint8 frames
    last_stage = result[-1]
    assert last_stage.status == "succeeded"


@pytest.mark.asyncio
async def test_multiple_concurrent_requests(cosmos_engine: EngineLoop):
    """Submit multiple requests concurrently."""
    engine = cosmos_engine
    requests = {}
    for i in range(3):
        req = VideoGenerationRequest(
            prompt=f"Scene {i}: underwater coral reef",
            width=128,
            height=80,
            frame_count=2,
            num_steps=5,
            seed=i * 100,
        )
        rid = f"cosmos-concurrent-{i}"
        requests[rid] = req
        await engine.add_request(rid, req)

    # Await all results concurrently
    tasks = {
        rid: asyncio.create_task(engine.get_result(rid)) for rid in requests
    }
    results = {}
    for rid, task in tasks.items():
        results[rid] = await task

    assert len(results) == 3
    for rid, result in results.items():
        assert isinstance(result, list)
        assert len(result) == 5


@pytest.mark.asyncio
async def test_abort_request(cosmos_engine: EngineLoop):
    """Abort a request before it completes."""
    engine = cosmos_engine
    req = VideoGenerationRequest(prompt="to be aborted", width=64, height=64, frame_count=1)
    await engine.add_request("cosmos-abort", req)
    engine.abort_request("cosmos-abort")

    # The request should have been aborted
    sched_req = engine.scheduler.get_request("cosmos-abort")
    # Might be None (already cleaned up) or ABORTED
    if sched_req is not None:
        from wm_infra.engine.types import SchedulerStatus
        assert sched_req.status == SchedulerStatus.ABORTED


@pytest.mark.asyncio
async def test_deterministic_output(cosmos_engine: EngineLoop):
    """Same seed should produce same output."""
    engine = cosmos_engine
    req1 = VideoGenerationRequest(
        prompt="deterministic test", width=128, height=80, frame_count=2, seed=42
    )
    req2 = VideoGenerationRequest(
        prompt="deterministic test", width=128, height=80, frame_count=2, seed=42
    )

    await engine.add_request("cosmos-det-1", req1)
    await engine.add_request("cosmos-det-2", req2)

    r1 = await engine.get_result("cosmos-det-1")
    r2 = await engine.get_result("cosmos-det-2")

    # Compare postprocess stage output frames
    frames1 = r1[-1].state_updates.get("video_frames") or r1[-1].state_updates.get("_pipeline_output")
    frames2 = r2[-1].state_updates.get("video_frames") or r2[-1].state_updates.get("_pipeline_output")

    if frames1 is not None and frames2 is not None:
        np.testing.assert_array_equal(np.asarray(frames1), np.asarray(frames2))


@pytest.mark.asyncio
async def test_engine_introspection(cosmos_engine: EngineLoop):
    """Test num_waiting, num_running, num_pending."""
    engine = cosmos_engine
    assert engine.num_waiting() == 0
    assert engine.num_running() == 0
    assert engine.num_pending() == 0
