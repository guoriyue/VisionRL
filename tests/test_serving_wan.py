"""Integration test: serve Wan model through the full engine stack.

StubWanModel → ComposedPipeline → CallableModelRunner → EngineLoop
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
import pytest_asyncio

from tests.conftest import StubWanModel, model_to_pipeline
from wm_infra.engine.interfaces import (
    FIFOBatchPlanner,
    SimpleResourceManager,
    SinglePassIterationController,
)
from wm_infra.engine.managers.engine_loop import EngineLoop
from wm_infra.engine.managers.scheduler import Scheduler
from wm_infra.engine.model_executor.model_runner import CallableModelRunner
from wm_infra.schemas.video_generation import VideoGenerationRequest


def _make_wan_engine() -> EngineLoop:
    """Build a full engine stack with Wan stub model."""
    model = StubWanModel()
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
async def wan_engine():
    engine = _make_wan_engine()
    await engine.start()
    yield engine
    await engine.stop()


@pytest.mark.asyncio
async def test_single_request(wan_engine: EngineLoop):
    """Submit one Wan request and verify output pipeline results."""
    engine = wan_engine
    request = VideoGenerationRequest(
        prompt="A serene mountain landscape with flowing rivers",
        width=512,
        height=320,
        frame_count=8,
        num_steps=20,
        seed=7,
    )
    await engine.add_request("wan-single", request)
    result = await engine.get_result("wan-single")

    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 5  # 5 stages

    # Verify each stage succeeded
    stage_names = ["encode_text", "encode_conditioning", "denoise", "decode_vae", "postprocess"]
    for i, stage_result in enumerate(result):
        assert stage_result.status == "succeeded", f"Stage {stage_names[i]} failed"


@pytest.mark.asyncio
async def test_output_shape(wan_engine: EngineLoop):
    """Verify decoded video frames have correct spatial dimensions."""
    engine = wan_engine
    W, H, F = 256, 160, 4
    request = VideoGenerationRequest(
        prompt="shape test", width=W, height=H, frame_count=F, seed=0
    )
    await engine.add_request("wan-shape", request)
    result = await engine.get_result("wan-shape")

    # Postprocess stage should have _pipeline_output
    postprocess = result[-1]
    frames = postprocess.state_updates.get("_pipeline_output")
    assert frames is not None
    arr = np.asarray(frames)
    assert arr.dtype == np.uint8
    assert arr.shape[0] == F  # frame count
    assert arr.shape[1] == H  # height
    assert arr.shape[2] == W  # width
    assert arr.shape[3] == 3  # RGB channels


@pytest.mark.asyncio
async def test_multiple_concurrent_requests(wan_engine: EngineLoop):
    """Submit multiple Wan requests concurrently."""
    engine = wan_engine
    n = 4
    for i in range(n):
        req = VideoGenerationRequest(
            prompt=f"Concurrent scene {i}",
            width=128,
            height=80,
            frame_count=2,
            seed=i,
        )
        await engine.add_request(f"wan-concurrent-{i}", req)

    tasks = [
        asyncio.create_task(engine.get_result(f"wan-concurrent-{i}"))
        for i in range(n)
    ]
    results = await asyncio.gather(*tasks)

    assert len(results) == n
    for result in results:
        assert isinstance(result, list)
        assert len(result) == 5


@pytest.mark.asyncio
async def test_abort_request(wan_engine: EngineLoop):
    """Abort a Wan request."""
    engine = wan_engine
    req = VideoGenerationRequest(prompt="abort me", width=64, height=64, frame_count=1)
    await engine.add_request("wan-abort", req)
    success = engine.abort_request("wan-abort")
    assert success is True

    # Second abort should return False (already aborted)
    success2 = engine.abort_request("wan-abort")
    assert success2 is False


@pytest.mark.asyncio
async def test_deterministic_output(wan_engine: EngineLoop):
    """Same seed produces identical Wan output."""
    engine = wan_engine
    for tag in ("a", "b"):
        req = VideoGenerationRequest(
            prompt="determinism", width=128, height=80, frame_count=2, seed=99
        )
        await engine.add_request(f"wan-det-{tag}", req)

    ra = await engine.get_result("wan-det-a")
    rb = await engine.get_result("wan-det-b")

    frames_a = np.asarray(ra[-1].state_updates["_pipeline_output"])
    frames_b = np.asarray(rb[-1].state_updates["_pipeline_output"])
    np.testing.assert_array_equal(frames_a, frames_b)


@pytest.mark.asyncio
async def test_different_seeds_produce_different_output(wan_engine: EngineLoop):
    """Different seeds should produce different frames."""
    engine = wan_engine
    for seed in (1, 2):
        req = VideoGenerationRequest(
            prompt="diff seeds", width=128, height=80, frame_count=2, seed=seed
        )
        await engine.add_request(f"wan-diff-{seed}", req)

    r1 = await engine.get_result("wan-diff-1")
    r2 = await engine.get_result("wan-diff-2")

    frames1 = np.asarray(r1[-1].state_updates["_pipeline_output"])
    frames2 = np.asarray(r2[-1].state_updates["_pipeline_output"])
    assert not np.array_equal(frames1, frames2)


@pytest.mark.asyncio
async def test_add_request_nowait(wan_engine: EngineLoop):
    """Test the IPC-compat add_request_nowait returns a Task."""
    engine = wan_engine
    req = VideoGenerationRequest(prompt="nowait test", width=64, height=64, frame_count=1, seed=0)
    task = engine.add_request_nowait("wan-nowait", req)
    result = await task

    # add_request_nowait wraps as RequestOutput
    from wm_infra.engine.types import RequestOutput
    assert isinstance(result, RequestOutput)
    assert result.request_id == "wan-nowait"
    assert result.finished is True
