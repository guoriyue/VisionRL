"""Tests for EngineLoop request scheduling."""

from __future__ import annotations

import asyncio

import pytest

from vrl.engine import ContinuousBatchPlanner, EngineLoop, Scheduler
from vrl.engine.scheduler_types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
)


class _EchoRunner:
    """Trivial runner that echoes request data as finished output."""

    execute_in_thread = False

    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        outputs = {}
        for req in scheduler_output.requests:
            outputs[req.request_id] = RequestOutput(
                request_id=req.request_id,
                data=req.data,
                finished=True,
                finish_reason="completed",
            )
        req_ids = [r.request_id for r in scheduler_output.requests]
        return ModelRunnerOutput(
            outputs=outputs,
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        )


def _build_engine(max_batch_size: int = 32) -> EngineLoop:
    return EngineLoop(
        scheduler=Scheduler(
            batch_planner=ContinuousBatchPlanner(max_batch_size=max_batch_size),
        ),
        model_runner=_EchoRunner(),
    )


@pytest.mark.asyncio
async def test_engine_loop_basic():
    """Single request flows through EngineLoop to completion."""
    engine = _build_engine()
    await engine.start()
    try:
        await engine.add_request("req-1", {"hello": "world"})
        result = await asyncio.wait_for(engine.get_result("req-1"), timeout=2.0)
        assert result == {"hello": "world"}
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_batch_planner_limits_one_tick_batch_size():
    """Batch size, not a resource counter, limits one scheduler tick."""
    engine = _build_engine(max_batch_size=1)
    await engine.start()
    try:
        await engine.add_request("req-1", "data-1")
        await engine.add_request("req-2", "data-2")

        r1 = await asyncio.wait_for(engine.get_result("req-1"), timeout=2.0)
        r2 = await asyncio.wait_for(engine.get_result("req-2"), timeout=2.0)

        assert r1 == "data-1"
        assert r2 == "data-2"
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_engine_loop_abort():
    """Abort cancels a request."""
    engine = _build_engine()
    await engine.start()
    try:
        await engine.add_request("req-1", "data")
        engine.abort_request("req-1")
        # get_result should raise since the request was aborted with an error
        # (abort sets status=ABORTED but no error, so it returns the request)
        request = await asyncio.wait_for(
            engine.scheduler.get_result("req-1"), timeout=2.0
        )
        from vrl.engine.scheduler_types import SchedulerStatus

        assert request.status == SchedulerStatus.ABORTED
    finally:
        await engine.stop()
