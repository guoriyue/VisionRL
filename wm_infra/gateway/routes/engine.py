"""Gateway routes for engine health and low-level rollout APIs."""

from __future__ import annotations

import base64
import io
from typing import AsyncIterator

import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from wm_infra.api.metrics import ACTIVE_ROLLOUTS, REQUEST_DURATION, REQUEST_TOTAL
from wm_infra.api.protocol import RolloutRequest, RolloutResponse, SSE_DONE, StepResult
from wm_infra.engine.types import RolloutJob
from wm_infra.gateway.bootstrap import build_rollout_job
from wm_infra.gateway.state import get_gateway_runtime


def register_engine_routes(app: FastAPI) -> None:
    """Register engine health and rollout routes."""
    router = APIRouter()

    @router.get("/v1/health")
    async def health(request: Request):
        runtime = get_gateway_runtime(request)
        engine = runtime.engine
        ready = engine is not None and engine.is_running
        ACTIVE_ROLLOUTS.set(engine.engine.state_manager.num_active if engine else 0)
        return {
            "status": "ready" if ready else "not_ready",
            "model_loaded": engine is not None,
            "engine_running": engine.is_running if engine else False,
            "active_rollouts": engine.engine.state_manager.num_active if engine else 0,
            "memory_used_gb": engine.engine.state_manager.memory_used_gb if engine else 0.0,
        }

    @router.post("/v1/rollout")
    async def submit_rollout(request: RolloutRequest, http_request: Request):
        import time as _time

        runtime = get_gateway_runtime(http_request)
        engine = runtime.engine
        if engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")

        job = build_rollout_job(request, runtime.config)

        if request.stream:
            REQUEST_TOTAL.labels(status="stream").inc()
            return StreamingResponse(
                _stream_rollout(engine, job, request),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        t0 = _time.monotonic()
        try:
            result = await engine.submit(job)
            REQUEST_TOTAL.labels(status="success").inc()
        except Exception:
            REQUEST_TOTAL.labels(status="error").inc()
            raise
        finally:
            REQUEST_DURATION.observe(_time.monotonic() - t0)

        response = RolloutResponse(
            job_id=result.job_id,
            model=request.model,
            steps_completed=result.steps_completed,
            elapsed_ms=result.elapsed_ms,
        )
        if result.predicted_latents is not None:
            response.latents = result.predicted_latents.cpu().tolist()
        if result.predicted_frames is not None:
            frames_b64 = []
            for t in range(result.predicted_frames.shape[0]):
                frame = result.predicted_frames[t]
                frame_np = (frame.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
                try:
                    from PIL import Image
                    img = Image.fromarray(frame_np)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    frames_b64.append(base64.b64encode(buf.getvalue()).decode())
                except ImportError:
                    frames_b64.append("")
            response.frames_b64 = frames_b64
        return response

    @router.get("/v1/rollout/{job_id}")
    async def get_rollout(job_id: str, request: Request):
        runtime = get_gateway_runtime(request)
        engine = runtime.engine
        if engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")
        result = engine.engine.get_result(job_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"job_id": result.job_id, "steps_completed": result.steps_completed, "elapsed_ms": result.elapsed_ms}

    app.include_router(router)


async def _stream_rollout(engine, job: RolloutJob, request: RolloutRequest) -> AsyncIterator[str]:
    async for step_idx, latent in engine.submit_stream(job):
        step_result = StepResult(step=step_idx)
        if request.return_latents:
            step_result.latent = latent.cpu().tolist()
        yield step_result.to_sse()
    yield SSE_DONE
