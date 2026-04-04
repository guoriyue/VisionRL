"""Integration tests for the FastAPI server.

Uses httpx AsyncClient with the ASGI transport to test the full
request/response cycle without starting a real server process.
"""

import asyncio
import json

import pytest
import pytest_asyncio
import httpx

from wm_infra.config import EngineConfig, DynamicsConfig, TokenizerConfig, StateCacheConfig
from wm_infra.api.server import create_app


def _test_config() -> EngineConfig:
    """Small CPU config for integration testing."""
    return EngineConfig(
        device="cpu",
        dtype="float32",
        dynamics=DynamicsConfig(
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            action_dim=8,
            latent_token_dim=6,
            max_rollout_steps=16,
        ),
        tokenizer=TokenizerConfig(
            spatial_downsample=2,
            temporal_downsample=1,
            latent_channels=16,
            fsq_levels=[4, 4, 4, 3, 3, 3],
        ),
        state_cache=StateCacheConfig(
            max_batch_size=8,
            max_rollout_steps=16,
            latent_dim=6,
            num_latent_tokens=16,
            pool_size_gb=0.1,
        ),
    )


@pytest_asyncio.fixture
async def client():
    """Create an httpx AsyncClient connected to the test app.

    Manually triggers FastAPI lifespan so the engine is initialized.
    """
    from contextlib import asynccontextmanager
    from asgi_lifespan import LifespanManager

    config = _test_config()
    app = create_app(config)

    async with LifespanManager(app) as manager:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=manager.app),
            base_url="http://test",
        ) as c:
            yield c


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        resp = await client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["model_loaded"] is True
        assert data["engine_running"] is True

    @pytest.mark.asyncio
    async def test_health_active_rollouts_zero_at_rest(self, client):
        resp = await client.get("/v1/health")
        assert resp.json()["active_rollouts"] == 0


class TestModels:
    @pytest.mark.asyncio
    async def test_list_models(self, client):
        resp = await client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert "latent_dynamics" in data["models"]


class TestRollout:
    @pytest.mark.asyncio
    async def test_basic_rollout(self, client):
        """Submit a basic rollout with default (random) initial state."""
        resp = await client.post("/v1/rollout", json={
            "num_steps": 2,
            "return_latents": True,
            "return_frames": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["steps_completed"] == 2
        assert data["elapsed_ms"] > 0
        assert data["latents"] is not None
        assert len(data["latents"]) == 2

    @pytest.mark.asyncio
    async def test_rollout_with_latent_input(self, client):
        """Submit rollout with explicit initial latent state."""
        N, D = 16, 6
        latent = [[0.1 * i for _ in range(D)] for i in range(N)]
        resp = await client.post("/v1/rollout", json={
            "initial_latent": latent,
            "num_steps": 3,
            "return_latents": True,
            "return_frames": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["steps_completed"] == 3

    @pytest.mark.asyncio
    async def test_rollout_with_actions(self, client):
        """Submit rollout with explicit action sequence."""
        N, D, A = 16, 6, 8
        latent = [[0.0] * D for _ in range(N)]
        actions = [[0.1] * A for _ in range(2)]
        resp = await client.post("/v1/rollout", json={
            "initial_latent": latent,
            "actions": actions,
            "num_steps": 2,
            "return_latents": True,
            "return_frames": False,
        })
        assert resp.status_code == 200
        assert resp.json()["steps_completed"] == 2

    @pytest.mark.asyncio
    async def test_concurrent_rollouts(self, client):
        """Multiple concurrent requests should all succeed."""
        async def single_request(i):
            return await client.post("/v1/rollout", json={
                "num_steps": 2,
                "return_latents": True,
                "return_frames": False,
            })

        responses = await asyncio.gather(*[single_request(i) for i in range(4)])
        for resp in responses:
            assert resp.status_code == 200
            assert resp.json()["steps_completed"] == 2


class TestStreaming:
    @pytest.mark.asyncio
    async def test_sse_streaming(self, client):
        """stream=true should return SSE events."""
        async with client.stream(
            "POST",
            "/v1/rollout",
            json={
                "num_steps": 3,
                "return_latents": True,
                "return_frames": False,
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]

            events = []
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    payload = line[len("data: "):]
                    if payload == "[DONE]":
                        events.append("[DONE]")
                    else:
                        events.append(json.loads(payload))

            # Should have 3 step events + [DONE]
            assert len(events) == 4
            assert events[-1] == "[DONE]"
            assert events[0]["step"] == 0
            assert events[1]["step"] == 1
            assert events[2]["step"] == 2
            # Latents should be present since return_latents=True
            assert events[0]["latent"] is not None


class TestMetrics:
    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, client):
        """The /metrics endpoint should return Prometheus format."""
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        text = resp.text
        assert "wm_request_total" in text or "wm_batch_size" in text


class TestErrors:
    @pytest.mark.asyncio
    async def test_invalid_num_steps(self, client):
        """num_steps must be 1-128."""
        resp = await client.post("/v1/rollout", json={
            "num_steps": 0,
            "return_frames": False,
        })
        assert resp.status_code == 422  # validation error

    @pytest.mark.asyncio
    async def test_nonexistent_job(self, client):
        """GET /v1/rollout/{job_id} for unknown job should 404."""
        resp = await client.get("/v1/rollout/nonexistent-id")
        assert resp.status_code == 404
