"""Tests for northbound auth and sample metrics."""

from __future__ import annotations

import pytest

from tests.base import BaseAppTestCase


@pytest.mark.asyncio
class TestApiGuardrails(BaseAppTestCase):
    async def test_api_key_guard_protects_sample_and_queue_endpoints(self, tmp_path):
        async with self.make_client(tmp_path, api_key="test-secret") as client:
            health_resp = await client.get("/v1/health")
            assert health_resp.status_code == 200

            denied = await client.post(
                "/v1/samples",
                json={
                    "task_type": "temporal_rollout",
                    "backend": "rollout-engine",
                    "model": "latent_dynamics",
                    "sample_spec": {"prompt": "guarded"},
                    "return_artifacts": ["latent"],
                },
            )
            assert denied.status_code == 401

            auth_metrics = await client.get("/metrics")
            assert auth_metrics.status_code == 200
            assert 'wm_api_auth_failures_total{endpoint="/v1/samples"} 1.0' in auth_metrics.text

            allowed = await client.post(
                "/v1/samples",
                headers={"X-API-Key": "test-secret"},
                json={
                    "task_type": "temporal_rollout",
                    "backend": "rollout-engine",
                    "model": "latent_dynamics",
                    "sample_spec": {"prompt": "guarded"},
                    "return_artifacts": ["latent"],
                },
            )
            assert allowed.status_code == 200
            assert allowed.json()["status"] == "succeeded"

    async def test_sample_metrics_are_exposed(self, tmp_path):
        async with self.make_client(tmp_path) as client:
            resp = await client.post(
                "/v1/samples",
                json={
                    "task_type": "temporal_rollout",
                    "backend": "rollout-engine",
                    "model": "latent_dynamics",
                    "sample_spec": {"prompt": "metrics"},
                    "return_artifacts": ["latent"],
                },
            )
            assert resp.status_code == 200

            await client.get("/v1/queue/status")
            metrics = await client.get("/metrics")
            assert metrics.status_code == 200
            assert 'wm_sample_total{backend="rollout-engine",status="succeeded"}' in metrics.text
            assert 'wm_sample_duration_seconds_count{backend="rollout-engine",status="succeeded"}' in metrics.text
            assert "wm_queue_depth" in metrics.text
