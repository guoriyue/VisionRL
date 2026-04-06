"""Integration tests for the FastAPI server.

Uses httpx AsyncClient with the ASGI transport to test the full
request/response cycle without starting a real server process.
"""

import asyncio
import json
import time
from urllib.parse import quote

import httpx
import pytest
import pytest_asyncio

from wm_infra.api.server import create_app
from wm_infra.backends import BackendRegistry, GenieRolloutBackend
from wm_infra.backends.genie_runner import GenieRunner
from wm_infra.config import ControlPlaneConfig, DynamicsConfig, EngineConfig, StateCacheConfig, TokenizerConfig
from wm_infra.controlplane import ArtifactKind, SampleManifestStore, TemporalStore


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
        controlplane=ControlPlaneConfig(),
    )


@pytest_asyncio.fixture
async def client(tmp_path):
    """Create an httpx AsyncClient connected to the test app."""
    from asgi_lifespan import LifespanManager

    config = _test_config()
    config.controlplane.cosmos_output_root = str(tmp_path / "cosmos")
    config.controlplane.wan_output_root = str(tmp_path / "wan")
    temporal_store = TemporalStore(tmp_path / "temporal")
    registry = BackendRegistry()
    genie_runner = GenieRunner()
    genie_runner._mode = "stub"
    genie_runner.load = lambda: "stub"  # type: ignore[method-assign]
    registry.register(
        GenieRolloutBackend(
            temporal_store,
            output_root=tmp_path / "genie",
            runner=genie_runner,
        )
    )
    app = create_app(
        config,
        sample_store=SampleManifestStore(tmp_path),
        backend_registry=registry,
        temporal_store=temporal_store,
    )

    async with LifespanManager(app) as manager:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=manager.app),
            base_url="http://test",
        ) as c:
            yield c


async def _wait_for_terminal_sample(client: httpx.AsyncClient, sample_id: str, timeout_s: float = 2.0):
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        resp = await client.get(f"/v1/samples/{sample_id}")
        assert resp.status_code == 200
        last = resp.json()
        if last["status"] in {"succeeded", "failed", "accepted"}:
            return last
        await asyncio.sleep(0.05)
    raise AssertionError(f"sample {sample_id} did not reach terminal state; last={last}")


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


class TestBackends:
    @pytest.mark.asyncio
    async def test_list_backends(self, client):
        resp = await client.get("/v1/backends")
        assert resp.status_code == 200
        backends = {backend["name"]: backend for backend in resp.json()["backends"]}
        assert "cosmos-predict" in backends
        assert "rollout-engine" in backends
        assert "wan-video" in backends
        assert backends["cosmos-predict"]["runner_mode"] == "stub"
        assert backends["cosmos-predict"]["async_queue"] is True
        assert backends["wan-video"]["shell_runner_configured"] is False
        assert backends["wan-video"]["runner_mode"] == "stub"
        assert backends["wan-video"]["async_queue"] is True
        assert backends["wan-video"]["admission_max_vram_gb"] == 32.0


class TestRollout:
    @pytest.mark.asyncio
    async def test_basic_rollout(self, client):
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
        N, D = 16, 6
        latent = [[0.1 * i for _ in range(D)] for i in range(N)]
        resp = await client.post("/v1/rollout", json={
            "initial_latent": latent,
            "num_steps": 3,
            "return_latents": True,
            "return_frames": False,
        })
        assert resp.status_code == 200
        assert resp.json()["steps_completed"] == 3

    @pytest.mark.asyncio
    async def test_rollout_with_actions(self, client):
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
        async def single_request(_i):
            return await client.post("/v1/rollout", json={
                "num_steps": 2,
                "return_latents": True,
                "return_frames": False,
            })

        responses = await asyncio.gather(*[single_request(i) for i in range(4)])
        for resp in responses:
            assert resp.status_code == 200
            assert resp.json()["steps_completed"] == 2


class TestSamples:
    @pytest.mark.asyncio
    async def test_create_and_get_sample_manifest(self, client):
        resp = await client.post("/v1/samples", json={
            "task_type": "world_model_rollout",
            "backend": "rollout-engine",
            "model": "latent_dynamics",
            "return_artifacts": [ArtifactKind.LATENT.value],
            "task_config": {"num_steps": 2, "frame_count": 9, "width": 832, "height": 480, "memory_profile": "low_vram"},
            "sample_spec": {
                "prompt": "predict the next dog jump",
                "width": 832,
                "height": 480,
            },
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "succeeded"
        assert data["runtime"]["steps_completed"] == 2
        assert data["task_config"]["num_steps"] == 2
        assert data["task_config"]["frame_count"] == 9
        assert data["task_config"]["memory_profile"] == "low_vram"
        assert data["artifacts"][0]["kind"] == ArtifactKind.LATENT.value
        assert data["resource_estimate"]["bottleneck"] == "frame_pressure"

        sample_id = data["sample_id"]
        get_resp = await client.get(f"/v1/samples/{sample_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["sample_id"] == sample_id

    @pytest.mark.asyncio
    async def test_create_sample_supports_legacy_num_steps_in_sample_metadata(self, client):
        resp = await client.post("/v1/samples", json={
            "task_type": "world_model_rollout",
            "backend": "rollout-engine",
            "model": "latent_dynamics",
            "return_artifacts": [ArtifactKind.LATENT.value],
            "sample_spec": {
                "prompt": "predict the next dog jump",
                "metadata": {"num_steps": 2},
            },
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_create_cosmos_sample_queues_and_completes(self, client):
        resp = await client.post("/v1/samples", json={
            "task_type": "text_to_video",
            "backend": "cosmos-predict",
            "model": "cosmos-predict1-7b-text2world",
            "sample_spec": {
                "prompt": "A warehouse robot navigating around boxes.",
                "width": 1024,
                "height": 640,
            },
            "task_config": {"num_steps": 12, "frame_count": 16, "width": 1024, "height": 640},
            "cosmos_config": {"variant": "predict1_text2world", "model_size": "7B", "frames_per_second": 16},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"
        terminal = await _wait_for_terminal_sample(client, data["sample_id"])
        assert terminal["status"] == "succeeded"
        assert terminal["runtime"]["runner_mode"] == "stub"

    @pytest.mark.asyncio
    async def test_create_sample_persists_under_experiment_directory(self, client, tmp_path):
        resp = await client.post("/v1/samples", json={
            "task_type": "world_model_rollout",
            "backend": "rollout-engine",
            "model": "latent_dynamics",
            "task_config": {"num_steps": 1},
            "experiment": {"experiment_id": "exp_server_test"},
            "sample_spec": {"prompt": "organized sample"},
        })
        assert resp.status_code == 200
        sample_id = resp.json()["sample_id"]
        assert (tmp_path / "samples" / "exp_server_test" / f"{sample_id}.json").exists()

    @pytest.mark.asyncio
    async def test_create_wan_video_sample_returns_queued(self, client):
        """Wan video POST /v1/samples now returns immediately with queued status."""
        resp = await client.post("/v1/samples", json={
            "task_type": "text_to_video",
            "backend": "wan-video",
            "model": "wan2.2-t2v-A14B",
            "sample_spec": {"prompt": "a corgi surfing through a data center"},
            "wan_config": {
                "num_steps": 4,
                "frame_count": 9,
                "width": 832,
                "height": 480,
                "memory_profile": "low_vram"
            }
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert data["wan_config"]["frame_count"] == 9
        assert data["resource_estimate"]["estimated_vram_gb"] > 0
        assert data["runtime"]["runner"] == "stub"
        assert data["runtime"]["async"] is True
        assert data["runtime"]["status_history"][0]["status"] == "queued"
        assert data["runtime"]["admission"]["admitted"] is True
        assert data["runtime"]["queue_position"] >= 0
        assert data["metadata"]["async"] is True

    @pytest.mark.asyncio
    async def test_wan_async_job_completes_in_background(self, client):
        """Submit a Wan job async and verify it transitions to succeeded."""
        resp = await client.post("/v1/samples", json={
            "task_type": "text_to_video",
            "backend": "wan-video",
            "model": "wan2.2-t2v-A14B",
            "sample_spec": {"prompt": "a cat jumping over a fence"},
            "wan_config": {"num_steps": 4, "frame_count": 9, "width": 832, "height": 480}
        })
        assert resp.status_code == 200
        sample_id = resp.json()["sample_id"]
        assert resp.json()["status"] == "queued"

        final = await _wait_for_terminal_sample(client, sample_id)
        assert final["status"] == "accepted"
        assert final["runtime"]["runner"] == "stub"
        assert final["runtime"]["elapsed_ms"] >= 0
        kinds = {a["kind"] for a in final["artifacts"]}
        assert "video" not in kinds
        assert {"log", "metadata"}.issubset(kinds)
        assert final["metadata"]["stubbed"] is True

    @pytest.mark.asyncio
    async def test_create_wan_video_requires_prompt_for_t2v(self, client):
        resp = await client.post("/v1/samples", json={
            "task_type": "text_to_video",
            "backend": "wan-video",
            "model": "wan2.2-t2v-A14B",
            "sample_spec": {"prompt": "   "},
        })
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_create_wan_video_can_be_rejected_by_admission(self, tmp_path):
        from asgi_lifespan import LifespanManager

        config = _test_config()
        config.controlplane.wan_output_root = str(tmp_path / "wan")
        config.controlplane.wan_admission_max_vram_gb = 10.0
        app = create_app(config, sample_store=SampleManifestStore(tmp_path))

        async with LifespanManager(app) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test",
            ) as c:
                resp = await c.post("/v1/samples", json={
                    "task_type": "text_to_video",
                    "backend": "wan-video",
                    "model": "wan2.2-t2v-A14B",
                    "sample_spec": {"prompt": "too big"},
                    "wan_config": {"num_steps": 4, "frame_count": 17, "width": 832, "height": 480},
                })
                assert resp.status_code == 200
                data = resp.json()
                assert data["status"] == "rejected"
                assert data["runtime"]["admission"]["admitted"] is False
                assert "estimated_vram_gb" in " ".join(data["runtime"]["admission"]["reasons"])
                fetched = await c.get(f"/v1/samples/{data['sample_id']}")
                assert fetched.status_code == 200
                assert fetched.json()["status"] == "rejected"

    @pytest.mark.asyncio
    async def test_create_sample_rejects_unknown_backend(self, client):
        resp = await client.post("/v1/samples", json={
            "task_type": "world_model_rollout",
            "backend": "missing-backend",
            "model": "latent_dynamics",
            "sample_spec": {"prompt": "bad backend"},
        })
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_list_samples(self, client):
        create_resp = await client.post("/v1/samples", json={
            "task_type": "text_to_video",
            "backend": "wan-video",
            "model": "wan2.2-t2v-A14B",
            "experiment": {"experiment_id": "exp_list"},
            "sample_spec": {"prompt": "list me"},
        })
        assert create_resp.status_code == 200

        resp = await client.get("/v1/samples", params={"backend": "wan-video", "experiment_id": "exp_list"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 1
        assert data["samples"][0]["backend"] == "wan-video"

    @pytest.mark.asyncio
    async def test_get_missing_sample(self, client):
        resp = await client.get("/v1/samples/does-not-exist")
        assert resp.status_code == 404


class TestArtifacts:
    @pytest.mark.asyncio
    async def test_list_artifacts(self, client):
        resp = await client.post("/v1/samples", json={
            "task_type": "text_to_video",
            "backend": "wan-video",
            "model": "wan2.2-t2v-A14B",
            "sample_spec": {"prompt": "artifact list"},
        })
        sample_id = resp.json()["sample_id"]
        data = await _wait_for_terminal_sample(client, sample_id)
        assert data["status"] == "accepted"

        art_resp = await client.get(f"/v1/samples/{sample_id}/artifacts")
        assert art_resp.status_code == 200
        artifact_kinds = {artifact["kind"] for artifact in art_resp.json()["artifacts"]}
        assert {"log", "metadata"}.issubset(artifact_kinds)
        assert "video" not in artifact_kinds
        assert art_resp.json()["count"] == len(art_resp.json()["artifacts"])

    @pytest.mark.asyncio
    async def test_get_artifact_metadata(self, client):
        """Submit wan job, wait for completion, check artifact metadata endpoint."""
        resp = await client.post("/v1/samples", json={
            "task_type": "text_to_video",
            "backend": "wan-video",
            "model": "wan2.2-t2v-A14B",
            "sample_spec": {"prompt": "artifact test"},
            "wan_config": {"num_steps": 4, "frame_count": 9, "width": 832, "height": 480}
        })
        sample_id = resp.json()["sample_id"]

        data = await _wait_for_terminal_sample(client, sample_id)
        assert data["status"] == "accepted"

        # Fetch log artifact metadata
        log_artifact_id = quote(f"{sample_id}:log", safe="")
        art_resp = await client.get(f"/v1/samples/{sample_id}/artifacts/{log_artifact_id}")
        assert art_resp.status_code == 200
        art_data = art_resp.json()
        assert art_data["kind"] == "log"
        assert art_data["mime_type"] == "text/plain"

    @pytest.mark.asyncio
    async def test_get_artifact_content(self, client):
        """Submit wan job, wait for completion, download artifact content."""
        resp = await client.post("/v1/samples", json={
            "task_type": "text_to_video",
            "backend": "wan-video",
            "model": "wan2.2-t2v-A14B",
            "sample_spec": {"prompt": "content test"},
            "wan_config": {"num_steps": 4, "frame_count": 9, "width": 832, "height": 480}
        })
        sample_id = resp.json()["sample_id"]

        await _wait_for_terminal_sample(client, sample_id)

        # Fetch the log file content
        log_artifact_id = quote(f"{sample_id}:log", safe="")
        content_resp = await client.get(f"/v1/samples/{sample_id}/artifacts/{log_artifact_id}/content")
        assert content_resp.status_code == 200
        assert "stub mode" in content_resp.text

    @pytest.mark.asyncio
    async def test_get_artifact_not_found(self, client):
        """Should 404 for missing artifact."""
        resp = await client.post("/v1/samples", json={
            "task_type": "text_to_video",
            "backend": "wan-video",
            "model": "wan2.2-t2v-A14B",
            "sample_spec": {"prompt": "missing art"},
        })
        sample_id = resp.json()["sample_id"]
        art_resp = await client.get(f"/v1/samples/{sample_id}/artifacts/nonexistent")
        assert art_resp.status_code == 404


class TestTemporalControlPlane:
    @pytest.mark.asyncio
    async def test_create_temporal_entities_and_genie_rollout(self, client):
        ep_resp = await client.post("/v1/episodes", json={"title": "Temporal MVP"})
        assert ep_resp.status_code == 200
        episode = ep_resp.json()

        branch_resp = await client.post("/v1/branches", json={
            "episode_id": episode["episode_id"],
            "name": "main",
        })
        assert branch_resp.status_code == 200
        branch = branch_resp.json()

        state_resp = await client.post("/v1/state-handles", json={
            "episode_id": episode["episode_id"],
            "branch_id": branch["branch_id"],
            "kind": "latent",
            "dtype": "float16",
            "shape": [16, 6],
        })
        assert state_resp.status_code == 200
        state = state_resp.json()

        sample_resp = await client.post("/v1/samples", json={
            "task_type": "genie_rollout",
            "backend": "genie-rollout",
            "model": "genie-local",
            "sample_spec": {"prompt": "roll forward in time"},
            "temporal": {
                "episode_id": episode["episode_id"],
                "branch_id": branch["branch_id"],
                "state_handle_id": state["state_handle_id"],
            },
            "task_config": {"num_steps": 3, "width": 832, "height": 480},
            "genie_config": {"num_frames": 9, "num_prompt_frames": 4, "maskgit_steps": 3, "temperature": 0.1},
            "return_artifacts": ["metadata"],
        })
        assert sample_resp.status_code == 200
        queued = sample_resp.json()
        assert queued["status"] == "queued"
        assert queued["runtime"]["async"] is True
        assert queued["genie_config"]["num_frames"] == 9
        assert queued["task_config"]["frame_count"] == 9
        sample = await _wait_for_terminal_sample(client, queued["sample_id"])
        assert sample["status"] == "succeeded"
        assert sample["runtime"]["runner"] == "genie-stub"
        assert sample["runtime"]["runner_mode"] == "stub"
        assert sample["genie_config"]["num_frames"] == 9
        assert sample["genie_config"]["num_prompt_frames"] == 4
        assert sample["runtime"]["genie_config"]["maskgit_steps"] == 3
        assert sample["runtime"]["genie_config"]["temperature"] == 0.1
        assert sample["temporal"]["episode_id"] == episode["episode_id"]
        assert sample["temporal"]["rollout_id"] is not None
        assert sample["temporal"]["checkpoint_id"] is not None
        assert sample["temporal"]["state_handle_id"] is not None

        # Verify persisted artifacts exist
        artifact_kinds = {a["kind"] for a in sample["artifacts"]}
        assert "metadata" in artifact_kinds
        assert "log" in artifact_kinds
        assert "latent" in artifact_kinds  # tokens.npy

        # Verify tokens artifact has shape metadata
        token_artifacts = [a for a in sample["artifacts"] if a["artifact_id"].endswith(":tokens")]
        assert len(token_artifacts) == 1
        assert token_artifacts[0]["metadata"]["format"] == "numpy"
        assert token_artifacts[0]["metadata"]["dtype"] == "uint32"

        # Verify state artifact exists
        state_artifacts = [a for a in sample["artifacts"] if a["artifact_id"].endswith(":state")]
        assert len(state_artifacts) == 1

        # Verify runtime has file paths
        assert sample["runtime"]["tokens_path"] is not None
        assert sample["runtime"]["state_path"] is not None
        assert sample["runtime"]["log_path"] is not None
        assert sample["runtime"]["frames_generated"] > 0
        assert sample["runtime"]["tokens_generated"] > 0

        # Verify metadata correctly reports mode
        assert sample["metadata"]["runner_mode"] == "stub"
        assert sample["metadata"]["stubbed"] is True

        rollout_resp = await client.get(f"/v1/rollouts/{sample['temporal']['rollout_id']}")
        assert rollout_resp.status_code == 200
        rollout = rollout_resp.json()
        assert rollout["status"] == "succeeded"
        assert rollout["checkpoint_ids"] == [sample["temporal"]["checkpoint_id"]]
        assert rollout["metrics"]["frames_generated"] > 0
        assert rollout["metrics"]["tokens_generated"] > 0

        checkpoints_resp = await client.get("/v1/checkpoints", params={"rollout_id": sample["temporal"]["rollout_id"]})
        assert checkpoints_resp.status_code == 200
        assert checkpoints_resp.json()["count"] == 1

        # Verify state handle has real URI
        sh_resp = await client.get(f"/v1/state-handles/{sample['temporal']['state_handle_id']}")
        assert sh_resp.status_code == 200
        sh = sh_resp.json()
        assert sh["uri"] is not None
        assert sh["uri"].startswith("file://")
        assert sh["dtype"] == "uint32"
        assert sh["shape"] == [sample["runtime"]["total_frames"], 16, 16]
        assert sh["checkpoint_id"] == sample["temporal"]["checkpoint_id"]

    @pytest.mark.asyncio
    async def test_genie_rollout_artifact_content_downloadable(self, client):
        """Submit a Genie rollout, then download the runner log content."""
        ep_resp = await client.post("/v1/episodes", json={"title": "Download test"})
        episode = ep_resp.json()
        branch_resp = await client.post("/v1/branches", json={
            "episode_id": episode["episode_id"], "name": "main",
        })
        branch = branch_resp.json()
        state_resp = await client.post("/v1/state-handles", json={
            "episode_id": episode["episode_id"],
            "branch_id": branch["branch_id"],
            "kind": "latent", "dtype": "float16", "shape": [16, 6],
        })
        state = state_resp.json()

        sample_resp = await client.post("/v1/samples", json={
            "task_type": "genie_rollout",
            "backend": "genie-rollout",
            "model": "genie-local",
            "sample_spec": {"prompt": "download test"},
            "temporal": {
                "episode_id": episode["episode_id"],
                "branch_id": branch["branch_id"],
                "state_handle_id": state["state_handle_id"],
            },
            "task_config": {"num_steps": 1},
        })
        assert sample_resp.status_code == 200
        sample = await _wait_for_terminal_sample(client, sample_resp.json()["sample_id"])
        sample_id = sample["sample_id"]

        log_artifact_id = quote(f"{sample_id}:log", safe="")
        content_resp = await client.get(f"/v1/samples/{sample_id}/artifacts/{log_artifact_id}/content")
        assert content_resp.status_code == 200
        assert "Genie runner mode: stub" in content_resp.text

    @pytest.mark.asyncio
    async def test_list_backends_includes_genie_rollout(self, client):
        resp = await client.get("/v1/backends")
        assert resp.status_code == 200
        backends = {backend["name"]: backend for backend in resp.json()["backends"]}
        assert "genie-rollout" in backends
        assert backends["genie-rollout"]["stateful"] is True
        assert backends["genie-rollout"]["runner_mode"] == "stub"
        assert backends["genie-rollout"]["async_queue"] is True
        assert "model" in backends["genie-rollout"]

    @pytest.mark.asyncio
    async def test_default_genie_backend_uses_controlplane_batching_config(self, tmp_path):
        from asgi_lifespan import LifespanManager

        config = _test_config()
        config.controlplane.manifest_store_root = str(tmp_path / "manifests")
        config.controlplane.genie_output_root = str(tmp_path / "genie")
        config.controlplane.wan_output_root = str(tmp_path / "wan")
        config.controlplane.cosmos_output_root = str(tmp_path / "cosmos")
        config.controlplane.genie_max_batch_size = 3
        config.controlplane.genie_batch_wait_ms = 9.0

        temporal_store = TemporalStore(tmp_path / "temporal")
        app = create_app(
            config,
            sample_store=SampleManifestStore(tmp_path / "manifests"),
            temporal_store=temporal_store,
        )

        async with LifespanManager(app):
            backend = app.state.backend_registry.get("genie-rollout")
            assert backend is not None
            assert backend._transition_batcher.max_batch_size == 3
            assert backend._transition_batcher.batch_wait_ms == 9.0
            assert app.state.genie_job_queue is not None
            assert app.state.genie_job_queue.snapshot()["max_batch_size"] == 2

    @pytest.mark.asyncio
    async def test_genie_rollout_accepts_inline_raw_tokens(self, client):
        ep_resp = await client.post("/v1/episodes", json={"title": "Raw token input"})
        episode = ep_resp.json()
        branch_resp = await client.post("/v1/branches", json={"episode_id": episode["episode_id"], "name": "main"})
        branch = branch_resp.json()
        state_resp = await client.post("/v1/state-handles", json={
            "episode_id": episode["episode_id"],
            "branch_id": branch["branch_id"],
            "kind": "latent",
        })
        state = state_resp.json()

        inline_tokens = list(range(4 * 16 * 16))
        sample_resp = await client.post("/v1/samples", json={
            "task_type": "genie_rollout",
            "backend": "genie-rollout",
            "model": "genie-local",
            "sample_spec": {"prompt": "raw tokens path"},
            "temporal": {
                "episode_id": episode["episode_id"],
                "branch_id": branch["branch_id"],
                "state_handle_id": state["state_handle_id"],
            },
            "token_input": {
                "source": "inline",
                "tokenizer_family": "magvit2",
                "layout": "flat",
                "shape": [4, 16, 16],
                "inline_tokens": inline_tokens,
                "dtype": "uint32",
            },
            "task_config": {"num_steps": 1, "frame_count": 8},
        })
        sample = await _wait_for_terminal_sample(client, sample_resp.json()["sample_id"])
        assert sample["status"] == "succeeded"
        assert sample["runtime"]["token_input"]["token_input_mode"] == "raw_tokens"
        assert sample["runtime"]["token_input"]["magvit2_scaffold"] is True
        input_artifacts = [a for a in sample["artifacts"] if a["artifact_id"].endswith(":input-tokens")]
        assert len(input_artifacts) == 1


class TestQueueStatus:
    @pytest.mark.asyncio
    async def test_queue_status_endpoint(self, client):
        resp = await client.get("/v1/queue/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["queue_enabled"] is True
        assert "pending" in data
        assert "running" in data
        assert "total_tracked" in data
        assert "queues" in data
        assert "max_queue_size" in data["queues"]["wan"]
        assert isinstance(data["queues"]["wan"]["queued_sample_ids"], list)


class TestShellRunner:
    @pytest.mark.asyncio
    async def test_wan_shell_runner_failure_is_recorded(self, tmp_path):
        from asgi_lifespan import LifespanManager

        config = _test_config()
        config.controlplane.wan_output_root = str(tmp_path / "wan")
        config.controlplane.wan_shell_runner = "python -c \"import sys; print('runner boom'); sys.exit(7)\""
        app = create_app(config, sample_store=SampleManifestStore(tmp_path))

        async with LifespanManager(app) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test",
            ) as shell_client:
                resp = await shell_client.post("/v1/samples", json={
                    "task_type": "text_to_video",
                    "backend": "wan-video",
                    "model": "wan2.2-t2v-A14B",
                    "sample_spec": {"prompt": "fail loudly"},
                })

                # Job is queued async — wait for background worker to complete it
                assert resp.status_code == 200
                sample_id = resp.json()["sample_id"]
                assert resp.json()["status"] == "queued"

                data = await _wait_for_terminal_sample(shell_client, sample_id)
                assert data["status"] == "failed"
                assert data["runtime"]["runner"] == "shell"
                assert data["runtime"]["returncode"] == 7
                failure_artifacts = [artifact for artifact in data["artifacts"] if artifact["metadata"].get("role") == "failure"]
                assert len(failure_artifacts) == 1
                failure_meta = await shell_client.get(f"/v1/samples/{sample_id}/artifacts/{quote(f'{sample_id}:failure', safe='')}")
                assert failure_meta.status_code == 200


class TestOfficialRunner:
    @pytest.mark.asyncio
    async def test_official_runner_mode_reflected_in_backend(self, tmp_path):
        """Verify that when wan_repo_dir is set, runner_mode is 'official'."""
        from asgi_lifespan import LifespanManager
        from wm_infra.backends import BackendRegistry, WanVideoBackend

        config = _test_config()
        wan_root = str(tmp_path / "wan")
        registry = BackendRegistry()
        wan_backend = WanVideoBackend(
            wan_root,
            wan_repo_dir="/fake/Wan2.2",
            wan_conda_env="test_env",
        )
        registry.register(wan_backend)

        app = create_app(config, sample_store=SampleManifestStore(tmp_path), backend_registry=registry)

        async with LifespanManager(app) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test",
            ) as c:
                resp = await c.get("/v1/backends")
                assert resp.status_code == 200
                backends = {b["name"]: b for b in resp.json()["backends"]}
                assert backends["wan-video"]["runner_mode"] == "official"

    def test_official_runner_builds_i2v_command_with_reference(self, tmp_path):
        from wm_infra.backends import WanVideoBackend
        from wm_infra.controlplane import ProduceSampleRequest

        backend = WanVideoBackend(
            str(tmp_path / "wan"),
            wan_repo_dir="/fake/Wan2.2",
            wan_conda_env="test_env",
        )
        request = ProduceSampleRequest.model_validate({
            "task_type": "image_to_video",
            "backend": "wan-video",
            "model": "wan2.2-i2v-A14B",
            "sample_spec": {"prompt": "animate this", "references": ["/tmp/input.png"]},
        })
        cmd = backend._build_official_command(request, "sample123", backend._resolve_wan_config(request), tmp_path / "out.mp4")
        assert "--task i2v-A14B" in cmd
        assert "--image /tmp/input.png" in cmd
        assert "--save_file" in cmd


class TestStreaming:
    @pytest.mark.asyncio
    async def test_sse_streaming(self, client):
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

            assert len(events) == 4
            assert events[-1] == "[DONE]"
            assert events[0]["step"] == 0
            assert events[1]["step"] == 1
            assert events[2]["step"] == 2
            assert events[0]["latent"] is not None


class TestMetrics:
    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, client):
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        text = resp.text
        assert "wm_request_total" in text or "wm_batch_size" in text


class TestErrors:
    @pytest.mark.asyncio
    async def test_invalid_num_steps(self, client):
        resp = await client.post("/v1/rollout", json={
            "num_steps": 0,
            "return_frames": False,
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_nonexistent_job(self, client):
        resp = await client.get("/v1/rollout/nonexistent-id")
        assert resp.status_code == 404
