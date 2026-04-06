from __future__ import annotations

import httpx
import pytest
import pytest_asyncio

from wm_infra.api.server import create_app
from wm_infra.backends import BackendRegistry, GenieRolloutBackend
from wm_infra.backends.genie_runner import GenieRunner
from wm_infra.config import ControlPlaneConfig, DynamicsConfig, EngineConfig, StateCacheConfig, TokenizerConfig
from wm_infra.controlplane import SampleManifestStore, TemporalStore


def _test_config() -> EngineConfig:
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


@pytest.mark.asyncio
async def test_rl_catalog_endpoints_expose_default_env_and_tasks(client):
    envs_resp = await client.get("/v1/env-specs")
    assert envs_resp.status_code == 200
    envs = envs_resp.json()["environment_specs"]
    assert any(item["env_name"] == "toy-line-v0" for item in envs)
    assert any(item["env_name"] == "genie-token-grid-v0" for item in envs)

    tasks_resp = await client.get("/v1/task-specs")
    assert tasks_resp.status_code == 200
    tasks = tasks_resp.json()["task_specs"]
    assert {item["task_id"] for item in tasks} >= {
        "toy-line-train",
        "toy-line-eval",
        "genie-token-train",
        "genie-token-eval",
    }


@pytest.mark.asyncio
async def test_create_step_and_persist_transition_and_trajectory(client):
    create_resp = await client.post(
        "/v1/envs",
        json={"env_name": "toy-line-v0", "task_id": "toy-line-eval", "seed": 7, "policy_version": "pi-1"},
    )
    assert create_resp.status_code == 200
    created = create_resp.json()
    assert created["current_step"] == 0
    assert created["task_id"] == "toy-line-eval"
    assert created["state_handle_id"] is not None

    step_resp = await client.post(
        f"/v1/envs/{created['env_id']}/step",
        json={"action": [0.0, 0.0, 1.0], "policy_version": "pi-1"},
    )
    assert step_resp.status_code == 200
    stepped = step_resp.json()
    assert stepped["step_idx"] == 1
    assert stepped["reward"] == pytest.approx(-0.16)
    assert stepped["terminated"] is False
    assert stepped["truncated"] is False
    assert stepped["trajectory_id"] == created["trajectory_id"]
    assert stepped["transition_id"] is not None
    assert stepped["state_handle_id"] != created["state_handle_id"]

    transitions_resp = await client.get("/v1/transitions", params={"env_id": created["env_id"]})
    assert transitions_resp.status_code == 200
    transitions = transitions_resp.json()["transitions"]
    assert len(transitions) == 1
    assert transitions[0]["reward"] == pytest.approx(-0.16)

    trajectories_resp = await client.get("/v1/trajectories", params={"env_id": created["env_id"]})
    assert trajectories_resp.status_code == 200
    trajectories = trajectories_resp.json()["trajectories"]
    assert len(trajectories) == 1
    assert trajectories[0]["num_steps"] == 1
    assert trajectories[0]["return_value"] == pytest.approx(-0.16)


@pytest.mark.asyncio
async def test_fork_and_step_many_batch_sessions(client):
    create_resp = await client.post(
        "/v1/envs",
        json={"env_name": "toy-line-v0", "task_id": "toy-line-eval", "seed": 3},
    )
    root = create_resp.json()

    fork_resp = await client.post(f"/v1/envs/{root['env_id']}/fork", json={"branch_name": "alt"})
    assert fork_resp.status_code == 200
    forked = fork_resp.json()
    assert forked["env_id"] != root["env_id"]
    assert forked["episode_id"] == root["episode_id"]

    step_many_resp = await client.post(
        f"/v1/envs/{root['env_id']}/step_many",
        json={
            "env_ids": [forked["env_id"]],
            "actions": [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            "policy_version": "pi-batch",
            "checkpoint": True,
        },
    )
    assert step_many_resp.status_code == 200
    payload = step_many_resp.json()
    assert payload["env_ids"] == [root["env_id"], forked["env_id"]]
    assert len(payload["results"]) == 2
    assert payload["runtime"]["execution_path"] == "chunked_env_step"
    assert payload["runtime"]["chunk_count"] >= 1
    assert payload["runtime"]["max_chunk_size"] >= 2
    assert all(item["step_idx"] == 1 for item in payload["results"])
    assert all(item["checkpoint_id"] is not None for item in payload["results"])

    transitions_resp = await client.get("/v1/transitions")
    assert transitions_resp.status_code == 200
    assert transitions_resp.json()["count"] == 2


@pytest.mark.asyncio
async def test_done_session_requires_reset_before_another_step(client):
    create_resp = await client.post(
        "/v1/envs",
        json={"env_name": "toy-line-v0", "task_id": "toy-line-eval", "seed": 5, "max_episode_steps": 1},
    )
    created = create_resp.json()

    first_step = await client.post(
        f"/v1/envs/{created['env_id']}/step",
        json={"action": [0.0, 0.0, 1.0]},
    )
    assert first_step.status_code == 200
    assert first_step.json()["truncated"] is True

    second_step = await client.post(
        f"/v1/envs/{created['env_id']}/step",
        json={"action": [0.0, 0.0, 1.0]},
    )
    assert second_step.status_code == 400
    assert "reset" in second_step.json()["detail"]

    reset_resp = await client.post(
        f"/v1/envs/{created['env_id']}/reset",
        json={"seed": 5},
    )
    assert reset_resp.status_code == 200
    assert reset_resp.json()["current_step"] == 0


@pytest.mark.asyncio
async def test_genie_env_create_and_step_smoke(client):
    create_resp = await client.post(
        "/v1/envs",
        json={"env_name": "genie-token-grid-v0", "task_id": "genie-token-eval", "seed": 17},
    )
    assert create_resp.status_code == 200
    created = create_resp.json()
    assert created["env_name"] == "genie-token-grid-v0"
    assert created["info"]["goal_token_grid"] is not None

    step_resp = await client.post(
        f"/v1/envs/{created['env_id']}/step",
        json={"action": [0.0, 1.0, 0.0, 0.0, 0.0], "policy_version": "pi-genie"},
    )
    assert step_resp.status_code == 200
    stepped = step_resp.json()
    assert stepped["step_idx"] == 1
    assert stepped["terminated"] in (True, False)
    assert "token_l1" in stepped["info"]
