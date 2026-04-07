from __future__ import annotations

from wm_infra.controlplane import TemporalStore
from wm_infra.rl.runtime import RLEnvironmentManager


def test_step_many_splits_into_multiple_chunks_when_batch_exceeds_limit(tmp_path) -> None:
    manager = RLEnvironmentManager(TemporalStore(tmp_path / "temporal"), max_chunk_size=2)
    sessions = [
        manager.create_session(
            env_name="toy-line-v0",
            task_id="toy-line-eval",
            seed=10 + index,
            policy_version="pi-runtime",
            max_episode_steps=4,
            labels={},
            metadata={},
        )
        for index in range(5)
    ]

    try:
        response = manager.step_many(
            sessions[0].env_id,
            env_ids=[item.env_id for item in sessions[1:]],
            actions=[[0.0, 0.0, 1.0] for _ in sessions],
            policy_version="pi-runtime",
            checkpoint=False,
            metadata={},
        )
    finally:
        for session in sessions:
            manager.delete_session(session.env_id)

    assert len(response.results) == 5
    assert response.runtime["chunk_count"] == 3
    assert response.runtime["chunk_sizes"] == [2, 2, 1]
    assert response.runtime["max_chunk_size"] == 2
    assert response.runtime["batch_policy"]["mode"] == "sync"
    assert response.runtime["batch_policy"]["max_chunk_size"] == 2
    assert response.runtime["step_semantics"] == "sync_step_many"
    assert response.runtime["northbound_reset_policy"] == "explicit_reset_required"
    assert response.runtime["reward_stage_ms"] >= 0.0
    assert response.runtime["trajectory_persist_ms"] >= 0.0


def test_genie_env_step_many_uses_genie_action_contract(tmp_path) -> None:
    manager = RLEnvironmentManager(TemporalStore(tmp_path / "temporal"), max_chunk_size=2)
    sessions = [
        manager.create_session(
            env_name="genie-token-grid-v0",
            task_id="genie-token-eval",
            seed=20 + index,
            policy_version="pi-genie",
            max_episode_steps=3,
            labels={},
            metadata={},
        )
        for index in range(3)
    ]

    try:
        response = manager.step_many(
            sessions[0].env_id,
            env_ids=[item.env_id for item in sessions[1:]],
            actions=[[1.0, 0.0, 0.0, 0.0, 0.0] for _ in sessions],
            policy_version="pi-genie",
            checkpoint=False,
            metadata={},
        )
    finally:
        for session in sessions:
            manager.delete_session(session.env_id)

    assert len(response.results) == 3
    assert response.runtime["chunk_count"] == 2
    assert response.runtime["chunk_sizes"] == [2, 1]
    assert response.runtime["batch_policy"]["return_when_ready_count"] == 2
    assert all("token_l1" in item.info for item in response.results)
    assert all(len(item.observation) == manager.genie_world_model.spec.state_token_count for item in response.results)
