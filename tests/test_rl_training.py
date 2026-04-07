from __future__ import annotations

from pathlib import Path

from wm_infra.controlplane import TemporalStore
from wm_infra.rl.training import ExperimentSpec, LocalActorCriticLearner, SynchronousCollector, run_local_experiment
from wm_infra.rl.runtime import RLEnvironmentManager


def test_synchronous_collector_persists_batched_transitions(tmp_path) -> None:
    temporal_root = tmp_path / "temporal"
    store = TemporalStore(temporal_root)
    manager = RLEnvironmentManager(store)
    spec = ExperimentSpec(
        num_envs=4,
        horizon=3,
        updates=1,
        eval_interval=1,
        eval_episodes=2,
        eval_num_envs=2,
        temporal_root=str(temporal_root),
    )
    collector = SynchronousCollector(manager, spec)
    learner = LocalActorCriticLearner(obs_dim=2, num_actions=manager.action_dim_for_env(spec.train_env_name), spec=spec)

    try:
        batch = collector.collect(learner, policy_version="toy-line-collector:update-0")
    finally:
        collector.close()

    assert len(batch.transition_ids) == spec.num_envs * spec.horizon
    assert len(store.transitions.list()) == spec.num_envs * spec.horizon
    assert batch.collection_ms > 0


def test_synchronous_collector_reports_auto_resets_for_short_episodes(tmp_path) -> None:
    temporal_root = tmp_path / "temporal_autoreset"
    store = TemporalStore(temporal_root)
    manager = RLEnvironmentManager(store)
    spec = ExperimentSpec(
        num_envs=4,
        horizon=4,
        updates=1,
        max_episode_steps=2,
        temporal_root=str(temporal_root),
        collector_auto_reset=True,
    )
    collector = SynchronousCollector(manager, spec)
    learner = LocalActorCriticLearner(obs_dim=2, num_actions=manager.action_dim_for_env(spec.train_env_name), spec=spec)

    try:
        batch = collector.collect(learner, policy_version="toy-line-collector:update-0")
    finally:
        collector.close()

    assert batch.runtime_profile["auto_reset_count"] > 0


def test_local_experiment_runs_eval_and_exports_replay(tmp_path) -> None:
    replay_dir = tmp_path / "replay"
    temporal_root = tmp_path / "temporal"
    result = run_local_experiment(
        ExperimentSpec(
            seed=7,
            updates=24,
            num_envs=16,
            horizon=8,
            max_episode_steps=8,
            eval_num_envs=8,
            eval_episodes=8,
            eval_interval=6,
            replay_dir=str(replay_dir),
            temporal_root=str(temporal_root),
        )
    )

    first_return = result["metrics"][0]["mean_return"]
    assert result["best_mean_return"] > first_return
    assert result["last_evaluation"] is not None
    assert result["last_evaluation"]["success_rate"] >= 0.95
    assert result["replay_shard"] is not None

    replay_path = Path(result["replay_shard"]["uri"])
    assert replay_path.exists()

    store = TemporalStore(temporal_root)
    assert len(store.evaluation_runs.list()) >= 1
    assert len(store.replay_shards.list()) >= 1


def test_local_experiment_supports_genie_env_smoke(tmp_path) -> None:
    replay_dir = tmp_path / "replay_genie"
    temporal_root = tmp_path / "temporal_genie"
    result = run_local_experiment(
        ExperimentSpec(
            experiment_name="genie-token-smoke",
            seed=11,
            train_env_name="genie-token-grid-v0",
            train_task_id="genie-token-train",
            eval_task_id="genie-token-eval",
            updates=3,
            num_envs=4,
            horizon=2,
            max_episode_steps=3,
            eval_num_envs=2,
            eval_episodes=2,
            eval_interval=1,
            replay_dir=str(replay_dir),
            temporal_root=str(temporal_root),
        )
    )

    assert result["experiment_name"] == "genie-token-smoke"
    assert result["last_evaluation"] is not None
    assert result["replay_shard"] is not None
    assert result["metrics"][-1]["chunk_count"] >= 1.0
    assert "reward_stage_latency_ms" in result["metrics"][-1]
    assert "auto_reset_count" in result["metrics"][-1]
