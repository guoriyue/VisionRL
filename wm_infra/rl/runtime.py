"""Northbound RL environment session runtime for wm-infra."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from wm_infra.api.protocol import (
    EnvironmentSessionResponse,
    EnvironmentStepManyResponse,
    EnvironmentStepResponse,
)
from wm_infra.controlplane import (
    BranchCreate,
    CheckpointCreate,
    EnvironmentSessionCreate,
    EnvironmentSessionRecord,
    EnvironmentSpec,
    EpisodeCreate,
    StateHandleCreate,
    StateHandleKind,
    TaskSpec,
    TemporalStatus,
    TemporalStore,
    TrajectoryCreate,
    TrajectoryRecord,
    TransitionCreate,
)
from wm_infra.core.execution import (
    BatchSignature,
    ExecutionBatchPolicy,
    ExecutionChunk,
    ExecutionEntity,
    build_execution_chunks,
    chunk_fill_ratio,
    summarize_execution_chunks,
)
from wm_infra.rl.env import GoalReward
from wm_infra.rl.genie_adapter import GenieRLSpec, GenieTokenReward, GenieWorldModelAdapter
from wm_infra.rl.toy import ToyLineWorldModel, ToyLineWorldSpec


def _default_environment_specs() -> list[EnvironmentSpec]:
    return [
        EnvironmentSpec(
            env_name="toy-line-v0",
            backend="toy-line-world-model",
            observation_mode="latent_goal_concat",
            action_space={
                "type": "discrete_one_hot",
                "num_actions": 3,
                "labels": ["left", "stay", "right"],
            },
            reward_schema={
                "type": "dense_goal_mse",
                "success_threshold": 0.01,
                "reward_scale": 4.0,
            },
            default_horizon=12,
            supports_batch_step=True,
            supports_fork=True,
            metadata={
                "runtime_family": "rl_env_session",
                "world_model_contract": "predict_next",
            },
        ),
        EnvironmentSpec(
            env_name="genie-token-grid-v0",
            backend="genie-rollout",
            observation_mode="token_context_goal_concat",
            action_space={
                "type": "discrete_one_hot",
                "num_actions": 5,
                "labels": ["stay", "shift_left", "shift_right", "token_plus", "token_minus"],
            },
            reward_schema={
                "type": "token_l1_goal",
                "success_threshold": 0.01,
                "reward_scale": 4.0,
            },
            default_horizon=12,
            supports_batch_step=True,
            supports_fork=True,
            metadata={
                "runtime_family": "rl_env_session",
                "world_model_contract": "predict_next",
                "action_conditioning": "latest_prompt_token_control",
            },
        ),
    ]


def _default_task_specs() -> list[TaskSpec]:
    return [
        TaskSpec(
            task_id="toy-line-train",
            env_name="toy-line-v0",
            task_family="goal_reaching",
            goal_spec={"mode": "uniform", "low": -0.8, "high": 0.8},
            seed_policy="explicit",
            difficulty="default",
            split="train",
        ),
        TaskSpec(
            task_id="toy-line-eval",
            env_name="toy-line-v0",
            task_family="goal_reaching",
            goal_spec={"mode": "fixed", "target": 0.4},
            seed_policy="explicit",
            difficulty="default",
            split="eval",
        ),
        TaskSpec(
            task_id="genie-token-train",
            env_name="genie-token-grid-v0",
            task_family="token_goal_reaching",
            goal_spec={"mode": "seeded_random"},
            seed_policy="explicit",
            difficulty="default",
            split="train",
        ),
        TaskSpec(
            task_id="genie-token-eval",
            env_name="genie-token-grid-v0",
            task_family="token_goal_reaching",
            goal_spec={"mode": "fixed_seed", "seed": 404},
            seed_policy="explicit",
            difficulty="default",
            split="eval",
        ),
    ]


@dataclass(slots=True)
class _LiveSession:
    env_id: str
    env_name: str
    task_id: str
    episode_id: str
    branch_id: str
    trajectory_id: str
    state_handle_id: str
    checkpoint_id: str | None
    policy_version: str | None
    max_episode_steps: int
    state: torch.Tensor
    goal: torch.Tensor
    step_idx: int = 0
    needs_reset: bool = False


class RLEnvironmentManager:
    """Environment registry + live session manager for trainer-facing RL APIs."""

    def __init__(self, temporal_store: TemporalStore, *, max_chunk_size: int = 32) -> None:
        self.temporal_store = temporal_store
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.max_chunk_size = max(1, max_chunk_size)
        self.env_step_batch_policy = ExecutionBatchPolicy(
            mode="sync",
            max_chunk_size=self.max_chunk_size,
            min_ready_size=1,
            return_when_ready_count=self.max_chunk_size,
            allow_partial_batch=True,
        )
        self.world_model = ToyLineWorldModel(ToyLineWorldSpec(), device=self.device, dtype=self.dtype)
        self.genie_world_model = GenieWorldModelAdapter(device=self.device, spec=GenieRLSpec())
        self._sessions: dict[str, _LiveSession] = {}
        self._env_specs = {spec.env_name: spec for spec in _default_environment_specs()}
        self._task_specs = {task.task_id: task for task in _default_task_specs()}
        self._register_catalog()

    def _register_catalog(self) -> None:
        for spec in self._env_specs.values():
            self.temporal_store.upsert_environment_spec(spec)
        for task in self._task_specs.values():
            self.temporal_store.upsert_task_spec(task)

    def list_environment_specs(self) -> list[EnvironmentSpec]:
        return self.temporal_store.environment_specs.list()

    def list_task_specs(self, env_name: str | None = None) -> list[TaskSpec]:
        tasks = self.temporal_store.task_specs.list()
        if env_name is not None:
            tasks = [task for task in tasks if task.env_name == env_name]
        return tasks

    def action_dim_for_env(self, env_name: str) -> int:
        return self._action_dim_for_env(env_name)

    def backend_for_env(self, env_name: str) -> str:
        return self._resolve_env_spec(env_name).backend

    def get_session(self, env_id: str) -> EnvironmentSessionRecord:
        session = self.temporal_store.environment_sessions.get(env_id)
        if session is None:
            raise KeyError(env_id)
        return session

    def list_sessions(self) -> list[EnvironmentSessionRecord]:
        return self.temporal_store.environment_sessions.list()

    def create_session(
        self,
        *,
        env_name: str,
        task_id: str | None,
        seed: int | None,
        policy_version: str | None,
        max_episode_steps: int | None,
        labels: dict[str, str],
        metadata: dict[str, Any],
    ) -> EnvironmentSessionResponse:
        spec = self._resolve_env_spec(env_name)
        task = self._resolve_task(task_id or self._default_task_for_env(env_name), env_name=env_name)
        episode = self.temporal_store.create_episode(
            EpisodeCreate(
                title=f"{env_name}:{task.task_id}",
                metadata={"env_name": env_name, "task_id": task.task_id, **metadata},
            )
        )
        branch = self.temporal_store.create_branch(
            BranchCreate(
                episode_id=episode.episode_id,
                name="main",
                metadata={"env_name": env_name, "task_id": task.task_id},
            )
        )
        state, goal = self._sample_initial_state(task, seed)
        state_handle = self._persist_state_handle(
            episode_id=episode.episode_id,
            branch_id=branch.branch_id,
            state=state,
            goal=goal,
            step_idx=0,
            task=task,
            env_name=env_name,
        )
        trajectory = self.temporal_store.create_trajectory(
            TrajectoryCreate(
                env_id="pending",
                episode_id=episode.episode_id,
                task_id=task.task_id,
                policy_version=policy_version,
                metadata={"env_name": env_name},
            )
        )
        session = self.temporal_store.create_environment_session(
            EnvironmentSessionCreate(
                env_name=env_name,
                episode_id=episode.episode_id,
                task_id=task.task_id,
                backend=spec.backend,
                current_step=0,
                state_handle_id=state_handle.state_handle_id,
                checkpoint_id=None,
                trajectory_id=trajectory.trajectory_id,
                branch_id=branch.branch_id,
                policy_version=policy_version,
                labels=labels,
                metadata={"env_name": env_name, "task_split": task.split, **metadata},
            )
        )
        trajectory.env_id = session.env_id
        self.temporal_store.update_trajectory(trajectory)
        live = _LiveSession(
            env_id=session.env_id,
            env_name=env_name,
            task_id=task.task_id,
            episode_id=episode.episode_id,
            branch_id=branch.branch_id,
            trajectory_id=trajectory.trajectory_id,
            state_handle_id=state_handle.state_handle_id,
            checkpoint_id=None,
            policy_version=policy_version,
            max_episode_steps=max_episode_steps or spec.default_horizon,
            state=state,
            goal=goal,
        )
        self._sessions[session.env_id] = live
        return self._session_response(session, live)

    def reset_session(
        self,
        env_id: str,
        *,
        seed: int | None,
        policy_version: str | None,
        metadata: dict[str, Any],
    ) -> EnvironmentSessionResponse:
        session = self.get_session(env_id)
        live = self._require_live_session(env_id)
        task = self._resolve_task(session.task_id, env_name=session.env_name)
        self._finalize_trajectory(live.trajectory_id, success=False)

        episode = self.temporal_store.create_episode(
            EpisodeCreate(
                title=f"{session.env_name}:{task.task_id}",
                metadata={"env_id": env_id, "reset_from_episode_id": session.episode_id, **metadata},
            )
        )
        branch = self.temporal_store.create_branch(
            BranchCreate(
                episode_id=episode.episode_id,
                name="main",
                metadata={"env_id": env_id, "reset": True},
            )
        )
        state, goal = self._sample_initial_state(task, seed)
        state_handle = self._persist_state_handle(
            episode_id=episode.episode_id,
            branch_id=branch.branch_id,
            state=state,
            goal=goal,
            step_idx=0,
            task=task,
            env_name=session.env_name,
        )
        trajectory = self.temporal_store.create_trajectory(
            TrajectoryCreate(
                env_id=env_id,
                episode_id=episode.episode_id,
                task_id=task.task_id,
                policy_version=policy_version or live.policy_version,
                metadata={"env_name": session.env_name, "reset": True},
            )
        )
        live.state = state
        live.goal = goal
        live.step_idx = 0
        live.needs_reset = False
        live.episode_id = episode.episode_id
        live.branch_id = branch.branch_id
        live.state_handle_id = state_handle.state_handle_id
        live.trajectory_id = trajectory.trajectory_id
        live.checkpoint_id = None
        if policy_version is not None:
            live.policy_version = policy_version

        session.episode_id = episode.episode_id
        session.branch_id = branch.branch_id
        session.current_step = 0
        session.state_handle_id = state_handle.state_handle_id
        session.trajectory_id = trajectory.trajectory_id
        session.checkpoint_id = None
        session.policy_version = live.policy_version
        session.completed_at = None
        session.status = TemporalStatus.ACTIVE
        session.metadata["needs_reset"] = False
        session.metadata.update(metadata)
        session = self.temporal_store.update_environment_session(session)
        return self._session_response(session, live)

    def step_session(
        self,
        env_id: str,
        *,
        action: list[float],
        policy_version: str | None,
        checkpoint: bool,
        metadata: dict[str, Any],
    ) -> EnvironmentStepResponse:
        batch_response = self.step_many(
            env_id,
            env_ids=[],
            actions=[action],
            policy_version=policy_version,
            checkpoint=checkpoint,
            metadata=metadata,
        )
        return batch_response.results[0]

    def step_many(
        self,
        env_id: str,
        *,
        env_ids: list[str],
        actions: list[list[float]],
        policy_version: str | None,
        checkpoint: bool,
        metadata: dict[str, Any],
    ) -> EnvironmentStepManyResponse:
        ordered_env_ids = [env_id, *[item for item in env_ids if item != env_id]]
        if len(ordered_env_ids) != len(actions):
            raise ValueError("step_many requires one action per env_id")

        sessions = [self._require_live_session(item) for item in ordered_env_ids]
        if any(session.needs_reset for session in sessions):
            raise ValueError("All sessions in step_many must be reset before further stepping")
        env_name = sessions[0].env_name
        if any(session.env_name != env_name for session in sessions):
            raise ValueError("step_many currently only supports batching the same env_name")

        action_tensor = torch.as_tensor(actions, dtype=self.dtype, device=self.device).view(len(actions), -1)
        expected_action_dim = self._action_dim_for_env(env_name)
        if action_tensor.shape[1] != expected_action_dim:
            raise ValueError(f"Expected action_dim={expected_action_dim}, got {action_tensor.shape[1]}")

        responses: list[EnvironmentStepResponse] = []
        reward_stage_ms = 0.0
        trajectory_persist_ms = 0.0
        chunk_history: list[dict[str, Any]] = []
        step_chunks = self._build_step_chunks(ordered_env_ids, sessions, action_tensor)
        chunk_summary = summarize_execution_chunks(step_chunks, policy=self.env_step_batch_policy)
        reward_fn = self._reward_fn_for_env(env_name)
        world_model = self._world_model_for_env(env_name)

        for chunk in step_chunks:
            chunk_history.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "chunk_size": chunk.size,
                    "signature": asdict(chunk.signature),
                    "fill_ratio": chunk_fill_ratio(chunk.size, self.env_step_batch_policy.max_chunk_size),
                }
            )

            reward_started_at = time.perf_counter()
            next_states = world_model.predict_next(chunk.latent_batch, chunk.action_batch)
            goal_batch = torch.cat([self._require_live_session(entity.rollout_id).goal for entity in chunk.entities], dim=0)
            rewards, terminated, info_tensors = reward_fn.evaluate(next_states, goal_batch)
            reward_stage_ms += (time.perf_counter() - reward_started_at) * 1000.0

            persist_started_at = time.perf_counter()
            for index, entity in enumerate(chunk.entities):
                session_id = entity.rollout_id
                live = self._require_live_session(session_id)
                session = self.get_session(session_id)
                task = self._resolve_task(live.task_id, env_name=live.env_name)
                step_idx = live.step_idx
                reward = float(rewards[index].item())
                is_terminated = bool(terminated[index].item())
                next_step_idx = step_idx + 1
                is_truncated = next_step_idx >= live.max_episode_steps
                next_state = next_states[index:index + 1]
                next_handle = self._persist_state_handle(
                    episode_id=live.episode_id,
                    branch_id=live.branch_id,
                    state=next_state,
                    goal=live.goal,
                    step_idx=next_step_idx,
                    task=task,
                    env_name=live.env_name,
                )
                info = self._transition_info_for_env(
                    env_name,
                    goal=live.goal,
                    info_tensors=info_tensors,
                    index=index,
                    step_idx=next_step_idx,
                )
                transition = self.temporal_store.create_transition(
                    TransitionCreate(
                        env_id=session_id,
                        episode_id=live.episode_id,
                        trajectory_id=live.trajectory_id,
                        task_id=live.task_id,
                        step_idx=step_idx,
                        observation_ref=live.state_handle_id,
                        action=list(chunk.action_batch[index].detach().cpu().numpy().tolist()),
                        reward=reward,
                        terminated=is_terminated,
                        truncated=is_truncated,
                        next_observation_ref=next_handle.state_handle_id,
                        info={**info, **metadata},
                        policy_version=policy_version or live.policy_version,
                        metadata={"batched": len(sessions) > 1, "chunk_id": chunk.chunk_id},
                    )
                )
                checkpoint_id = None
                live.state = next_state
                live.state_handle_id = next_handle.state_handle_id
                live.step_idx = next_step_idx
                if policy_version is not None:
                    live.policy_version = policy_version
                if checkpoint or is_terminated or is_truncated:
                    checkpoint_id = self._checkpoint_live_session(
                        live,
                        tag=f"step-{next_step_idx}",
                        metadata={"batched": len(sessions) > 1, "chunk_id": chunk.chunk_id, **metadata},
                    )
                trajectory = self.temporal_store.trajectories.get(live.trajectory_id)
                assert trajectory is not None
                trajectory.num_steps += 1
                trajectory.return_value += reward
                trajectory.success = trajectory.success or bool(info["success"])
                trajectory.transition_refs.append(transition.transition_id)
                if is_terminated:
                    trajectory.status = TemporalStatus.SUCCEEDED
                    trajectory.completed_at = time.time()
                elif is_truncated:
                    trajectory.status = TemporalStatus.ARCHIVED
                    trajectory.completed_at = time.time()
                self.temporal_store.update_trajectory(trajectory)

                session.current_step = next_step_idx
                session.state_handle_id = next_handle.state_handle_id
                session.checkpoint_id = checkpoint_id
                session.policy_version = live.policy_version
                session.metadata["needs_reset"] = is_terminated or is_truncated
                session.metadata["last_transition_id"] = transition.transition_id
                session.metadata["last_chunk_id"] = chunk.chunk_id
                if is_terminated or is_truncated:
                    live.needs_reset = True
                session = self.temporal_store.update_environment_session(session)
                responses.append(
                    EnvironmentStepResponse(
                        env_id=session_id,
                        episode_id=live.episode_id,
                        task_id=live.task_id,
                        trajectory_id=live.trajectory_id,
                        state_handle_id=next_handle.state_handle_id,
                        checkpoint_id=checkpoint_id,
                        transition_id=transition.transition_id,
                        policy_version=live.policy_version,
                        step_idx=next_step_idx,
                        observation=self._observation(next_state, live.goal),
                        reward=reward,
                        terminated=is_terminated,
                        truncated=is_truncated,
                        info=info,
                    )
                )
            trajectory_persist_ms += (time.perf_counter() - persist_started_at) * 1000.0
        return EnvironmentStepManyResponse(
            env_ids=ordered_env_ids,
            results=responses,
            runtime={
                "execution_path": "chunked_env_step",
                "env_step_chunk_total": len(step_chunks),
                "reward_stage_ms": reward_stage_ms,
                "trajectory_persist_ms": trajectory_persist_ms,
                "state_locality_hit_rate": 1.0,
                "state_locality_mode": "in_process_live_session",
                "step_semantics": "sync_step_many",
                "northbound_reset_policy": "explicit_reset_required",
                **chunk_summary,
                "chunks": chunk_history,
            },
        )

    def fork_session(
        self,
        env_id: str,
        *,
        branch_name: str | None,
        policy_version: str | None,
        metadata: dict[str, Any],
    ) -> EnvironmentSessionResponse:
        source = self._require_live_session(env_id)
        source_record = self.get_session(env_id)
        branch = self.temporal_store.create_branch(
            BranchCreate(
                episode_id=source.episode_id,
                parent_branch_id=source.branch_id,
                forked_from_checkpoint_id=source.checkpoint_id,
                name=branch_name or f"fork-{source.step_idx}",
                metadata={"source_env_id": env_id, **metadata},
            )
        )
        state_handle = self._persist_state_handle(
            episode_id=source.episode_id,
            branch_id=branch.branch_id,
            state=source.state.clone(),
            goal=source.goal.clone(),
            step_idx=source.step_idx,
            task=self._resolve_task(source.task_id, env_name=source.env_name),
            env_name=source.env_name,
        )
        trajectory = self.temporal_store.create_trajectory(
            TrajectoryCreate(
                env_id="pending",
                episode_id=source.episode_id,
                task_id=source.task_id,
                policy_version=policy_version or source.policy_version,
                metadata={"forked_from_env_id": env_id},
            )
        )
        session = self.temporal_store.create_environment_session(
            EnvironmentSessionCreate(
                env_name=source.env_name,
                episode_id=source.episode_id,
                task_id=source.task_id,
                backend=source_record.backend,
                current_step=source.step_idx,
                state_handle_id=state_handle.state_handle_id,
                checkpoint_id=source.checkpoint_id,
                trajectory_id=trajectory.trajectory_id,
                branch_id=branch.branch_id,
                policy_version=policy_version or source.policy_version,
                labels=dict(source_record.labels),
                metadata={"forked_from_env_id": env_id, **metadata},
            )
        )
        trajectory.env_id = session.env_id
        self.temporal_store.update_trajectory(trajectory)
        live = _LiveSession(
            env_id=session.env_id,
            env_name=source.env_name,
            task_id=source.task_id,
            episode_id=source.episode_id,
            branch_id=branch.branch_id,
            trajectory_id=trajectory.trajectory_id,
            state_handle_id=state_handle.state_handle_id,
            checkpoint_id=source.checkpoint_id,
            policy_version=policy_version or source.policy_version,
            max_episode_steps=source.max_episode_steps,
            state=source.state.clone(),
            goal=source.goal.clone(),
            step_idx=source.step_idx,
            needs_reset=source.needs_reset,
        )
        self._sessions[session.env_id] = live
        return self._session_response(session, live)

    def checkpoint_session(self, env_id: str, *, tag: str | None, metadata: dict[str, Any]) -> str:
        live = self._require_live_session(env_id)
        return self._checkpoint_live_session(live, tag=tag or f"step-{live.step_idx}", metadata=metadata)

    def delete_session(self, env_id: str) -> None:
        session = self.get_session(env_id)
        live = self._sessions.pop(env_id, None)
        if live is not None:
            self._finalize_trajectory(live.trajectory_id, success=False)
        session.status = TemporalStatus.ARCHIVED
        session.completed_at = time.time()
        self.temporal_store.update_environment_session(session)

    def list_transitions(self, env_id: str | None = None, trajectory_id: str | None = None) -> list[Any]:
        items = self.temporal_store.transitions.list()
        if env_id is not None:
            items = [item for item in items if item.env_id == env_id]
        if trajectory_id is not None:
            items = [item for item in items if item.trajectory_id == trajectory_id]
        return items

    def list_trajectories(self, env_id: str | None = None, episode_id: str | None = None) -> list[TrajectoryRecord]:
        items = self.temporal_store.trajectories.list()
        if env_id is not None:
            items = [item for item in items if item.env_id == env_id]
        if episode_id is not None:
            items = [item for item in items if item.episode_id == episode_id]
        return items

    def list_evaluation_runs(self) -> list[Any]:
        return self.temporal_store.evaluation_runs.list()

    def _session_response(self, session: EnvironmentSessionRecord, live: _LiveSession) -> EnvironmentSessionResponse:
        return EnvironmentSessionResponse(
            env_id=session.env_id,
            env_name=session.env_name,
            episode_id=session.episode_id,
            task_id=session.task_id,
            branch_id=session.branch_id,
            state_handle_id=session.state_handle_id,
            checkpoint_id=session.checkpoint_id,
            trajectory_id=session.trajectory_id,
            current_step=session.current_step,
            policy_version=session.policy_version,
            status=session.status.value,
            observation=self._observation(live.state, live.goal),
            info=self._session_info_for_env(session.env_name, live.goal, live.step_idx),
        )

    def _sample_initial_state(self, task: TaskSpec, seed: int | None) -> tuple[torch.Tensor, torch.Tensor]:
        if task.env_name == "genie-token-grid-v0":
            state = self.genie_world_model.sample_initial_state(seed=seed).to(self.device, self.dtype)
            goal_mode = task.goal_spec.get("mode", "seeded_random")
            goal_seed = seed
            if goal_mode == "fixed_seed":
                goal_seed = int(task.goal_spec.get("seed", 404))
            goal = self.genie_world_model.sample_goal_state(seed=goal_seed).to(self.device, self.dtype)
            return state, goal

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
        state = torch.zeros(1, 1, 1, device=self.device, dtype=self.dtype)
        goal_mode = task.goal_spec.get("mode", "fixed")
        if goal_mode == "uniform":
            low = float(task.goal_spec.get("low", -0.8))
            high = float(task.goal_spec.get("high", 0.8))
            goal = torch.empty(1, 1, 1, device=self.device, dtype=self.dtype)
            goal.uniform_(low, high, generator=generator)
        else:
            target = float(task.goal_spec.get("target", 0.4))
            goal = torch.full((1, 1, 1), target, device=self.device, dtype=self.dtype)
        return state, goal

    def _persist_state_handle(
        self,
        *,
        episode_id: str,
        branch_id: str,
        state: torch.Tensor,
        goal: torch.Tensor,
        step_idx: int,
        task: TaskSpec,
        env_name: str,
    ):
        observation = self._observation(state, goal)
        return self.temporal_store.create_state_handle(
            StateHandleCreate(
                episode_id=episode_id,
                branch_id=branch_id,
                kind=StateHandleKind.LATENT,
                shape=list(state.shape[1:]),
                dtype=str(state.dtype).replace("torch.", ""),
                metadata={
                    "env_name": env_name,
                    "task_id": task.task_id,
                    "step_idx": step_idx,
                    "latent_state": state.squeeze(0).detach().cpu().numpy().tolist(),
                    "goal_state": goal.squeeze(0).detach().cpu().numpy().tolist(),
                    "observation": observation,
                },
            )
        )

    def _checkpoint_live_session(self, live: _LiveSession, *, tag: str, metadata: dict[str, Any]) -> str:
        checkpoint = self.temporal_store.create_checkpoint(
            CheckpointCreate(
                episode_id=live.episode_id,
                branch_id=live.branch_id,
                state_handle_id=live.state_handle_id,
                step_index=live.step_idx,
                tag=tag,
                metadata={"env_name": live.env_name, "task_id": live.task_id, **metadata},
            )
        )
        live.checkpoint_id = checkpoint.checkpoint_id
        session = self.get_session(live.env_id)
        session.checkpoint_id = checkpoint.checkpoint_id
        self.temporal_store.update_environment_session(session)
        state_handle = self.temporal_store.state_handles.get(live.state_handle_id)
        if state_handle is not None:
            state_handle.checkpoint_id = checkpoint.checkpoint_id
            self.temporal_store.state_handles.put(state_handle)
        return checkpoint.checkpoint_id

    def _finalize_trajectory(self, trajectory_id: str, *, success: bool) -> None:
        trajectory = self.temporal_store.trajectories.get(trajectory_id)
        if trajectory is None or trajectory.completed_at is not None:
            return
        trajectory.success = trajectory.success or success
        trajectory.status = TemporalStatus.SUCCEEDED if trajectory.success else TemporalStatus.ARCHIVED
        trajectory.completed_at = time.time()
        self.temporal_store.update_trajectory(trajectory)

    def _require_live_session(self, env_id: str) -> _LiveSession:
        session = self._sessions.get(env_id)
        if session is None:
            raise KeyError(env_id)
        return session

    def _world_model_for_env(self, env_name: str):
        if env_name == "genie-token-grid-v0":
            return self.genie_world_model
        return self.world_model

    def _reward_fn_for_env(self, env_name: str):
        reward_schema = self._env_specs[env_name].reward_schema
        if env_name == "genie-token-grid-v0":
            return GenieTokenReward(
                self.genie_world_model.spec,
                success_threshold=float(reward_schema.get("success_threshold", 0.01)),
                reward_scale=float(reward_schema.get("reward_scale", 4.0)),
            )
        return GoalReward(
            success_threshold=float(reward_schema.get("success_threshold", 0.01)),
            reward_scale=float(reward_schema.get("reward_scale", 4.0)),
        )

    def _action_dim_for_env(self, env_name: str) -> int:
        if env_name == "genie-token-grid-v0":
            return self.genie_world_model.spec.action_dim
        return self.world_model.spec.action_dim

    def _session_info_for_env(self, env_name: str, goal: torch.Tensor, step_idx: int) -> dict[str, Any]:
        info: dict[str, Any] = {"step": step_idx}
        info["goal"] = goal.squeeze(0).detach().cpu().numpy().tolist()
        if env_name == "genie-token-grid-v0":
            info["goal_token_grid"] = goal[:, -self.genie_world_model.spec.frame_token_count :, :].reshape(
                1,
                self.genie_world_model.spec.spatial_h,
                self.genie_world_model.spec.spatial_w,
            ).squeeze(0).detach().cpu().numpy().tolist()
        return info

    def _transition_info_for_env(
        self,
        env_name: str,
        *,
        goal: torch.Tensor,
        info_tensors: dict[str, torch.Tensor],
        index: int,
        step_idx: int,
    ) -> dict[str, Any]:
        info = self._session_info_for_env(env_name, goal, step_idx)
        if env_name == "genie-token-grid-v0":
            info["token_l1"] = float(info_tensors["token_l1"][index].item())
        else:
            info["goal_mse"] = float(info_tensors["goal_mse"][index].item())
        info["success"] = bool(info_tensors["success"][index].item() > 0)
        return info

    def _build_step_chunks(
        self,
        env_ids: list[str],
        sessions: list[_LiveSession],
        action_tensor: torch.Tensor,
    ) -> list[ExecutionChunk]:
        if not env_ids:
            return []
        signature = BatchSignature(
            stage="env_step",
            latent_shape=tuple(sessions[0].state.shape[-2:]),
            action_dim=int(action_tensor.shape[-1]),
            dtype=str(self.dtype).replace("torch.", ""),
            device=str(self.device),
            needs_decode=False,
        )
        entities = [
            ExecutionEntity(
                entity_id=f"{env_id}:env_step:{session.step_idx}",
                rollout_id=env_id,
                stage="env_step",
                step_idx=session.step_idx,
                batch_signature=signature,
            )
            for env_id, session in zip(env_ids, sessions, strict=True)
        ]
        latent_items = [session.state for session in sessions]
        action_items = [action_tensor[index:index + 1] for index in range(action_tensor.shape[0])]
        return build_execution_chunks(
            signature=signature,
            entities=entities,
            latent_items=latent_items,
            action_items=action_items,
            policy=self.env_step_batch_policy,
            chunk_id_prefix="env_step",
            latent_join=lambda items: torch.cat(items, dim=0),
            action_join=lambda items: torch.cat(items, dim=0),
        )

    def _resolve_env_spec(self, env_name: str) -> EnvironmentSpec:
        spec = self.temporal_store.environment_specs.get(env_name)
        if spec is None:
            raise KeyError(env_name)
        return spec

    def _resolve_task(self, task_id: str, *, env_name: str) -> TaskSpec:
        task = self.temporal_store.task_specs.get(task_id)
        if task is None or task.env_name != env_name:
            raise KeyError(task_id)
        return task

    def _default_task_for_env(self, env_name: str) -> str:
        for task in self.temporal_store.task_specs.list():
            if task.env_name == env_name and task.split == "train":
                return task.task_id
        raise KeyError(env_name)

    @staticmethod
    def _observation(state: torch.Tensor, goal: torch.Tensor) -> list[list[float]]:
        return torch.cat([state, goal], dim=-1).squeeze(0).detach().cpu().numpy().tolist()
