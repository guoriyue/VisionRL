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
    TransitionContextResponse,
    TransitionPredictManyResponse,
    TransitionPredictResponse,
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
class _StatelessTransitionContext:
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
    step_idx: int
    scope_id: str


class RLEnvironmentManager:
    """Environment registry + stateless transition manager for trainer-facing RL APIs."""

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
        initialized = self.initialize_transition_context(
            env_name=env_name,
            task_id=task_id,
            seed=seed,
            policy_version=policy_version,
            max_episode_steps=max_episode_steps,
            branch_name="main",
            labels=labels,
            metadata=metadata,
        )
        session = self.temporal_store.create_environment_session(
            EnvironmentSessionCreate(
                env_name=env_name,
                episode_id=initialized.episode_id,
                task_id=initialized.task_id,
                backend=spec.backend,
                current_step=0,
                state_handle_id=initialized.state_handle_id,
                checkpoint_id=None,
                trajectory_id=initialized.trajectory_id,
                branch_id=initialized.branch_id,
                policy_version=initialized.policy_version,
                labels=labels,
                metadata={
                    "env_name": env_name,
                    "task_split": self._resolve_task(initialized.task_id, env_name=env_name).split,
                    "max_episode_steps": initialized.max_episode_steps,
                    "needs_reset": False,
                    "compat_session": True,
                    **metadata,
                },
            )
        )
        trajectory = self.temporal_store.trajectories.get(initialized.trajectory_id)
        assert trajectory is not None
        trajectory.env_id = session.env_id
        self.temporal_store.update_trajectory(trajectory)
        return self._session_response(session)

    def initialize_transition_context(
        self,
        *,
        env_name: str,
        task_id: str | None,
        seed: int | None,
        policy_version: str | None,
        max_episode_steps: int | None,
        branch_name: str | None,
        labels: dict[str, str],
        metadata: dict[str, Any],
    ) -> TransitionContextResponse:
        spec = self._resolve_env_spec(env_name)
        task = self._resolve_task(task_id or self._default_task_for_env(env_name), env_name=env_name)
        episode = self.temporal_store.create_episode(
            EpisodeCreate(
                title=f"{env_name}:{task.task_id}",
                labels=labels,
                metadata={"env_name": env_name, "task_id": task.task_id, "stateless": True, **metadata},
            )
        )
        branch = self.temporal_store.create_branch(
            BranchCreate(
                episode_id=episode.episode_id,
                name=branch_name or "main",
                labels=labels,
                metadata={"env_name": env_name, "task_id": task.task_id, "stateless": True, **metadata},
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
        trajectory = self._ensure_stateless_trajectory(
            env_name=env_name,
            task=task,
            episode_id=episode.episode_id,
            branch_id=branch.branch_id,
            trajectory_id=None,
            policy_version=policy_version,
            max_episode_steps=max_episode_steps or spec.default_horizon,
            metadata={"seed": seed, **metadata},
        )
        return TransitionContextResponse(
            env_name=env_name,
            episode_id=episode.episode_id,
            task_id=task.task_id,
            branch_id=branch.branch_id,
            state_handle_id=state_handle.state_handle_id,
            checkpoint_id=None,
            trajectory_id=trajectory.trajectory_id,
            current_step=0,
            policy_version=policy_version,
            max_episode_steps=int(trajectory.metadata.get("max_episode_steps", spec.default_horizon)),
            observation=self._observation(state, goal),
            info=self._session_info_for_env(env_name, goal, 0),
        )

    def reset_session(
        self,
        env_id: str,
        *,
        seed: int | None,
        policy_version: str | None,
        metadata: dict[str, Any],
    ) -> EnvironmentSessionResponse:
        session = self.get_session(env_id)
        task = self._resolve_task(session.task_id, env_name=session.env_name)
        if session.trajectory_id:
            self._finalize_trajectory(session.trajectory_id, success=False)

        initialized = self.initialize_transition_context(
            env_name=session.env_name,
            task_id=session.task_id,
            seed=seed,
            policy_version=policy_version or session.policy_version,
            max_episode_steps=int(session.metadata.get("max_episode_steps", self._resolve_env_spec(session.env_name).default_horizon)),
            branch_name="main",
            labels=dict(session.labels),
            metadata={"env_id": env_id, "reset_from_episode_id": session.episode_id, **metadata},
        )
        trajectory = self.temporal_store.trajectories.get(initialized.trajectory_id)
        assert trajectory is not None
        trajectory.env_id = env_id
        self.temporal_store.update_trajectory(trajectory)

        session.episode_id = initialized.episode_id
        session.branch_id = initialized.branch_id
        session.current_step = 0
        session.state_handle_id = initialized.state_handle_id
        session.trajectory_id = initialized.trajectory_id
        session.checkpoint_id = None
        session.policy_version = initialized.policy_version
        session.completed_at = None
        session.status = TemporalStatus.ACTIVE
        session.metadata["needs_reset"] = False
        session.metadata["max_episode_steps"] = initialized.max_episode_steps
        session.metadata["task_split"] = task.split
        session.metadata.update(metadata)
        session = self.temporal_store.update_environment_session(session)
        return self._session_response(session)

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

        session_records = [self.get_session(item) for item in ordered_env_ids]
        if any(bool(session.metadata.get("needs_reset")) for session in session_records):
            raise ValueError("All sessions in step_many must be reset before further stepping")
        env_name = session_records[0].env_name
        if any(session.env_name != env_name for session in session_records):
            raise ValueError("step_many currently only supports batching the same env_name")

        action_tensor = torch.as_tensor(actions, dtype=self.dtype, device=self.device).view(len(actions), -1)
        expected_action_dim = self._action_dim_for_env(env_name)
        if action_tensor.shape[1] != expected_action_dim:
            raise ValueError(f"Expected action_dim={expected_action_dim}, got {action_tensor.shape[1]}")

        prediction = self.predict_many_transitions(
            items=[
                {
                    "state_handle_id": session.state_handle_id,
                    "trajectory_id": session.trajectory_id,
                    "action": action,
                    "max_episode_steps": int(session.metadata.get("max_episode_steps", self._resolve_env_spec(session.env_name).default_horizon)),
                }
                for session, action in zip(session_records, actions, strict=True)
            ],
            policy_version=policy_version,
            checkpoint=checkpoint,
            metadata=metadata,
        )

        responses: list[EnvironmentStepResponse] = []
        chunk_ids = [chunk["chunk_id"] for chunk in prediction.runtime.get("chunks", [])]
        for index, (session, result) in enumerate(zip(session_records, prediction.results, strict=True)):
            session.current_step = result.step_idx
            session.state_handle_id = result.state_handle_id
            session.checkpoint_id = result.checkpoint_id
            session.policy_version = result.policy_version
            session.metadata["needs_reset"] = result.terminated or result.truncated
            session.metadata["last_transition_id"] = result.transition_id
            if chunk_ids:
                session.metadata["last_chunk_id"] = chunk_ids[min(index // self.max_chunk_size, len(chunk_ids) - 1)]
            session = self.temporal_store.update_environment_session(session)
            responses.append(
                EnvironmentStepResponse(
                    env_id=session.env_id,
                    episode_id=result.episode_id,
                    task_id=result.task_id,
                    trajectory_id=result.trajectory_id,
                    state_handle_id=result.state_handle_id,
                    checkpoint_id=result.checkpoint_id,
                    transition_id=result.transition_id,
                    policy_version=result.policy_version,
                    step_idx=result.step_idx,
                    observation=result.observation,
                    reward=result.reward,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    info=result.info,
                )
            )

        runtime = dict(prediction.runtime)
        runtime["execution_path"] = "chunked_env_step"
        runtime["env_step_chunk_total"] = runtime.get("chunk_count", 0)
        runtime["state_locality_mode"] = "legacy_env_wrapper"
        runtime["step_semantics"] = "sync_step_many"
        runtime["northbound_reset_policy"] = "explicit_reset_required"
        return EnvironmentStepManyResponse(
            env_ids=ordered_env_ids,
            results=responses,
            runtime=runtime,
        )

    def predict_transition(
        self,
        *,
        state_handle_id: str,
        action: list[float],
        trajectory_id: str | None,
        policy_version: str | None,
        checkpoint: bool,
        max_episode_steps: int | None,
        metadata: dict[str, Any],
    ) -> TransitionPredictResponse:
        response = self.predict_many_transitions(
            items=[{
                "state_handle_id": state_handle_id,
                "action": action,
                "trajectory_id": trajectory_id,
                "max_episode_steps": max_episode_steps,
            }],
            policy_version=policy_version,
            checkpoint=checkpoint,
            metadata=metadata,
        )
        return response.results[0]

    def predict_many_transitions(
        self,
        *,
        items: list[dict[str, Any]],
        policy_version: str | None,
        checkpoint: bool,
        metadata: dict[str, Any],
    ) -> TransitionPredictManyResponse:
        if not items:
            raise ValueError("predict_many requires at least one item")

        contexts = [
            self._load_stateless_context(
                state_handle_id=str(item["state_handle_id"]),
                trajectory_id=item.get("trajectory_id"),
                max_episode_steps=item.get("max_episode_steps"),
                policy_version=policy_version,
            )
            for item in items
        ]
        env_name = contexts[0].env_name
        if any(context.env_name != env_name for context in contexts):
            raise ValueError("predict_many currently only supports batching the same env_name")

        actions = [item["action"] for item in items]
        action_tensor = torch.as_tensor(actions, dtype=self.dtype, device=self.device).view(len(actions), -1)
        expected_action_dim = self._action_dim_for_env(env_name)
        if action_tensor.shape[1] != expected_action_dim:
            raise ValueError(f"Expected action_dim={expected_action_dim}, got {action_tensor.shape[1]}")

        reward_stage_ms = 0.0
        trajectory_persist_ms = 0.0
        chunk_sizes: list[int] = []
        chunk_history: list[dict[str, Any]] = []
        responses: list[TransitionPredictResponse] = []
        step_chunks = self._build_stateless_step_chunks(contexts, action_tensor)
        chunk_summary = summarize_execution_chunks(step_chunks, policy=self.env_step_batch_policy)
        context_by_id = {context.state_handle_id: context for context in contexts}
        reward_fn = self._reward_fn_for_env(env_name)
        world_model = self._world_model_for_env(env_name)

        for chunk in step_chunks:
            chunk_sizes.append(chunk.size)
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
            chunk_contexts = [context_by_id[entity.rollout_id] for entity in chunk.entities]
            goal_batch = torch.cat([context.goal for context in chunk_contexts], dim=0)
            rewards, terminated, info_tensors = reward_fn.evaluate(next_states, goal_batch)
            reward_stage_ms += (time.perf_counter() - reward_started_at) * 1000.0

            persist_started_at = time.perf_counter()
            for index, context in enumerate(chunk_contexts):
                task = self._resolve_task(context.task_id, env_name=context.env_name)
                reward = float(rewards[index].item())
                is_terminated = bool(terminated[index].item())
                next_step_idx = context.step_idx + 1
                is_truncated = next_step_idx >= context.max_episode_steps
                next_state = next_states[index:index + 1]
                next_handle = self._persist_state_handle(
                    episode_id=context.episode_id,
                    branch_id=context.branch_id,
                    state=next_state,
                    goal=context.goal,
                    step_idx=next_step_idx,
                    task=task,
                    env_name=context.env_name,
                )
                info = self._transition_info_for_env(
                    context.env_name,
                    goal=context.goal,
                    info_tensors=info_tensors,
                    index=index,
                    step_idx=next_step_idx,
                )
                transition = self.temporal_store.create_transition(
                    TransitionCreate(
                        env_id=context.scope_id,
                        episode_id=context.episode_id,
                        trajectory_id=context.trajectory_id,
                        task_id=context.task_id,
                        step_idx=context.step_idx,
                        observation_ref=context.state_handle_id,
                        action=list(chunk.action_batch[index].detach().cpu().numpy().tolist()),
                        reward=reward,
                        terminated=is_terminated,
                        truncated=is_truncated,
                        next_observation_ref=next_handle.state_handle_id,
                        info={**info, **metadata},
                        policy_version=policy_version or context.policy_version,
                        metadata={"batched": len(contexts) > 1, "chunk_id": chunk.chunk_id, "stateless": True},
                    )
                )
                checkpoint_id = None
                if checkpoint or is_terminated or is_truncated:
                    checkpoint_id = self._checkpoint_state_handle(
                        state_handle_id=next_handle.state_handle_id,
                        episode_id=context.episode_id,
                        branch_id=context.branch_id,
                        step_idx=next_step_idx,
                        tag=f"step-{next_step_idx}",
                        metadata={
                            "env_name": context.env_name,
                            "task_id": context.task_id,
                            "trajectory_id": context.trajectory_id,
                            "batched": len(contexts) > 1,
                            "chunk_id": chunk.chunk_id,
                            "stateless": True,
                            **metadata,
                        },
                    )
                trajectory = self.temporal_store.trajectories.get(context.trajectory_id)
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
                responses.append(
                    TransitionPredictResponse(
                        env_name=context.env_name,
                        episode_id=context.episode_id,
                        task_id=context.task_id,
                        branch_id=context.branch_id,
                        trajectory_id=context.trajectory_id,
                        state_handle_id=next_handle.state_handle_id,
                        checkpoint_id=checkpoint_id,
                        transition_id=transition.transition_id,
                        policy_version=policy_version or context.policy_version,
                        step_idx=next_step_idx,
                        max_episode_steps=context.max_episode_steps,
                        observation=self._observation(next_state, context.goal),
                        reward=reward,
                        terminated=is_terminated,
                        truncated=is_truncated,
                        info=info,
                    )
                )
            trajectory_persist_ms += (time.perf_counter() - persist_started_at) * 1000.0

        return TransitionPredictManyResponse(
            results=responses,
            runtime={
                "execution_path": "chunked_stateless_transition",
                "chunk_count": len(step_chunks),
                "chunk_sizes": chunk_sizes,
                "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
                "avg_chunk_size": (sum(chunk_sizes) / len(chunk_sizes)) if chunk_sizes else 0.0,
                "reward_stage_ms": reward_stage_ms,
                "trajectory_persist_ms": trajectory_persist_ms,
                "state_locality_hit_rate": 0.0,
                "state_locality_mode": "explicit_state_handle",
                "step_semantics": "explicit_state_transition",
                "northbound_reset_policy": "resource_reference_required",
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
        source_record = self.get_session(env_id)
        context = self._load_stateless_context(
            state_handle_id=source_record.state_handle_id,
            trajectory_id=source_record.trajectory_id,
            max_episode_steps=int(source_record.metadata.get("max_episode_steps", self._resolve_env_spec(source_record.env_name).default_horizon)),
            policy_version=policy_version or source_record.policy_version,
        )
        branch = self.temporal_store.create_branch(
            BranchCreate(
                episode_id=context.episode_id,
                parent_branch_id=context.branch_id,
                forked_from_checkpoint_id=source_record.checkpoint_id,
                name=branch_name or f"fork-{context.step_idx}",
                metadata={"source_env_id": env_id, **metadata},
            )
        )
        state_handle = self._persist_state_handle(
            episode_id=context.episode_id,
            branch_id=branch.branch_id,
            state=context.state.clone(),
            goal=context.goal.clone(),
            step_idx=context.step_idx,
            task=self._resolve_task(context.task_id, env_name=context.env_name),
            env_name=context.env_name,
        )
        trajectory = self._ensure_stateless_trajectory(
            env_name=context.env_name,
            task=self._resolve_task(context.task_id, env_name=context.env_name),
            episode_id=context.episode_id,
            branch_id=branch.branch_id,
            trajectory_id=None,
            policy_version=policy_version or context.policy_version,
            max_episode_steps=context.max_episode_steps,
            metadata={"forked_from_env_id": env_id},
        )
        session = self.temporal_store.create_environment_session(
            EnvironmentSessionCreate(
                env_name=context.env_name,
                episode_id=context.episode_id,
                task_id=context.task_id,
                backend=source_record.backend,
                current_step=context.step_idx,
                state_handle_id=state_handle.state_handle_id,
                checkpoint_id=source_record.checkpoint_id,
                trajectory_id=trajectory.trajectory_id,
                branch_id=branch.branch_id,
                policy_version=policy_version or context.policy_version,
                labels=dict(source_record.labels),
                metadata={
                    "forked_from_env_id": env_id,
                    "max_episode_steps": context.max_episode_steps,
                    "needs_reset": False,
                    "compat_session": True,
                    **metadata,
                },
            )
        )
        trajectory.env_id = session.env_id
        self.temporal_store.update_trajectory(trajectory)
        return self._session_response(session)

    def checkpoint_session(self, env_id: str, *, tag: str | None, metadata: dict[str, Any]) -> str:
        session = self.get_session(env_id)
        return self._checkpoint_state_handle(
            state_handle_id=session.state_handle_id,
            episode_id=session.episode_id,
            branch_id=session.branch_id or "",
            step_idx=session.current_step,
            tag=tag or f"step-{session.current_step}",
            metadata={"env_name": session.env_name, "task_id": session.task_id, **metadata},
        )

    def delete_session(self, env_id: str) -> None:
        session = self.get_session(env_id)
        if session.trajectory_id:
            self._finalize_trajectory(session.trajectory_id, success=False)
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

    def _session_response(self, session: EnvironmentSessionRecord) -> EnvironmentSessionResponse:
        state, goal, _step_idx = self._load_state_goal_from_handle(session.state_handle_id)
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
            observation=self._observation(state, goal),
            info=self._session_info_for_env(session.env_name, goal, session.current_step),
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

    def _finalize_trajectory(self, trajectory_id: str, *, success: bool) -> None:
        trajectory = self.temporal_store.trajectories.get(trajectory_id)
        if trajectory is None or trajectory.completed_at is not None:
            return
        trajectory.success = trajectory.success or success
        trajectory.status = TemporalStatus.SUCCEEDED if trajectory.success else TemporalStatus.ARCHIVED
        trajectory.completed_at = time.time()
        self.temporal_store.update_trajectory(trajectory)

    def _checkpoint_state_handle(
        self,
        *,
        state_handle_id: str,
        episode_id: str,
        branch_id: str | None,
        step_idx: int,
        tag: str,
        metadata: dict[str, Any],
    ) -> str:
        checkpoint = self.temporal_store.create_checkpoint(
            CheckpointCreate(
                episode_id=episode_id,
                branch_id=branch_id,
                state_handle_id=state_handle_id,
                step_index=step_idx,
                tag=tag,
                metadata=metadata,
            )
        )
        state_handle = self.temporal_store.state_handles.get(state_handle_id)
        if state_handle is not None:
            state_handle.checkpoint_id = checkpoint.checkpoint_id
            self.temporal_store.state_handles.put(state_handle)
        return checkpoint.checkpoint_id

    def _load_state_goal_from_handle(self, state_handle_id: str) -> tuple[torch.Tensor, torch.Tensor, int]:
        state_handle = self.temporal_store.state_handles.get(state_handle_id)
        if state_handle is None:
            raise KeyError(state_handle_id)
        metadata = state_handle.metadata
        if "latent_state" not in metadata or "goal_state" not in metadata:
            raise ValueError(f"state_handle {state_handle_id} is missing latent_state/goal_state metadata")
        state = torch.as_tensor(metadata["latent_state"], dtype=self.dtype, device=self.device).unsqueeze(0)
        goal = torch.as_tensor(metadata["goal_state"], dtype=self.dtype, device=self.device).unsqueeze(0)
        step_idx = int(metadata.get("step_idx", 0))
        return state, goal, step_idx

    def _load_stateless_context(
        self,
        *,
        state_handle_id: str,
        trajectory_id: str | None,
        max_episode_steps: int | None,
        policy_version: str | None,
    ) -> _StatelessTransitionContext:
        state_handle = self.temporal_store.state_handles.get(state_handle_id)
        if state_handle is None:
            raise KeyError(state_handle_id)
        metadata = state_handle.metadata
        env_name = str(metadata.get("env_name"))
        task_id = str(metadata.get("task_id"))
        if not env_name or env_name == "None":
            raise ValueError(f"state_handle {state_handle_id} is missing metadata.env_name")
        if not task_id or task_id == "None":
            raise ValueError(f"state_handle {state_handle_id} is missing metadata.task_id")
        if state_handle.branch_id is None:
            raise ValueError(f"state_handle {state_handle_id} is missing branch_id")
        state, goal, step_idx = self._load_state_goal_from_handle(state_handle_id)
        task = self._resolve_task(task_id, env_name=env_name)
        spec = self._resolve_env_spec(env_name)
        trajectory = self._ensure_stateless_trajectory(
            env_name=env_name,
            task=task,
            episode_id=state_handle.episode_id,
            branch_id=state_handle.branch_id,
            trajectory_id=trajectory_id,
            policy_version=policy_version,
            max_episode_steps=max_episode_steps or spec.default_horizon,
            metadata={"source_state_handle_id": state_handle_id},
        )
        return _StatelessTransitionContext(
            env_name=env_name,
            task_id=task_id,
            episode_id=state_handle.episode_id,
            branch_id=state_handle.branch_id,
            trajectory_id=trajectory.trajectory_id,
            state_handle_id=state_handle_id,
            checkpoint_id=state_handle.checkpoint_id,
            policy_version=policy_version or trajectory.policy_version,
            max_episode_steps=int(trajectory.metadata.get("max_episode_steps", max_episode_steps or spec.default_horizon)),
            state=state,
            goal=goal,
            step_idx=step_idx,
            scope_id=trajectory.env_id,
        )

    def _ensure_stateless_trajectory(
        self,
        *,
        env_name: str,
        task: TaskSpec,
        episode_id: str,
        branch_id: str,
        trajectory_id: str | None,
        policy_version: str | None,
        max_episode_steps: int,
        metadata: dict[str, Any],
    ) -> TrajectoryRecord:
        if trajectory_id is not None:
            trajectory = self.temporal_store.trajectories.get(trajectory_id)
            if trajectory is None:
                raise KeyError(trajectory_id)
            if trajectory.episode_id != episode_id:
                raise ValueError("trajectory_id does not belong to the requested episode/state")
            if trajectory.task_id != task.task_id:
                raise ValueError("trajectory_id does not match the requested task")
            if policy_version is not None and trajectory.policy_version != policy_version:
                trajectory.policy_version = policy_version
                self.temporal_store.update_trajectory(trajectory)
            if "max_episode_steps" not in trajectory.metadata:
                trajectory.metadata["max_episode_steps"] = max_episode_steps
                self.temporal_store.update_trajectory(trajectory)
            return trajectory

        scope_id = f"stateless:{episode_id}:{branch_id}"
        return self.temporal_store.create_trajectory(
            TrajectoryCreate(
                env_id=scope_id,
                episode_id=episode_id,
                task_id=task.task_id,
                policy_version=policy_version,
                metadata={
                    "env_name": env_name,
                    "branch_id": branch_id,
                    "task_split": task.split,
                    "max_episode_steps": max_episode_steps,
                    "stateless": True,
                    **metadata,
                },
            )
        )

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

    def _build_stateless_step_chunks(
        self,
        contexts: list[_StatelessTransitionContext],
        action_tensor: torch.Tensor,
    ) -> list[ExecutionChunk]:
        if not contexts:
            return []
        signature = BatchSignature(
            stage="stateless_transition",
            latent_shape=tuple(contexts[0].state.shape[-2:]),
            action_dim=int(action_tensor.shape[-1]),
            dtype=str(self.dtype).replace("torch.", ""),
            device=str(self.device),
            needs_decode=False,
        )
        entities = [
            ExecutionEntity(
                entity_id=f"{context.state_handle_id}:transition:{context.step_idx}",
                rollout_id=context.state_handle_id,
                stage="stateless_transition",
                step_idx=context.step_idx,
                batch_signature=signature,
            )
            for context in contexts
        ]
        latent_items = [context.state for context in contexts]
        action_items = [action_tensor[index:index + 1] for index in range(action_tensor.shape[0])]
        return build_execution_chunks(
            signature=signature,
            entities=entities,
            latent_items=latent_items,
            action_items=action_items,
            policy=self.env_step_batch_policy,
            chunk_id_prefix="stateless_transition",
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
