"""First-class temporal control-plane entities and persistence.

This module gives wm-infra explicit, persisted world-model concepts instead of
forcing temporal state into loose sample metadata.
"""

from __future__ import annotations

import json
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field


class TemporalStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ARCHIVED = "archived"


class StateHandleKind(str, Enum):
    LATENT = "latent"
    VIDEO_LATENT = "video_latent"
    TOKEN_CACHE = "token_cache"
    FRAME = "frame"
    ACTION_TRACE = "action_trace"
    METADATA = "metadata"


class EpisodeRecord(BaseModel):
    episode_id: str
    title: Optional[str] = None
    description: Optional[str] = None
    status: TemporalStatus = TemporalStatus.ACTIVE
    labels: dict[str, str] = Field(default_factory=dict)
    seed: Optional[int] = None
    initial_prompt: Optional[str] = None
    parent_episode_id: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BranchRecord(BaseModel):
    branch_id: str
    episode_id: str
    parent_branch_id: Optional[str] = None
    forked_from_rollout_id: Optional[str] = None
    forked_from_checkpoint_id: Optional[str] = None
    name: str
    status: TemporalStatus = TemporalStatus.ACTIVE
    labels: dict[str, str] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StateHandleRecord(BaseModel):
    state_handle_id: str
    episode_id: str
    branch_id: Optional[str] = None
    rollout_id: Optional[str] = None
    checkpoint_id: Optional[str] = None
    kind: StateHandleKind = StateHandleKind.LATENT
    uri: Optional[str] = None
    shape: list[int] = Field(default_factory=list)
    dtype: Optional[str] = None
    version: int = 1
    is_terminal: bool = False
    created_at: float = Field(default_factory=time.time)
    artifact_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RolloutRecord(BaseModel):
    rollout_id: str
    episode_id: str
    branch_id: Optional[str] = None
    backend: str
    model: str
    status: TemporalStatus = TemporalStatus.PENDING
    sample_id: Optional[str] = None
    request_id: Optional[str] = None
    input_state_handle_id: Optional[str] = None
    output_state_handle_id: Optional[str] = None
    checkpoint_ids: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    step_count: int = 0
    priority: float = 0.0
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    updated_at: float = Field(default_factory=time.time)
    metrics: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointRecord(BaseModel):
    checkpoint_id: str
    episode_id: str
    rollout_id: Optional[str] = None
    branch_id: Optional[str] = None
    state_handle_id: Optional[str] = None
    artifact_ids: list[str] = Field(default_factory=list)
    step_index: int = 0
    tag: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EpisodeCreate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    labels: dict[str, str] = Field(default_factory=dict)
    seed: Optional[int] = None
    initial_prompt: Optional[str] = None
    parent_episode_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BranchCreate(BaseModel):
    episode_id: str
    parent_branch_id: Optional[str] = None
    forked_from_rollout_id: Optional[str] = None
    forked_from_checkpoint_id: Optional[str] = None
    name: str
    labels: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StateHandleCreate(BaseModel):
    episode_id: str
    branch_id: Optional[str] = None
    rollout_id: Optional[str] = None
    checkpoint_id: Optional[str] = None
    kind: StateHandleKind = StateHandleKind.LATENT
    uri: Optional[str] = None
    shape: list[int] = Field(default_factory=list)
    dtype: Optional[str] = None
    version: int = 1
    is_terminal: bool = False
    artifact_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RolloutCreate(BaseModel):
    episode_id: str
    branch_id: Optional[str] = None
    backend: str
    model: str
    sample_id: Optional[str] = None
    request_id: Optional[str] = None
    input_state_handle_id: Optional[str] = None
    artifact_ids: list[str] = Field(default_factory=list)
    step_count: int = 0
    priority: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointCreate(BaseModel):
    episode_id: str
    rollout_id: Optional[str] = None
    branch_id: Optional[str] = None
    state_handle_id: Optional[str] = None
    artifact_ids: list[str] = Field(default_factory=list)
    step_index: int = 0
    tag: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


T = TypeVar("T", bound=BaseModel)


class _EntityStore(Generic[T]):
    def __init__(self, root: Path, bucket: str, model_cls: type[T], id_field: str) -> None:
        self.root = root / bucket
        self.root.mkdir(parents=True, exist_ok=True)
        self.model_cls = model_cls
        self.id_field = id_field

    def _path(self, entity_id: str) -> Path:
        return self.root / f"{entity_id}.json"

    def put(self, record: T) -> T:
        self._path(getattr(record, self.id_field)).write_text(json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True))
        return record

    def get(self, entity_id: str) -> T | None:
        path = self._path(entity_id)
        if not path.exists():
            return None
        return self.model_cls.model_validate_json(path.read_text())

    def list(self) -> list[T]:
        return [self.model_cls.model_validate_json(path.read_text()) for path in sorted(self.root.glob("*.json"))]


class TemporalStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.episodes = _EntityStore(self.root, "episodes", EpisodeRecord, "episode_id")
        self.rollouts = _EntityStore(self.root, "rollouts", RolloutRecord, "rollout_id")
        self.state_handles = _EntityStore(self.root, "state_handles", StateHandleRecord, "state_handle_id")
        self.branches = _EntityStore(self.root, "branches", BranchRecord, "branch_id")
        self.checkpoints = _EntityStore(self.root, "checkpoints", CheckpointRecord, "checkpoint_id")

    def create_episode(self, request: EpisodeCreate) -> EpisodeRecord:
        now = time.time()
        record = EpisodeRecord(episode_id=str(uuid.uuid4()), updated_at=now, created_at=now, **request.model_dump())
        return self.episodes.put(record)

    def create_branch(self, request: BranchCreate) -> BranchRecord:
        now = time.time()
        record = BranchRecord(branch_id=str(uuid.uuid4()), updated_at=now, created_at=now, **request.model_dump())
        return self.branches.put(record)

    def create_state_handle(self, request: StateHandleCreate) -> StateHandleRecord:
        record = StateHandleRecord(state_handle_id=str(uuid.uuid4()), **request.model_dump())
        return self.state_handles.put(record)

    def create_rollout(self, request: RolloutCreate, *, status: TemporalStatus = TemporalStatus.PENDING) -> RolloutRecord:
        now = time.time()
        record = RolloutRecord(
            rollout_id=str(uuid.uuid4()),
            status=status,
            created_at=now,
            updated_at=now,
            started_at=now if status == TemporalStatus.ACTIVE else None,
            **request.model_dump(),
        )
        return self.rollouts.put(record)

    def update_rollout(self, rollout: RolloutRecord) -> RolloutRecord:
        rollout.updated_at = time.time()
        return self.rollouts.put(rollout)

    def create_checkpoint(self, request: CheckpointCreate) -> CheckpointRecord:
        record = CheckpointRecord(checkpoint_id=str(uuid.uuid4()), **request.model_dump())
        return self.checkpoints.put(record)

    def attach_checkpoint_to_rollout(self, rollout_id: str, checkpoint_id: str) -> RolloutRecord | None:
        rollout = self.rollouts.get(rollout_id)
        if rollout is None:
            return None
        if checkpoint_id not in rollout.checkpoint_ids:
            rollout.checkpoint_ids.append(checkpoint_id)
        return self.update_rollout(rollout)
