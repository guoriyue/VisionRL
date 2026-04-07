"""Request/response protocol for the low-level rollout runtime API.

This module is intentionally narrower than the higher-level temporal sample
production API exposed through ``ProduceSampleRequest``.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class RolloutRequest(BaseModel):
    """Client request for a world model rollout."""

    model: str = Field(default="latent_dynamics", description="Model name from registry")
    initial_observation_b64: Optional[str] = Field(
        default=None,
        description="Base64-encoded initial observation (image/video frame)",
    )
    initial_latent: Optional[list[list[float]]] = Field(
        default=None,
        description="Pre-encoded latent state [N, D]",
    )
    actions: Optional[list[list[float]]] = Field(
        default=None,
        description="Action sequence [T, A] for each prediction step",
    )
    num_steps: int = Field(default=1, ge=1, le=128, description="Number of prediction steps")
    return_frames: bool = Field(default=True, description="Decode latents back to pixel frames")
    return_latents: bool = Field(default=False, description="Return raw latent states")
    stream: bool = Field(default=False, description="Stream results step by step via SSE")


class StepResult(BaseModel):
    """One prediction step result (used in streaming)."""

    step: int
    latent: Optional[list[list[float]]] = None
    frame_b64: Optional[str] = None

    def to_sse(self) -> str:
        """Format as a Server-Sent Event data line."""
        return f"data: {self.model_dump_json()}\n\n"


SSE_DONE = "data: [DONE]\n\n"


class RolloutResponse(BaseModel):
    """Complete rollout response."""

    job_id: str
    model: str
    steps_completed: int
    elapsed_ms: float
    latents: Optional[list[list[list[float]]]] = Field(
        default=None,
        description="Predicted latent states [T, N, D]",
    )
    frames_b64: Optional[list[str]] = Field(
        default=None,
        description="Base64-encoded predicted frames",
    )


class HealthResponse(BaseModel):
    """Server health check with readiness and liveness semantics."""

    status: str = "ok"
    model_loaded: bool = False
    engine_running: bool = False
    active_rollouts: int = 0
    memory_used_gb: float = 0.0


class ModelInfo(BaseModel):
    """Model information."""

    name: str
    num_parameters: int
    device: str
    dtype: str



class EnvironmentSessionResponse(BaseModel):
    env_id: str
    env_name: str
    episode_id: str
    task_id: str
    branch_id: Optional[str] = None
    state_handle_id: Optional[str] = None
    checkpoint_id: Optional[str] = None
    trajectory_id: Optional[str] = None
    current_step: int = 0
    policy_version: Optional[str] = None
    status: str
    observation: list[list[float]] = Field(default_factory=list)
    info: dict[str, Any] = Field(default_factory=dict)


class EnvironmentStepResponse(BaseModel):
    env_id: str
    episode_id: str
    task_id: str
    trajectory_id: Optional[str] = None
    state_handle_id: str
    checkpoint_id: Optional[str] = None
    transition_id: Optional[str] = None
    policy_version: Optional[str] = None
    step_idx: int
    observation: list[list[float]]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = Field(default_factory=dict)


class EnvironmentStepManyResponse(BaseModel):
    env_ids: list[str]
    results: list[EnvironmentStepResponse]
    runtime: dict[str, Any] = Field(default_factory=dict)


class TransitionInitializeRequest(BaseModel):
    env_name: str
    task_id: Optional[str] = None
    seed: Optional[int] = None
    policy_version: Optional[str] = None
    max_episode_steps: Optional[int] = Field(default=None, ge=1)
    branch_name: Optional[str] = None
    labels: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TransitionContextResponse(BaseModel):
    env_name: str
    episode_id: str
    task_id: str
    branch_id: str
    state_handle_id: str
    checkpoint_id: Optional[str] = None
    trajectory_id: str
    current_step: int = 0
    policy_version: Optional[str] = None
    max_episode_steps: int
    observation: list[list[float]] = Field(default_factory=list)
    info: dict[str, Any] = Field(default_factory=dict)


class TransitionPredictRequest(BaseModel):
    state_handle_id: str
    action: list[float]
    trajectory_id: Optional[str] = None
    policy_version: Optional[str] = None
    checkpoint: bool = False
    max_episode_steps: Optional[int] = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TransitionPredictItem(BaseModel):
    state_handle_id: str
    action: list[float]
    trajectory_id: Optional[str] = None
    max_episode_steps: Optional[int] = Field(default=None, ge=1)


class TransitionPredictResponse(BaseModel):
    env_name: str
    episode_id: str
    task_id: str
    branch_id: Optional[str] = None
    trajectory_id: str
    state_handle_id: str
    checkpoint_id: Optional[str] = None
    transition_id: str
    policy_version: Optional[str] = None
    step_idx: int
    max_episode_steps: int
    observation: list[list[float]]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = Field(default_factory=dict)


class TransitionPredictManyRequest(BaseModel):
    items: list[TransitionPredictItem] = Field(default_factory=list)
    policy_version: Optional[str] = None
    checkpoint: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class TransitionPredictManyResponse(BaseModel):
    results: list[TransitionPredictResponse] = Field(default_factory=list)
    runtime: dict[str, Any] = Field(default_factory=dict)
