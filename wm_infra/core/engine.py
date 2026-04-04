"""WorldModelEngine: orchestrates tokenization, prediction, and decoding.

The engine manages the full rollout lifecycle:
  1. Accept rollout requests (observation + action sequence)
  2. Tokenize observation into latent state
  3. Schedule prediction steps across concurrent rollouts
  4. Execute dynamics model predictions (batched)
  5. Optionally decode latent states back to pixel frames
  6. Stream results back to the client
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

import torch
import torch.nn as nn

from wm_infra.config import EngineConfig
from wm_infra.core.state import LatentStateManager
from wm_infra.core.scheduler import RolloutScheduler, RolloutRequest, ScheduledBatch
from wm_infra.models.base import WorldModel, RolloutInput, RolloutOutput
from wm_infra.tokenizer.video_tokenizer import VideoTokenizer


@dataclass(slots=True)
class RolloutJob:
    """A user-facing rollout job."""

    job_id: str
    initial_observation: Optional[torch.Tensor] = None  # [C, H, W] or [T, C, H, W]
    initial_latent: Optional[torch.Tensor] = None  # [N, D] pre-encoded
    actions: Optional[torch.Tensor] = None  # [T, A]
    num_steps: int = 1
    return_frames: bool = True
    return_latents: bool = False
    stream: bool = False
    created_at: float = field(default_factory=time.monotonic)


@dataclass(slots=True)
class RolloutResult:
    """Result of a completed rollout."""

    job_id: str
    predicted_frames: Optional[torch.Tensor] = None  # [T, C, H, W]
    predicted_latents: Optional[torch.Tensor] = None  # [T, N, D]
    elapsed_ms: float = 0.0
    steps_completed: int = 0


class WorldModelEngine:
    """Main inference engine for world model serving.

    Orchestrates:
    - VideoTokenizer: observation -> latent tokens
    - LatentDynamicsModel: latent + action -> next latent
    - LatentStateManager: temporal state across rollout steps
    - RolloutScheduler: batching concurrent rollouts
    """

    def __init__(
        self,
        config: EngineConfig,
        dynamics_model: nn.Module,
        tokenizer: Optional[VideoTokenizer] = None,
    ):
        self.config = config
        self.dynamics_model = dynamics_model
        self.tokenizer = tokenizer

        dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
        self.dtype = dtype_map.get(config.dtype, torch.float16)
        device_str = config.device.value if hasattr(config.device, 'value') else str(config.device)
        self.device = torch.device(device_str)

        # Move model to device
        self.dynamics_model = self.dynamics_model.to(self.device, self.dtype)
        self.dynamics_model.eval()
        if self.tokenizer is not None:
            self.tokenizer = self.tokenizer.to(self.device, self.dtype)
            self.tokenizer.eval()

        # State management
        self.state_manager = LatentStateManager(
            max_concurrent=config.state_cache.max_batch_size,
            max_memory_gb=config.state_cache.pool_size_gb,
            device=self.device,
        )

        # Scheduling
        self.scheduler = RolloutScheduler(config.scheduler)

        # Job tracking
        self._jobs: dict[str, RolloutJob] = {}
        self._results: dict[str, RolloutResult] = {}

    def submit_job(self, job: RolloutJob) -> str:
        """Submit a rollout job. Returns job_id."""
        if not job.job_id:
            job.job_id = str(uuid.uuid4())

        self._jobs[job.job_id] = job
        self.scheduler.submit(RolloutRequest(
            request_id=job.job_id,
            num_steps=job.num_steps,
        ))
        return job.job_id

    @torch.inference_mode()
    def step(self) -> list[str]:
        """Run one engine step: schedule + execute one batch of predictions.

        Returns:
            List of job IDs that completed during this step
        """
        # 1. Admit pending jobs and encode initial states
        admitted = self.scheduler.admit()
        for job_id in admitted:
            self._initialize_rollout(job_id)

        # 2. Schedule a batch
        batch = self.scheduler.schedule_batch()
        if batch.size == 0:
            return []

        # 3. Execute predictions for the batch
        completed_ids = self._execute_batch(batch)

        # 4. Finalize completed jobs
        for job_id in completed_ids:
            self._finalize_job(job_id)

        return completed_ids

    def run_until_done(self) -> list[RolloutResult]:
        """Run engine until all submitted jobs are complete."""
        all_completed = []
        while self.scheduler.has_work():
            completed = self.step()
            for job_id in completed:
                if job_id in self._results:
                    all_completed.append(self._results[job_id])
        return all_completed

    def get_result(self, job_id: str) -> Optional[RolloutResult]:
        return self._results.get(job_id)

    def has_pending_work(self) -> bool:
        return self.scheduler.has_work()

    # ─── Internal ───

    def _initialize_rollout(self, job_id: str) -> None:
        """Encode initial observation and create rollout state."""
        job = self._jobs[job_id]

        if job.initial_latent is not None:
            initial_state = job.initial_latent.to(self.device, self.dtype)
        elif job.initial_observation is not None and self.tokenizer is not None:
            obs = job.initial_observation.to(self.device, self.dtype)
            if obs.ndim == 3:  # [C, H, W] single frame
                obs = obs.unsqueeze(0).unsqueeze(0)  # [1, 1, C, H, W]
            elif obs.ndim == 4:  # [T, C, H, W]
                obs = obs.unsqueeze(0)  # [1, T, C, H, W]
            z_q, _ = self.tokenizer.encode(obs)
            initial_state = z_q.squeeze(0)[-1]  # last frame's tokens: [N, D]
        else:
            raise ValueError(f"Job {job_id}: must provide initial_observation or initial_latent")

        if initial_state.ndim == 2:
            initial_state = initial_state.unsqueeze(0)  # [1, N, D]

        self.state_manager.create(job_id, initial_state, max_steps=job.num_steps)

    def _execute_batch(self, batch: ScheduledBatch) -> list[str]:
        """Execute one prediction step for a batch of rollouts."""
        completed = []

        for i, job_id in enumerate(batch.request_ids):
            step_idx = batch.step_indices[i]
            job = self._jobs[job_id]
            state = self.state_manager.get(job_id)

            # Get action for this step
            if job.actions is not None and step_idx < job.actions.shape[0]:
                action = job.actions[step_idx].unsqueeze(0).to(self.device, self.dtype)
            else:
                # Zero action if not provided
                action_dim = self.config.dynamics.action_dim
                action = torch.zeros(1, action_dim, device=self.device, dtype=self.dtype)

            # Get current latent state
            current_state = state.latent_states[-1]
            if current_state.ndim == 2:
                current_state = current_state.unsqueeze(0)

            # Predict next state
            next_state = self.dynamics_model.predict_next(current_state, action)

            # Update state
            self.state_manager.append_step(job_id, action.squeeze(0), next_state.squeeze(0))

            # Check completion
            if self.scheduler.step_completed(job_id):
                self.scheduler.complete(job_id)
                completed.append(job_id)

        return completed

    def _finalize_job(self, job_id: str) -> None:
        """Build result for a completed job."""
        job = self._jobs[job_id]
        state = self.state_manager.get(job_id)
        start_time = job.created_at

        # Stack predicted states (skip initial state)
        predicted_latents = torch.stack(state.latent_states[1:], dim=0)  # [T, N, D] or [T, B, N, D]

        result = RolloutResult(
            job_id=job_id,
            steps_completed=state.current_step,
            elapsed_ms=(time.monotonic() - start_time) * 1000,
        )

        if job.return_latents:
            result.predicted_latents = predicted_latents

        if job.return_frames and self.tokenizer is not None:
            # Decode latents back to frames
            if predicted_latents.ndim == 3:
                predicted_latents = predicted_latents.unsqueeze(0)  # [1, T, N, D]
            frames = self.tokenizer.decode(predicted_latents)
            result.predicted_frames = frames.squeeze(0)  # [T, C, H, W]

        self._results[job_id] = result

        # Cleanup state
        self.state_manager.remove(job_id)
        self._jobs.pop(job_id, None)
