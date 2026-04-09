from __future__ import annotations

import types

import pytest
import torch

from tests.base import BaseTestCase
from wm_infra.backends.matrix_game import MatrixGameBackend
from wm_infra.controlplane import (
    ArtifactKind,
    ProduceSampleRequest,
    RolloutTaskConfig,
    SampleSpec,
    TaskType,
    WorldModelKind,
)
from wm_infra.engine.compat_rollout import RolloutResult


class _FakeAsyncEngine:
    def __init__(self) -> None:
        self.engine = types.SimpleNamespace(
            config=types.SimpleNamespace(
                state_cache=types.SimpleNamespace(num_latent_tokens=4),
                dynamics=types.SimpleNamespace(latent_token_dim=3, action_dim=2),
            )
        )
        self.seen_job = None

    async def submit(self, job):
        self.seen_job = job
        return RolloutResult(
            job_id="matrix-job-1",
            predicted_latents=torch.ones((job.num_steps, 4, 3), dtype=torch.float32),
            elapsed_ms=12.5,
            steps_completed=job.num_steps,
        )


class TestMatrixGameBackend(BaseTestCase):
    @staticmethod
    def make_request() -> ProduceSampleRequest:
        return ProduceSampleRequest(
            task_type=TaskType.TEMPORAL_ROLLOUT,
            backend="matrix-game",
            model="matrix-game-bringup",
            world_model_kind=WorldModelKind.DYNAMICS,
            sample_spec=SampleSpec(
                prompt="step the matrix world",
                controls={"actions": [[1.0, 0.0], [0.0, 1.0]]},
            ),
            task_config=RolloutTaskConfig(num_steps=2, frame_count=2),
            return_artifacts=[ArtifactKind.LATENT],
        )

    @pytest.mark.asyncio
    async def test_produces_dynamics_sample(self) -> None:
        engine = _FakeAsyncEngine()
        backend = MatrixGameBackend(engine)

        record = await backend.produce_sample(self.make_request())

        assert record.world_model_kind == WorldModelKind.DYNAMICS
        assert record.runtime["runtime_substrate"] == "rollout-engine"
        assert record.runtime["action_count"] == 2
        assert engine.seen_job is not None
        assert engine.seen_job.actions.shape == (2, 2)
        assert any(artifact.kind == ArtifactKind.LATENT for artifact in record.artifacts)

    @pytest.mark.asyncio
    async def test_rejects_mismatched_world_model_kind(self) -> None:
        engine = _FakeAsyncEngine()
        backend = MatrixGameBackend(engine)
        request = self.make_request().model_copy(update={"world_model_kind": WorldModelKind.GENERATION})

        with pytest.raises(ValueError, match="world_model_kind=dynamics"):
            await backend.produce_sample(request)
