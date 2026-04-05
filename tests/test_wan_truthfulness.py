"""Focused tests for Wan backend truthfulness in stub mode."""

import pytest

from wm_infra.backends.wan import WanVideoBackend
from wm_infra.controlplane import ArtifactKind, ProduceSampleRequest, SampleSpec, SampleStatus, TaskType


@pytest.mark.asyncio
async def test_stub_mode_does_not_emit_video_artifact(tmp_path):
    backend = WanVideoBackend(str(tmp_path / "wan"))
    request = ProduceSampleRequest(
        task_type=TaskType.TEXT_TO_VIDEO,
        backend="wan-video",
        model="wan2.2-t2v-A14B",
        sample_spec=SampleSpec(prompt="stub truthfulness"),
    )

    record = await backend.produce_sample(request)

    assert record.status == SampleStatus.ACCEPTED
    assert record.runtime["runner"] == "stub"
    assert record.metadata["stubbed"] is True
    assert not (tmp_path / "wan" / record.sample_id / "sample.mp4").exists()

    artifact_kinds = {artifact.kind for artifact in record.artifacts}
    assert ArtifactKind.VIDEO not in artifact_kinds
    assert {ArtifactKind.LOG, ArtifactKind.METADATA}.issubset(artifact_kinds)
