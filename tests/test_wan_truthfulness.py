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


@pytest.mark.asyncio
async def test_stub_batch_records_scheduler_and_warm_pool_state(tmp_path):
    backend = WanVideoBackend(
        str(tmp_path / "wan"),
        max_batch_size=4,
        prewarm_common_signatures=False,
    )
    request = ProduceSampleRequest(
        task_type=TaskType.TEXT_TO_VIDEO,
        backend="wan-video",
        model="wan2.2-t2v-A14B",
        sample_spec=SampleSpec(prompt="batched stub"),
    )

    records = await backend.execute_job_batch(
        [
            (request.model_copy(deep=True), "sample-a"),
            (request.model_copy(deep=True), "sample-b"),
        ]
    )

    assert len(records) == 2
    assert records[0].metadata["queue_batched"] is True
    assert records[0].runtime["scheduler"]["batch_size"] == 2
    assert records[0].runtime["scheduler"]["batched_across_requests"] is True
    assert records[0].runtime["compiled_graph_pool"]["warm_profile_hit"] is False
    assert records[0].runtime["engine_pool_snapshot"]["profiles"] == 1

    warm_followup = await backend.execute_job(request.model_copy(deep=True), "sample-c")
    assert warm_followup.runtime["compiled_graph_pool"]["warm_profile_hit"] is True
