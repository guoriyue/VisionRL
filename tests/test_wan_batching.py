from __future__ import annotations

import pytest

from wm_infra.backends.wan import WanVideoBackend
from wm_infra.controlplane import ProduceSampleRequest, SampleSpec, TaskType, WanTaskConfig


def _wan_request(
    *,
    prompt: str = "A corgi runs through a data center.",
    num_steps: int = 4,
    frame_count: int = 9,
    width: int = 832,
    height: int = 480,
    guidance_scale: float = 4.0,
    shift: float = 12.0,
) -> ProduceSampleRequest:
    return ProduceSampleRequest(
        task_type=TaskType.TEXT_TO_VIDEO,
        backend="wan-video",
        model="wan2.2-t2v-A14B",
        sample_spec=SampleSpec(prompt=prompt),
        wan_config=WanTaskConfig(
            num_steps=num_steps,
            frame_count=frame_count,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            shift=shift,
        ),
    )


def test_queue_batch_score_allows_nearby_wan_shapes(tmp_path):
    backend = WanVideoBackend(tmp_path / "wan")

    reference = _wan_request(num_steps=4, guidance_scale=4.0)
    nearby = _wan_request(num_steps=5, guidance_scale=4.5)

    assert backend.queue_batch_key(reference) != backend.queue_batch_key(nearby)
    assert backend.queue_batch_score(reference, nearby) is not None
    assert backend.queue_batch_score(reference, nearby) > 0


@pytest.mark.asyncio
async def test_execute_job_batch_records_scheduler_and_warm_pool_metadata(tmp_path):
    backend = WanVideoBackend(tmp_path / "wan", prewarm_common_signatures=True)

    records = await backend.execute_job_batch(
        [
            (_wan_request(prompt="batch-a"), "sample-a"),
            (_wan_request(prompt="batch-b"), "sample-b"),
        ]
    )

    assert len(records) == 2
    for record in records:
        assert record.metadata["queue_batched"] is True
        assert record.runtime["scheduler"]["batch_size"] == 2
        assert record.runtime["scheduler"]["execution_mode"] == "queue_coalesced_serial"
        assert record.runtime["compiled_graph_pool"]["compile_state"] == "prewarmed"
        assert record.runtime["engine_pool_snapshot"]["prewarmed_profiles"] >= 1
        assert record.runtime["scheduler"]["batch_signature"]["width"] == 832
        assert record.runtime["scheduler"]["batch_signature"]["num_steps"] == 4


def test_queue_batch_size_limit_respects_backend_cap(tmp_path):
    backend = WanVideoBackend(tmp_path / "wan", max_batch_size=3)

    assert backend.queue_batch_size_limit(1) == 1
    assert backend.queue_batch_size_limit(3) == 3
    assert backend.queue_batch_size_limit(8) == 3
