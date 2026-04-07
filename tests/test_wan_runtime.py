"""Focused tests for Wan runtime batching and warm profile hints."""

import pytest

from wm_infra.backends.wan import WanVideoBackend
from wm_infra.controlplane import ProduceSampleRequest, SampleSpec, TaskType, WanTaskConfig


def _wan_request(
    *,
    width: int = 832,
    height: int = 480,
    frame_count: int = 9,
    num_steps: int = 4,
    guidance_scale: float = 4.0,
) -> ProduceSampleRequest:
    return ProduceSampleRequest(
        task_type=TaskType.TEXT_TO_VIDEO,
        backend="wan-video",
        model="wan2.2-t2v-A14B",
        sample_spec=SampleSpec(prompt="wan runtime test"),
        wan_config=WanTaskConfig(
            width=width,
            height=height,
            frame_count=frame_count,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
        ),
    )


def test_queue_batch_key_groups_same_shape_and_near_cfg(tmp_path):
    backend = WanVideoBackend(tmp_path / "wan")

    key_a = backend.queue_batch_key(_wan_request(guidance_scale=4.1))
    key_b = backend.queue_batch_key(_wan_request(guidance_scale=4.2))
    key_c = backend.queue_batch_key(_wan_request(guidance_scale=4.8))
    key_d = backend.queue_batch_key(_wan_request(width=960))

    assert key_a == key_b
    assert key_a != key_c
    assert key_a != key_d


@pytest.mark.asyncio
async def test_execute_job_batch_records_shared_scheduler_and_profile(tmp_path):
    backend = WanVideoBackend(tmp_path / "wan", max_batch_size=4, batch_wait_ms=2.0)
    batch_records = await backend.execute_job_batch(
        [
            (_wan_request(), "sample-a"),
            (_wan_request(), "sample-b"),
        ]
    )

    assert len(batch_records) == 2
    assert batch_records[0].runtime["scheduler"]["batched_across_requests"] is True
    assert batch_records[0].runtime["scheduler"]["batch_id"] == batch_records[1].runtime["scheduler"]["batch_id"]
    assert batch_records[0].runtime["compiled_graph_pool"]["profile_id"] == batch_records[1].runtime["compiled_graph_pool"]["profile_id"]
    assert batch_records[0].runtime["compiled_graph_pool"]["compile_state"] == "cold_start"

    warm_record = await backend.execute_job(_wan_request(), "sample-c")
    assert warm_record.runtime["compiled_graph_pool"]["warm_profile_hit"] is True
    assert warm_record.runtime["compiled_graph_pool"]["compile_state"] in {
        "warm_profile_new_batch_size",
        "warm_profile_batch_hit",
    }


def test_admission_quality_cost_hints_offer_step_and_preview_fallbacks(tmp_path):
    backend = WanVideoBackend(
        tmp_path / "wan",
        wan_admission_max_vram_gb=10.0,
        wan_admission_max_units=6.0,
    )
    request = _wan_request(width=1280, height=720, frame_count=21, num_steps=12)
    wan_config = backend._resolve_wan_config(request)

    admitted, admission, _estimate = backend._admission_result(request, wan_config)

    assert admitted is False
    policies = {item["policy"] for item in admission["quality_cost_hints"]["suggested_adjustments"]}
    assert "auto_step_reduction" in policies
    assert "progressive_preview" in policies
