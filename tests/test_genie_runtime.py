import numpy as np

from wm_infra.backends.genie_checkpoint import build_checkpoint_delta, checkpoint_due
from wm_infra.backends.genie_runtime import (
    GenieBatchSignature,
    GenieExecutionEntity,
    GeniePagedStateStore,
    GenieResidencyTier,
    GenieRuntimeState,
    build_benchmark_profile,
    build_runtime_state_profile,
    build_scheduler_profile,
    build_stage_profile,
    build_transition_entities,
    default_window_size,
    temperature_bucket,
)
from wm_infra.backends.genie_scheduler import GenieScheduler


def _root_entity() -> GenieExecutionEntity:
    signature = GenieBatchSignature(
        backend="genie-rollout",
        model_name="genie-local",
        stage="transition",
        device="cpu",
        dtype="uint32",
        tokenizer_kind="genie_stmaskgit",
        spatial_h=16,
        spatial_w=16,
        window_num_frames=12,
        num_prompt_frames=4,
        maskgit_steps=2,
        temperature_bucket=temperature_bucket(0.0),
        checkpoint_every_n_frames=4,
        runner_mode="stub",
        needs_persist=False,
    )
    return GenieExecutionEntity(
        entity_id="sample:root",
        rollout_id="rollout-1",
        episode_id="episode-1",
        branch_id="branch-1",
        sample_id="sample",
        input_state_handle_id="state-1",
        current_stage="transition",
        next_stage="artifact_persist",
        window_start_frame=4,
        window_num_frames=12,
        total_frames=16,
        num_prompt_frames=4,
        checkpoint_every_n_frames=4,
        priority=1.0,
        deadline_s=None,
        batch_signature=signature,
        queue_lane="hot_continuation",
    )


def test_default_window_size_respects_checkpoint_cadence():
    assert default_window_size(total_frames=16, prompt_frames=4, checkpoint_every_n_frames=0) == 12
    assert default_window_size(total_frames=16, prompt_frames=4, checkpoint_every_n_frames=4) == 4


def test_build_transition_entities_splits_by_checkpoint_window():
    entities = build_transition_entities(_root_entity())
    assert len(entities) == 3
    assert entities[0].window_start_frame == 4
    assert entities[0].window_end_frame == 8
    assert entities[1].window_start_frame == 8
    assert entities[2].window_end_frame == 16


def test_scheduler_prefers_hot_continuation_lane():
    runtime_state = GenieRuntimeState(
        rollout_id="rollout-1",
        prompt_tokens_ref=None,
        generated_tokens_ref=None,
        last_completed_frame=4,
        resident_tier="hot_gpu",
        ancestor_state_ref="state-1",
        checkpoint_delta_ref=None,
        materialized_bytes=4096,
        dirty_since_checkpoint=False,
        reuse_hits=1,
        reuse_misses=0,
    )
    scheduler = GenieScheduler(max_chunk_size=8)
    chunks = scheduler.build_chunks(build_transition_entities(_root_entity()), runtime_state)
    assert len(chunks) == 3
    assert chunks[0].queue_lane == "checkpoint_heavy"
    assert chunks[0].expected_occupancy > 0


def test_checkpoint_due_and_delta_metadata():
    assert checkpoint_due(frame_end=8, total_frames=16, checkpoint_every_n_frames=4) is True
    assert checkpoint_due(frame_end=16, total_frames=16, checkpoint_every_n_frames=4) is False

    token_window = np.zeros((4, 16, 16), dtype=np.uint32)
    delta = build_checkpoint_delta(
        rollout_id="rollout-1",
        sample_id="sample",
        parent_state_handle_id="state-1",
        all_tokens=np.concatenate([np.zeros((4, 16, 16), dtype=np.uint32), token_window], axis=0),
        start_frame=4,
        end_frame=8,
        checkpoint_every_n_frames=4,
        runner_mode="stub",
    )
    assert delta.artifact_id == "sample:checkpoint-delta:0008"
    assert delta.bytes_size == token_window.nbytes
    assert delta.metadata["frame_count"] == 4


def test_stage_and_scheduler_profiles_are_stable():
    stage_history = [
        {
            "stage": "transition",
            "elapsed_ms": 12.5,
            "queue_lane": "hot_continuation",
            "runner_mode": "stub",
            "chunk_id": "transition:0",
            "chunk_size": 2,
            "frame_range": [4, 8],
        },
        {
            "stage": "artifact_persist",
            "elapsed_ms": 1.2,
            "queue_lane": "persist_only",
            "runner_mode": "stub",
        },
    ]
    stage_timings_ms = {
        "transition_ms": 12.5,
        "artifact_persist_ms": 1.2,
        "controlplane_commit_ms": 0.7,
        "total_elapsed_ms": 16.0,
    }
    scheduler_profile = build_scheduler_profile(
        execution_path="runner_window_batch",
        transition_entities=3,
        chunks=[
            {
                "chunk_id": "transition:0",
                "queue_lane": "hot_continuation",
                "frame_range": [4, 8],
                "chunk_size": 2,
                "expected_occupancy": 1.0,
            }
        ],
        scheduler_inputs=[{"queue_lane": "hot_continuation"}],
        observed_batch_sizes=[2],
        batched_across_requests=True,
        cross_request_batcher=None,
    )
    benchmark_profile = build_benchmark_profile(
        stage_timings_ms=stage_timings_ms,
        scheduler_profile=scheduler_profile,
    )
    runtime_state_profile = build_runtime_state_profile(
        GenieRuntimeState(
            rollout_id="rollout-1",
            resident_tier=GenieResidencyTier.HOT_GPU,
            materialized_bytes=4096,
            last_completed_frame=8,
            dirty_since_checkpoint=True,
            source_cache_key="state_handle:1",
            reuse_hits=2,
            reuse_misses=1,
        )
    )
    stage_profile = build_stage_profile(stage_history, stage_timings_ms)

    assert stage_profile["completed_stages"] == ["transition", "artifact_persist"]
    assert stage_profile["stages"]["transition"]["max_chunk_size"] == 2
    assert scheduler_profile["execution_path"] == "runner_window_batch"
    assert scheduler_profile["max_observed_batch_size"] == 2
    assert benchmark_profile["chunk_count"] == 1
    assert benchmark_profile["batched_across_requests"] is True
    assert runtime_state_profile["dirty_since_checkpoint"] is True
    assert runtime_state_profile["source_cache_key"] == "state_handle:1"
    assert runtime_state_profile["layout_key"] == "token_frames_contiguous"
    assert runtime_state_profile["page_size_tokens"] == 1024
    assert runtime_state_profile["transfer_plan"] is None


def test_runtime_state_profile_exposes_residency_and_transfer_plan():
    runtime_state = GenieRuntimeState(
        rollout_id="rollout-2",
        resident_tier=GenieResidencyTier.WARM_PINNED_CPU,
        materialized_bytes=8192,
        page_size_tokens=512,
        page_count=4,
    )
    profile = build_runtime_state_profile(runtime_state)

    assert profile["resident_tier"] == "warm_pinned_cpu"
    assert profile["page_size_tokens"] == 512
    assert profile["page_count"] == 4
    assert profile["residency"] == []


def test_paged_state_store_updates_window_and_reports_page_span():
    tokens = np.arange(6 * 4 * 4, dtype=np.uint32).reshape(6, 4, 4)
    store = GeniePagedStateStore.from_tokens(
        store_id="rollout-1:paged-state",
        tokens=tokens,
        page_size_tokens=16,
    )

    assert store.page_count > 0
    left, right = store.page_span_for_frames(2, 4)
    assert left <= right

    replacement = np.zeros((2, 4, 4), dtype=np.uint32)
    store.update_window(frame_start=2, frame_end=4, tokens=replacement)
    materialized = store.materialize()
    window = store.window_tokens(2, 4)
    snapshot = store.snapshot()

    assert materialized.shape == tokens.shape
    assert np.all(materialized[2:4] == 0)
    assert window.shape == (2, 4, 4)
    assert np.all(window == 0)
    assert snapshot["page_pool"]["physical_bytes"] >= store.bytes_size
    assert snapshot["page_pool"]["dirty_page_count"] >= 1
    assert snapshot["page_pool"]["hot_page_count"] >= 1
    assert snapshot["page_pool"]["host_pool"]["page_count"] == store.page_count
    assert snapshot["page_pool"]["gpu_pool"]["page_count"] >= 1

    prefetch = store.prefetch_window(frame_start=0, frame_end=2, async_requested=True)
    assert prefetch["transfer"]["direction"] == "host_to_gpu"
    assert prefetch["transfer"]["async_requested"] is True
    assert prefetch["transfer"]["status"] == "pending"
    assert prefetch["prefetched_page_ids"]
    assert store.snapshot()["page_pool"]["hot_page_count"] >= 1
    assert store.transfer_queue_snapshot()["pending"] >= 1

    completed = store.poll_transfers()
    assert completed
    assert completed[0]["status"] == "completed"
    reclaimed = store.reclaim_completed_transfers()
    assert reclaimed

    evict = store.evict_window(frame_start=0, frame_end=2, async_requested=True)
    assert evict["transfer"]["direction"] == "gpu_to_host"
    assert not set(evict["evicted_page_ids"]) & set(store.snapshot()["hot_page_ids"])
    assert store.snapshot()["page_pool"]["hot_page_count"] <= store.gpu_hot_page_limit
    assert len(store.snapshot()["transfer_history"]) >= 2
    assert "transfer_queue" in store.snapshot()
