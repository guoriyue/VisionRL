"""Tests for generation executor micro-batching helpers."""

from __future__ import annotations


def test_plan_prompt_group_microbatches_prompt_major() -> None:
    from vrl.engine.microbatching import plan_prompt_group_microbatches

    plan = plan_prompt_group_microbatches(
        ["a", "b"],
        samples_per_prompt=5,
        max_samples_per_microbatch=2,
    )

    got = [
        (mb.prompt_index, mb.prompt, mb.sample_start, mb.sample_count) for mb in plan.micro_batches
    ]
    assert got == [
        (0, "a", 0, 2),
        (0, "a", 2, 2),
        (0, "a", 4, 1),
        (1, "b", 0, 2),
        (1, "b", 2, 2),
        (1, "b", 4, 1),
    ]
    assert plan.total_samples == 10


def test_run_microbatches_with_oom_retry_splits_until_success() -> None:
    from vrl.engine.microbatching import (
        MicroBatchPlan,
        run_microbatches_with_oom_retry,
    )

    seen: list[tuple[int, int]] = []

    def run_one(micro_batch: MicroBatchPlan) -> int:
        seen.append((micro_batch.sample_start, micro_batch.sample_count))
        if micro_batch.sample_count > 2:
            raise RuntimeError("CUDA out of memory while allocating tensor")
        return micro_batch.sample_count

    results = run_microbatches_with_oom_retry(
        [
            MicroBatchPlan(
                prompt_index=0,
                prompt="a",
                sample_start=0,
                sample_count=5,
            )
        ],
        run_one,
    )

    assert results == [2, 1, 2]
    assert seen == [(0, 5), (0, 2), (2, 3), (2, 1), (3, 2)]
