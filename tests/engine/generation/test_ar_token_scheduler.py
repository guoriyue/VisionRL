"""Tests for AR active sequence scheduling."""

from __future__ import annotations

from vrl.engine.generation.ar import ActiveSequence, ARTokenScheduler


def _seq(
    sample_id: str,
    *,
    family: str = "janus_pro",
    task: str = "ar_t2i",
    tokenizer_key: str = "janus-pro-1b",
    dtype: str = "bfloat16",
    max_new_tokens: int = 576,
    position: int = 0,
) -> ActiveSequence:
    return ActiveSequence(
        request_id=f"req-{sample_id}",
        sample_id=sample_id,
        family=family,
        task=task,
        tokenizer_key=tokenizer_key,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        position=position,
    )


def test_active_sequence_advance_marks_finished_at_token_limit() -> None:
    sequence = _seq("s0", max_new_tokens=3)

    sequence.advance()
    assert sequence.position == 1
    assert sequence.finished is False
    assert sequence.remaining_tokens == 2

    sequence.advance(2)
    assert sequence.position == 3
    assert sequence.finished is True
    assert sequence.remaining_tokens == 0


def test_token_scheduler_batches_same_family_task_token_shape_and_position() -> None:
    scheduler = ARTokenScheduler(max_batch_size=2)
    scheduler.add_many([
        _seq("a", position=0),
        _seq("b", position=0),
        _seq("c", max_new_tokens=1024),
        _seq("d", position=8),
    ])

    first = scheduler.pop_batch()
    assert first is not None
    assert first.sample_ids == ["a", "b"]
    assert first.key.family == "janus_pro"
    assert first.key.max_new_tokens == 576

    second = scheduler.pop_batch()
    assert second is not None
    assert second.sample_ids == ["c"]
    assert second.key.max_new_tokens == 1024

    third = scheduler.pop_batch()
    assert third is not None
    assert third.sample_ids == ["d"]
    assert third.key.max_new_tokens == 576


def test_token_scheduler_does_not_mix_tokenizers_or_dtypes() -> None:
    scheduler = ARTokenScheduler(max_batch_size=4)
    scheduler.add_many([
        _seq("janus", tokenizer_key="janus"),
        _seq("nextstep", family="nextstep_1", tokenizer_key="nextstep"),
        _seq("fp16", dtype="float16"),
    ])

    batches = [
        scheduler.pop_batch(),
        scheduler.pop_batch(),
        scheduler.pop_batch(),
    ]

    assert [batch.sample_ids for batch in batches if batch is not None] == [
        ["janus"],
        ["nextstep"],
        ["fp16"],
    ]


def test_token_scheduler_can_push_back_unfinished_sequences() -> None:
    scheduler = ARTokenScheduler(max_batch_size=2)
    scheduler.add_many([_seq("a", max_new_tokens=2), _seq("b", max_new_tokens=2)])

    batch = scheduler.pop_batch()
    assert batch is not None
    batch.sequences[0].advance()
    batch.sequences[1].advance(2)
    scheduler.push_back_unfinished(batch)

    next_batch = scheduler.pop_batch()
    assert next_batch is not None
    assert next_batch.sample_ids == ["a"]
