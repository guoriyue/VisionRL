"""Token-level scheduler for AR image generation executors."""

from __future__ import annotations

from dataclasses import dataclass

from vrl.executors.ar.sequence import ActiveSequence, ARSequenceKey


@dataclass(slots=True)
class ARTokenBatch:
    """One token-forward batch inside an AR executor."""

    key: ARSequenceKey
    sequences: list[ActiveSequence]

    def __post_init__(self) -> None:
        if not self.sequences:
            raise ValueError("ARTokenBatch.sequences must be non-empty")
        for sequence in self.sequences:
            if sequence.key != self.key:
                raise ValueError("ARTokenBatch sequences must share the same key")

    @property
    def request_ids(self) -> list[str]:
        return [sequence.request_id for sequence in self.sequences]

    @property
    def sample_ids(self) -> list[str]:
        return [sequence.sample_id for sequence in self.sequences]


class ARTokenScheduler:
    """Group active AR sequences into same-backend token batches.

    This is an executor-internal scheduler. The global engine Scheduler still
    owns ``GenerationRequest`` lifecycle; this class only batches token
    forwards after a Janus/NextStep executor has expanded requests into active
    sequences.
    """

    def __init__(self, max_batch_size: int) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        self.max_batch_size = max_batch_size
        self._pending: list[ActiveSequence] = []

    def __len__(self) -> int:
        return sum(not sequence.finished for sequence in self._pending)

    def add(self, sequence: ActiveSequence) -> None:
        if not sequence.finished:
            self._pending.append(sequence)

    def add_many(self, sequences: list[ActiveSequence]) -> None:
        for sequence in sequences:
            self.add(sequence)

    def pop_batch(self) -> ARTokenBatch | None:
        groups: dict[tuple[ARSequenceKey, int], list[ActiveSequence]] = {}
        ordered_keys: list[tuple[ARSequenceKey, int]] = []
        retained: list[ActiveSequence] = []

        for sequence in self._pending:
            if sequence.finished:
                continue
            group_key = (sequence.key, sequence.position)
            if group_key not in groups:
                groups[group_key] = []
                ordered_keys.append(group_key)
            groups[group_key].append(sequence)
            retained.append(sequence)

        self._pending = retained
        if not groups:
            return None

        key, _position = ordered_keys[0]
        selected_group = groups[ordered_keys[0]]
        selected = selected_group[: self.max_batch_size]
        selected_ids = {id(sequence) for sequence in selected}
        self._pending = [
            sequence
            for sequence in self._pending
            if id(sequence) not in selected_ids
        ]
        return ARTokenBatch(key=key, sequences=selected)

    def push_back_unfinished(self, batch: ARTokenBatch) -> None:
        for sequence in batch.sequences:
            if not sequence.finished:
                self._pending.append(sequence)
