"""Active autoregressive sequence state for generation executors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ARSequenceKey:
    """Grouping key for token-level AR batching."""

    family: str
    task: str
    tokenizer_key: str
    dtype: str
    max_new_tokens: int


@dataclass(slots=True)
class ActiveSequence:
    """One in-flight AR image sequence inside a family executor."""

    request_id: str
    sample_id: str
    family: str
    task: str
    tokenizer_key: str
    dtype: str
    max_new_tokens: int
    position: int = 0
    finished: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("ActiveSequence.request_id must be non-empty")
        if not self.sample_id:
            raise ValueError("ActiveSequence.sample_id must be non-empty")
        if self.max_new_tokens < 1:
            raise ValueError("ActiveSequence.max_new_tokens must be >= 1")
        if self.position < 0:
            raise ValueError("ActiveSequence.position must be >= 0")
        if self.position >= self.max_new_tokens:
            self.finished = True

    @property
    def key(self) -> ARSequenceKey:
        return ARSequenceKey(
            family=self.family,
            task=self.task,
            tokenizer_key=self.tokenizer_key,
            dtype=self.dtype,
            max_new_tokens=self.max_new_tokens,
        )

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_new_tokens - self.position)

    def advance(self, steps: int = 1) -> None:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if self.finished:
            return
        self.position += steps
        if self.position >= self.max_new_tokens:
            self.position = self.max_new_tokens
            self.finished = True

    def mark_finished(self) -> None:
        self.position = min(self.position, self.max_new_tokens)
        self.finished = True
