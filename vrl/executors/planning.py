"""Execution planning helpers for generation executors."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class MicroBatchPlan:
    """One executor micro-batch for a prompt-major rollout request."""

    prompt_index: int
    prompt: str
    sample_start: int
    sample_count: int

    def __post_init__(self) -> None:
        if self.prompt_index < 0:
            raise ValueError("prompt_index must be >= 0")
        if self.sample_start < 0:
            raise ValueError("sample_start must be >= 0")
        if self.sample_count < 1:
            raise ValueError("sample_count must be >= 1")

    @property
    def sample_end(self) -> int:
        return self.sample_start + self.sample_count

    def split(self) -> tuple[MicroBatchPlan, MicroBatchPlan]:
        """Split this micro-batch into two ordered smaller chunks."""

        if self.sample_count <= 1:
            raise ValueError("Cannot split a single-sample micro-batch")
        left_count = self.sample_count // 2
        right_count = self.sample_count - left_count
        left = MicroBatchPlan(
            prompt_index=self.prompt_index,
            prompt=self.prompt,
            sample_start=self.sample_start,
            sample_count=left_count,
        )
        right = MicroBatchPlan(
            prompt_index=self.prompt_index,
            prompt=self.prompt,
            sample_start=self.sample_start + left_count,
            sample_count=right_count,
        )
        return left, right


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Prompt-major execution plan for one GenerationRequest."""

    prompts: tuple[str, ...]
    samples_per_prompt: int
    max_samples_per_microbatch: int
    micro_batches: tuple[MicroBatchPlan, ...]

    @property
    def total_samples(self) -> int:
        return len(self.prompts) * self.samples_per_prompt


@dataclass(frozen=True, slots=True)
class RolloutShardPlan:
    """Executable chunks for one large rollout request."""

    request_id: str
    chunks: tuple[MicroBatchPlan, ...]


def plan_prompt_group_microbatches(
    prompts: Sequence[str],
    samples_per_prompt: int,
    max_samples_per_microbatch: int,
) -> ExecutionPlan:
    """Plan prompt-major micro-batches without changing RL group semantics."""

    if not prompts:
        raise ValueError("prompts must be non-empty")
    if samples_per_prompt < 1:
        raise ValueError("samples_per_prompt must be >= 1")
    if max_samples_per_microbatch < 1:
        raise ValueError("max_samples_per_microbatch must be >= 1")

    micro_batches: list[MicroBatchPlan] = []
    for prompt_index, prompt in enumerate(prompts):
        sample_start = 0
        remaining = samples_per_prompt
        while remaining > 0:
            sample_count = min(max_samples_per_microbatch, remaining)
            micro_batches.append(
                MicroBatchPlan(
                    prompt_index=prompt_index,
                    prompt=prompt,
                    sample_start=sample_start,
                    sample_count=sample_count,
                )
            )
            sample_start += sample_count
            remaining -= sample_count

    return ExecutionPlan(
        prompts=tuple(prompts),
        samples_per_prompt=samples_per_prompt,
        max_samples_per_microbatch=max_samples_per_microbatch,
        micro_batches=tuple(micro_batches),
    )


def run_microbatches_with_oom_retry(
    micro_batches: Sequence[MicroBatchPlan],
    run_one: Callable[[MicroBatchPlan], T],
    *,
    min_sample_count: int = 1,
) -> list[T]:
    """Run micro-batches, splitting CUDA-OOM chunks until the floor is reached."""

    if min_sample_count < 1:
        raise ValueError("min_sample_count must be >= 1")

    results: list[T] = []
    pending = list(micro_batches)
    while pending:
        micro_batch = pending.pop(0)
        try:
            results.append(run_one(micro_batch))
        except RuntimeError as exc:
            if (
                not _is_cuda_oom(exc)
                or micro_batch.sample_count <= min_sample_count
            ):
                raise
            _clear_cuda_cache()
            left, right = micro_batch.split()
            pending.insert(0, right)
            pending.insert(0, left)
    return results


def _is_cuda_oom(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return "cuda" in msg and "out of memory" in msg


def _clear_cuda_cache() -> None:
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


__all__ = [
    "ExecutionPlan",
    "MicroBatchPlan",
    "RolloutShardPlan",
    "plan_prompt_group_microbatches",
    "run_microbatches_with_oom_retry",
]
