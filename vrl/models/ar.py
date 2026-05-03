"""AutoregressivePolicy protocol for AR image-generation models.

Janus-Pro and NextStep-1 explicitly inherit this Protocol while keeping
family-specific replay math in their concrete policy classes
(categorical for Janus, Gaussian/flow for NextStep). The Protocol captures
only the minimum shared ownership: device, LoRA toggle, and the
replay-forward call.

The ``replay_forward`` return dict schema is intentionally NOT typed —
Janus returns ``{"logits", "target_tokens"}``, NextStep returns
``{"log_probs", "target_tokens"}``. Evaluators dispatch on dict keys.
See ``SPRINT_ar_support.md`` §5 for the rationale on not unifying.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch


@dataclass(slots=True)
class ARStepResult:
    """One scheduled AR token step.

    ``sequence_ids`` and every tensor-like value must follow the same order as
    the input active sequence list. ``replay_extras`` is reserved for per-step
    tensors needed by training replay; for NextStep-1 it must include
    ``saved_noise``.
    """

    sequence_ids: list[str]
    positions: list[int]
    token: Any
    log_prob: Any
    replay_extras: dict[str, Any] = field(default_factory=dict)


def ar_split_rows(value: Any, batch_size: int) -> list[Any]:
    """Split a batched AR cache/value into one-row values.

    HF-style ``past_key_values`` are nested tuples whose tensors carry batch as
    dim 0. Splitting to per-row caches avoids invalid scatter when partial AR
    scheduling lets rows reach different sequence lengths.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    value = _to_tuple_cache_if_needed(value)
    if _is_tensor(value):
        if value.shape[0] != batch_size:
            raise ValueError(
                f"cannot split tensor with batch={value.shape[0]} into "
                f"{batch_size} rows"
            )
        return [value[row : row + 1] for row in range(batch_size)]
    if isinstance(value, Mapping):
        split_items = {
            key: ar_split_rows(inner, batch_size) for key, inner in value.items()
        }
        return [
            type(value)((key, parts[row]) for key, parts in split_items.items())
            for row in range(batch_size)
        ]
    if isinstance(value, tuple):
        split_items = [ar_split_rows(inner, batch_size) for inner in value]
        return [
            tuple(parts[row] for parts in split_items)
            for row in range(batch_size)
        ]
    if isinstance(value, list):
        split_items = [ar_split_rows(inner, batch_size) for inner in value]
        return [
            [parts[row] for parts in split_items]
            for row in range(batch_size)
        ]
    return [value for _ in range(batch_size)]


def ar_concat_rows(values: Sequence[Any]) -> Any:
    """Concatenate one-row AR cache/value objects along batch dim 0."""

    if not values:
        raise ValueError("values must be non-empty")
    first = _to_tuple_cache_if_needed(values[0])
    rest = [_to_tuple_cache_if_needed(value) for value in values[1:]]
    values = [first, *rest]
    if _is_tensor(first):
        return torch.cat(list(values), dim=0)
    if isinstance(first, Mapping):
        return type(first)(
            (key, ar_concat_rows([value[key] for value in values]))
            for key in first
        )
    if isinstance(first, tuple):
        return tuple(
            ar_concat_rows([value[index] for value in values])
            for index in range(len(first))
        )
    if isinstance(first, list):
        return [
            ar_concat_rows([value[index] for value in values])
            for index in range(len(first))
        ]
    if any(value != first for value in values[1:]):
        raise ValueError("cannot concatenate non-tensor AR values that differ")
    return first


def _to_tuple_cache_if_needed(value: Any) -> Any:
    to_legacy_cache = getattr(value, "to_legacy_cache", None)
    if callable(to_legacy_cache):
        return to_legacy_cache()
    return value


def _is_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor)


@runtime_checkable
class AutoregressivePolicy(Protocol):
    """Minimal AR policy protocol shared by Janus-Pro and NextStep-1."""

    @property
    def device(self) -> torch.device:
        """Device the policy currently lives on."""
        ...

    def disable_adapter(self) -> AbstractContextManager[None]:
        """Context manager that disables LoRA / adapter weights.

        Used by evaluators to compute the reference-policy log-prob
        (LoRA-off forward) without unloading weights.
        """
        ...

    def replay_forward(self, batch: Any, timestep_idx: int = 0) -> dict[str, Any]:
        """Recompute the training-time forward to produce logits / log-probs.

        The return-dict schema is family-specific:
          - Janus  → ``{"logits": Tensor[B, L, V], "target_tokens": Tensor[B, L]}``
          - NextStep → ``{"log_probs": Tensor[B, L], "target_tokens": Tensor[B, L, D]}``

        Evaluators dispatch on dict keys; do not unify the schema here.
        """
        ...
