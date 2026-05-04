"""Shared request spec for autoregressive generation executors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ARGenerationSpec:
    """Family-neutral AR request fields.

    Family executors keep their own sampling math and output packing. This
    spec only captures shape, seed, and scheduler fields that Janus-Pro and
    NextStep-1 already parse identically.
    """

    image_token_num: int
    image_size: int
    max_text_length: int
    seed: int | None
    use_ar_scheduler: bool


__all__ = ["ARGenerationSpec"]
