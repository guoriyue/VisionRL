"""Tests for AR per-row cache helpers."""

from __future__ import annotations

import pytest
import torch

from vrl.models.ar import ar_concat_rows, ar_split_rows


def test_ar_split_and_concat_rows_preserve_nested_kv_order() -> None:
    key = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4)
    value = key + 100
    cache = {
        "past_key_values": ((key, value),),
        "last_hidden": torch.arange(3 * 5, dtype=torch.float32).reshape(3, 5),
    }

    rows = ar_split_rows(cache, 3)
    assert len(rows) == 3
    assert torch.equal(rows[1]["past_key_values"][0][0], key[1:2])
    assert torch.equal(rows[2]["last_hidden"], cache["last_hidden"][2:3])

    merged = ar_concat_rows([rows[2], rows[0]])
    assert torch.equal(merged["past_key_values"][0][0], torch.cat([key[2:3], key[0:1]]))
    assert torch.equal(
        merged["last_hidden"],
        torch.cat([cache["last_hidden"][2:3], cache["last_hidden"][0:1]]),
    )


def test_ar_split_rows_rejects_wrong_batch_size() -> None:
    with pytest.raises(ValueError, match="cannot split tensor"):
        ar_split_rows(torch.zeros(2, 4), 3)
