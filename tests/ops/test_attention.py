"""Tests for attention backend dispatch."""

import pytest

from wm_infra.ops.attention import resolve_attention_backend


def test_resolve_attention_backend_rejects_triton_alias():
    with pytest.raises(ValueError, match="Unknown attention backend: triton"):
        resolve_attention_backend("triton")
