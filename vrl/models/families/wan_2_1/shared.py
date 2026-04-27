"""Shared utilities for Wan model family."""

from __future__ import annotations

import hashlib
from urllib.parse import unquote, urlparse


def stable_hash(value: str) -> str:
    """Short stable hash for cache keys."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def resolve_wan_reference_path(reference: str) -> str:
    if reference.startswith("file://"):
        parsed = urlparse(reference)
        return unquote(parsed.path)
    return reference
