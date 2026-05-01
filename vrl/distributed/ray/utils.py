"""Small Ray helper functions used by the distributed rollout layer."""

from __future__ import annotations

import importlib
import socket
from typing import Any


def require_ray() -> Any:
    """Import Ray lazily so base package imports do not require Ray."""

    try:
        import ray
    except ImportError as exc:  # pragma: no cover - exercised only without Ray
        raise ImportError(
            "Ray distributed rollout support requires `ray`. Install Ray or "
            "disable the Ray backend.",
        ) from exc
    return ray


def import_from_path(path: str) -> Any:
    """Load ``module:attribute`` or ``module.attribute`` import paths."""

    if ":" in path:
        module_name, attr_name = path.split(":", 1)
    else:
        module_name, _, attr_name = path.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"invalid import path: {path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def sort_node_gpu_key(item: tuple[int, str, int]) -> tuple[list[int], int, int]:
    """Stable sort key for placement bundles.

    Adapted from slime's Ray placement sorting: sort by node IP, then GPU ID,
    then bundle index. The bundle index tie-breaker keeps CPU-only local-mode
    tests deterministic.
    """

    index, node_identifier, gpu_id = item
    try:
        node_ip_parts = [int(part) for part in node_identifier.split(".")]
    except ValueError:
        try:
            resolved = socket.gethostbyname(node_identifier)
            node_ip_parts = [int(part) for part in resolved.split(".")]
        except (socket.gaierror, TypeError):
            node_ip_parts = [ord(char) for char in node_identifier]
    return node_ip_parts, gpu_id, index


def split_round_robin(items: list[Any], num_shards: int) -> list[list[Any]]:
    """Split items by ``items[i::num_shards]``."""

    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    return [items[i::num_shards] for i in range(num_shards)]
