"""Lazy dependency loading and Ray runtime metadata helpers."""

from __future__ import annotations

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


def current_node_ip() -> str:
    """Return the Ray node IP for the current actor process."""

    ray = require_ray()
    return str(ray.util.get_node_ip_address())


def current_gpu_ids() -> list[int]:
    """Return integer GPU IDs assigned to the current Ray actor."""

    ray = require_ray()
    ids = ray.get_gpu_ids()
    out: list[int] = []
    for gpu_id in ids:
        try:
            out.append(int(gpu_id))
        except (TypeError, ValueError):
            # Ray may report accelerator IDs that are not CUDA ordinals. Keep
            # this metadata path best-effort instead of failing rollout work.
            continue
    return out


__all__ = [
    "current_gpu_ids",
    "current_node_ip",
    "require_ray",
]
