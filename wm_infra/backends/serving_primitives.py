"""Shared low-level serving primitives for backend execution paths.

These objects are intentionally small and JSON-friendly. They are designed to
capture the fast-path execution facts that matter for extreme serving work:

- what execution family a request belongs to
- which compiled profile / graph family was reused
- where runtime objects currently reside
- how many bytes moved across host/device and artifact boundaries
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover - torch is expected in runtime, but keep import-safe helpers.
    torch = None


class ResidencyTier(str, Enum):
    """Generic runtime residency tier."""

    GPU_HOT = "gpu_hot"
    CPU_PINNED_WARM = "cpu_pinned_warm"
    CPU_WARM = "cpu_warm"
    DURABLE_ONLY = "durable_only"


def batch_size_family(batch_size: int) -> str:
    """Return a stable batch-size family key."""

    size = max(int(batch_size), 1)
    if size == 1:
        return "singleton"
    if size <= 2:
        return "pair"
    if size <= 4:
        return "small"
    if size <= 8:
        return "medium"
    return "large"


def stable_graph_key(parts: list[str | int | float | bool | None]) -> str:
    normalized = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class ExecutionFamily:
    """Backend-agnostic family key for low-level serving execution."""

    backend: str
    model: str
    stage: str
    device: str
    dtype: str
    runner_mode: str
    batch_size_family: str
    width: int | None = None
    height: int | None = None
    frame_count: int | None = None
    num_steps: int | None = None
    prompt_frames: int | None = None
    memory_mode: str | None = None
    layout_key: str | None = None
    execution_kind: str | None = None
    tokenizer_kind: str | None = None

    @property
    def cache_key(self) -> str:
        return stable_graph_key(
            [
                self.backend,
                self.model,
                self.stage,
                self.device,
                self.dtype,
                self.runner_mode,
                self.batch_size_family,
                self.width,
                self.height,
                self.frame_count,
                self.num_steps,
                self.prompt_frames,
                self.memory_mode,
                self.layout_key,
                self.execution_kind,
                self.tokenizer_kind,
            ]
        )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cache_key"] = self.cache_key
        return payload


@dataclass(slots=True)
class CompiledProfile:
    """Low-level compiled profile / graph-cache reuse metadata."""

    profile_id: str
    execution_family: ExecutionFamily
    graph_key: str
    compile_state: str
    warm_profile_hit: bool
    compiled_batch_size_hit: bool
    compiled_batch_sizes: list[int]
    reuse_count: int
    prewarmed: bool = False
    compile_latency_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["execution_family"] = self.execution_family.as_dict()
        return payload


@dataclass(slots=True)
class ResidencyRecord:
    """Best-effort residency record for one runtime object."""

    object_id: str
    tier: ResidencyTier
    bytes_size: int
    layout_key: str | None = None
    pinned: bool = False
    reusable: bool = True
    source: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tier"] = self.tier.value
        return payload


@dataclass(slots=True)
class TransferPlan:
    """Best-effort transfer accounting for one execution path."""

    h2d_bytes: int = 0
    d2h_bytes: int = 0
    device_to_device_bytes: int = 0
    artifact_io_bytes: int = 0
    staging_bytes: int = 0
    overlap_h2d_with_compute: bool = False
    overlap_d2h_with_io: bool = False
    staging_tier: ResidencyTier = ResidencyTier.CPU_PINNED_WARM
    notes: list[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return (
            int(self.h2d_bytes)
            + int(self.d2h_bytes)
            + int(self.device_to_device_bytes)
            + int(self.artifact_io_bytes)
        )

    def add_h2d(self, bytes_size: int) -> None:
        self.h2d_bytes += max(int(bytes_size), 0)

    def add_d2h(self, bytes_size: int) -> None:
        self.d2h_bytes += max(int(bytes_size), 0)

    def add_device_to_device(self, bytes_size: int) -> None:
        self.device_to_device_bytes += max(int(bytes_size), 0)

    def add_artifact_io(self, bytes_size: int) -> None:
        self.artifact_io_bytes += max(int(bytes_size), 0)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["staging_tier"] = self.staging_tier.value
        payload["total_bytes"] = self.total_bytes
        return payload


@dataclass(slots=True)
class CompiledGraphLifecycle:
    """Track capture/replay lifecycle for one compiled execution family."""

    execution_family: ExecutionFamily
    graph_key: str
    captures: int = 0
    replays: int = 0
    capture_latency_ms: float = 0.0
    last_capture_at: float = 0.0
    last_replay_at: float = 0.0
    notes: list[str] = field(default_factory=list)

    def record_capture(self, latency_ms: float, *, note: str | None = None, timestamp: float = 0.0) -> None:
        self.captures += 1
        self.capture_latency_ms = max(float(latency_ms), 0.0)
        self.last_capture_at = timestamp
        if note:
            self.notes.append(note)

    def record_replay(self, *, note: str | None = None, timestamp: float = 0.0) -> None:
        self.replays += 1
        self.last_replay_at = timestamp
        if note:
            self.notes.append(note)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["execution_family"] = self.execution_family.as_dict()
        return payload


class PinnedHostStagingBuffer:
    """Pinned host staging buffer for async host/device transfer overlap."""

    def __init__(self, size_bytes: int) -> None:
        self.size_bytes = max(int(size_bytes), 0)
        self.pinned = bool(torch is not None)
        if torch is not None and self.size_bytes > 0:
            self.buffer = torch.empty(self.size_bytes, dtype=torch.uint8, pin_memory=True)
        else:
            self.buffer = bytearray(self.size_bytes)

    def fits(self, size_bytes: int) -> bool:
        return max(int(size_bytes), 0) <= self.size_bytes

    def as_dict(self) -> dict[str, Any]:
        return {
            "size_bytes": self.size_bytes,
            "pinned": self.pinned,
        }


class PinnedHostStagingPool:
    """Best-effort ring-style pinned staging pool."""

    def __init__(self, total_size_bytes: int) -> None:
        self.total_size_bytes = max(int(total_size_bytes), 0)
        self._lock = threading.Lock()
        self._head = 0
        self._allocations: dict[int, tuple[int, int]] = {}
        self._next_alloc_id = 0
        self._buffer = PinnedHostStagingBuffer(self.total_size_bytes)

    def reserve(self, size_bytes: int) -> tuple[int, int] | None:
        size = max(int(size_bytes), 0)
        if size == 0 or size > self.total_size_bytes:
            return None
        with self._lock:
            if self._head + size > self.total_size_bytes:
                self._head = 0
            alloc_id = self._next_alloc_id
            self._next_alloc_id += 1
            offset = self._head
            self._allocations[alloc_id] = (offset, size)
            self._head += size
            return alloc_id, offset

    def release(self, alloc_id: int) -> None:
        with self._lock:
            self._allocations.pop(int(alloc_id), None)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            used = sum(size for _offset, size in self._allocations.values())
            return {
                "total_size_bytes": self.total_size_bytes,
                "used_bytes": used,
                "free_bytes": max(self.total_size_bytes - used, 0),
                "allocations": len(self._allocations),
                "pinned": self._buffer.pinned,
            }
