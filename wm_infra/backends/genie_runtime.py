"""Execution-plane objects and state helpers for stage-oriented Genie runtime."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for cpu-only tests
    torch = None  # type: ignore[assignment]

from wm_infra.backends.serving_primitives import (
    ExecutionFamily,
    ResidencyRecord,
    ResidencyTier,
    TransferPlan,
    batch_size_family,
)

GENIE_STAGE_GRAPH = [
    "admission",
    "state_materialize",
    "prompt_prepare",
    "transition",
    "checkpoint",
    "artifact_persist",
    "controlplane_commit",
]


class GenieQueueLane(str, Enum):
    """Execution lane for Genie stage scheduling."""

    HOT_CONTINUATION = "hot_continuation"
    COLD_MATERIALIZE = "cold_materialize"
    CHECKPOINT_HEAVY = "checkpoint_heavy"
    PERSIST_ONLY = "persist_only"


class GenieResidencyTier(str, Enum):
    """Best-effort residency tier for prompt / token state."""

    HOT_GPU = "hot_gpu"
    WARM_PINNED_CPU = "warm_pinned_cpu"
    COLD_FILE = "cold_file"


@dataclass(frozen=True, slots=True)
class GenieBatchSignature:
    """Execution signature for grouping homogeneous Genie work."""

    backend: str
    model_name: str
    stage: str
    device: str
    dtype: str
    tokenizer_kind: str
    spatial_h: int
    spatial_w: int
    window_num_frames: int
    num_prompt_frames: int
    maskgit_steps: int
    temperature_bucket: str
    checkpoint_every_n_frames: int
    runner_mode: str
    needs_persist: bool


@dataclass(slots=True)
class GenieExecutionEntity:
    """A bounded schedulable window of Genie rollout work."""

    entity_id: str
    rollout_id: str
    episode_id: str
    branch_id: str | None
    sample_id: str
    input_state_handle_id: str | None
    current_stage: str
    next_stage: str | None
    window_start_frame: int
    window_num_frames: int
    total_frames: int
    num_prompt_frames: int
    checkpoint_every_n_frames: int
    priority: float
    deadline_s: float | None
    batch_signature: GenieBatchSignature
    queue_lane: GenieQueueLane
    stage_attempts: int = 0
    last_scheduled_at: float = 0.0

    @property
    def window_end_frame(self) -> int:
        return self.window_start_frame + self.window_num_frames


@dataclass(slots=True)
class GenieRuntimeState:
    """Hot execution state for a Genie rollout."""

    rollout_id: str
    prompt_tokens_ref: object | None = None
    generated_tokens_ref: object | None = None
    last_completed_frame: int = 0
    resident_tier: GenieResidencyTier = GenieResidencyTier.COLD_FILE
    ancestor_state_ref: str | None = None
    checkpoint_delta_ref: str | None = None
    materialized_bytes: int = 0
    dirty_since_checkpoint: bool = False
    prompt_reuse_hit: bool = False
    source_cache_key: str | None = None
    reuse_hits: int = 0
    reuse_misses: int = 0
    layout_key: str = "token_frames_contiguous"
    page_size_tokens: int = 1024
    page_count: int = 0
    paged_state_key: str | None = None
    paged_state_snapshot: dict[str, Any] | None = None
    transfer_plan: TransferPlan | None = None
    transfer_fast_path: dict[str, Any] | None = None
    residency_records: list[ResidencyRecord] = field(default_factory=list)


@dataclass(slots=True)
class GenieStatePage:
    """One page in a paged token-state store."""

    page_id: int
    token_start: int
    token_end: int
    frame_start: int
    frame_end: int
    bytes_size: int
    resident_tier: GenieResidencyTier = GenieResidencyTier.COLD_FILE
    dirty: bool = False
    access_count: int = 0
    last_accessed_at: float = 0.0
    last_updated_at: float = 0.0
    prefetch_count: int = 0
    eviction_count: int = 0
    last_transfer_kind: str | None = None
    last_transfer_async: bool = False
    last_transfer_at: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "bytes_size": self.bytes_size,
            "resident_tier": self.resident_tier.value,
            "dirty": self.dirty,
            "access_count": self.access_count,
            "last_accessed_at": self.last_accessed_at,
            "last_updated_at": self.last_updated_at,
            "prefetch_count": self.prefetch_count,
            "eviction_count": self.eviction_count,
            "last_transfer_kind": self.last_transfer_kind,
            "last_transfer_async": self.last_transfer_async,
            "last_transfer_at": self.last_transfer_at,
        }


@dataclass(slots=True)
class GeniePageTransferRequest:
    """Best-effort scaffold for a page-level host/device transfer."""

    transfer_id: int
    direction: str
    page_ids: list[int]
    bytes_size: int
    async_requested: bool
    status: str
    from_tier: GenieResidencyTier
    to_tier: GenieResidencyTier
    frame_window: tuple[int, int] | None
    reason: str
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "transfer_id": self.transfer_id,
            "direction": self.direction,
            "page_ids": list(self.page_ids),
            "bytes_size": self.bytes_size,
            "async_requested": self.async_requested,
            "status": self.status,
            "from_tier": self.from_tier.value,
            "to_tier": self.to_tier.value,
            "frame_window": list(self.frame_window) if self.frame_window is not None else None,
            "reason": self.reason,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


@dataclass(slots=True)
class GeniePagedStateStore:
    """Page/block-oriented token state store for Genie runtime work."""

    store_id: str
    total_frames: int
    spatial_h: int
    spatial_w: int
    page_size_tokens: int
    dtype: str
    pages: list[GenieStatePage]
    page_buffers: dict[int, np.ndarray]
    gpu_page_buffers: dict[int, Any] = field(default_factory=dict)
    gpu_pool_device: str | None = None
    gpu_pool_emulated: bool = False
    gpu_hot_page_limit: int = 8
    resident_tier: GenieResidencyTier = GenieResidencyTier.COLD_FILE
    hot_page_ids: set[int] = field(default_factory=set)
    dirty_page_ids: set[int] = field(default_factory=set)
    transfer_history: list[GeniePageTransferRequest] = field(default_factory=list)
    pending_transfer_ids: list[int] = field(default_factory=list)
    next_transfer_id: int = 0
    reclaimed_transfer_count: int = 0
    touch_count: int = 0
    write_count: int = 0
    last_page_span: tuple[int, int] | None = None
    last_frame_window: tuple[int, int] | None = None
    last_transfer_fast_path: dict[str, Any] | None = None

    @classmethod
    def from_tokens(
        cls,
        *,
        store_id: str,
        tokens: np.ndarray,
        page_size_tokens: int = 1024,
    ) -> "GeniePagedStateStore":
        tokens_np = np.asarray(tokens, dtype=np.uint32)
        if tokens_np.ndim != 3:
            raise ValueError("GeniePagedStateStore requires a [T,H,W] token tensor")
        flat = tokens_np.reshape(-1).copy()
        page_size, _page_count = page_layout(flat.size, page_size_tokens)
        frame_size = int(tokens_np.shape[1]) * int(tokens_np.shape[2])
        pages: list[GenieStatePage] = []
        page_buffers: dict[int, np.ndarray] = {}
        for page_id, token_start in enumerate(range(0, flat.size, page_size)):
            token_end = min(token_start + page_size, flat.size)
            page_buffer = flat[token_start:token_end].copy()
            pages.append(
                GenieStatePage(
                    page_id=page_id,
                    token_start=token_start,
                    token_end=token_end,
                    frame_start=int(token_start // frame_size),
                    frame_end=int(math.ceil(token_end / frame_size)),
                    bytes_size=int(page_buffer.nbytes),
                    resident_tier=GenieResidencyTier.WARM_PINNED_CPU,
                )
            )
            page_buffers[page_id] = page_buffer
        return cls(
            store_id=store_id,
            total_frames=int(tokens_np.shape[0]),
            spatial_h=int(tokens_np.shape[1]),
            spatial_w=int(tokens_np.shape[2]),
            page_size_tokens=page_size,
            dtype=str(tokens_np.dtype),
            pages=pages,
            page_buffers=page_buffers,
            resident_tier=GenieResidencyTier.WARM_PINNED_CPU,
        )

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def bytes_size(self) -> int:
        return int(sum(buffer.nbytes for buffer in self.page_buffers.values()))

    @property
    def physical_bytes(self) -> int:
        gpu_bytes = 0
        for buffer in self.gpu_page_buffers.values():
            if torch is not None and isinstance(buffer, torch.Tensor):
                gpu_bytes += int(buffer.numel() * buffer.element_size())
            else:
                gpu_bytes += int(np.asarray(buffer, dtype=np.uint32).nbytes)
        return self.bytes_size + gpu_bytes

    def _gpu_available(self) -> bool:
        return bool(torch is not None and torch.cuda.is_available())

    def _gpu_buffer_bytes(self, page_id: int) -> int:
        buffer = self.gpu_page_buffers.get(page_id)
        if buffer is None:
            return 0
        if torch is not None and isinstance(buffer, torch.Tensor):
            return int(buffer.numel() * buffer.element_size())
        return int(np.asarray(buffer, dtype=np.uint32).nbytes)

    def _record_transfer(
        self,
        *,
        direction: str,
        page_ids: list[int],
        async_requested: bool,
        from_tier: GenieResidencyTier,
        to_tier: GenieResidencyTier,
        frame_window: tuple[int, int] | None,
        reason: str,
    ) -> GeniePageTransferRequest:
        completed = (not async_requested) or len(page_ids) == 0
        request = GeniePageTransferRequest(
            transfer_id=self.next_transfer_id,
            direction=direction,
            page_ids=list(page_ids),
            bytes_size=sum(self.pages[page_id].bytes_size for page_id in page_ids),
            async_requested=async_requested,
            status="completed" if completed else "pending",
            from_tier=from_tier,
            to_tier=to_tier,
            frame_window=frame_window,
            reason=reason,
            completed_at=time.time() if completed else 0.0,
        )
        self.next_transfer_id += 1
        self.transfer_history.append(request)
        if not completed:
            self.pending_transfer_ids.append(request.transfer_id)
        return request

    def poll_transfers(self, *, max_to_complete: int | None = None) -> list[dict[str, Any]]:
        completed: list[dict[str, Any]] = []
        limit = len(self.pending_transfer_ids) if max_to_complete is None else max(int(max_to_complete), 0)
        transfer_by_id = {request.transfer_id: request for request in self.transfer_history}
        while self.pending_transfer_ids and len(completed) < limit:
            transfer_id = self.pending_transfer_ids.pop(0)
            request = transfer_by_id.get(transfer_id)
            if request is None or request.status != "pending":
                continue
            request.status = "completed"
            request.completed_at = time.time()
            completed.append(request.as_dict())
        return completed

    def reclaim_completed_transfers(self, *, max_to_reclaim: int | None = None) -> list[int]:
        reclaimed: list[int] = []
        limit = len(self.transfer_history) if max_to_reclaim is None else max(int(max_to_reclaim), 0)
        for request in self.transfer_history:
            if len(reclaimed) >= limit:
                break
            if request.status != "completed":
                continue
            request.status = "reclaimed"
            reclaimed.append(request.transfer_id)
        self.reclaimed_transfer_count += len(reclaimed)
        return reclaimed

    def _prefetch_page(self, page_id: int, *, async_requested: bool, reason: str) -> None:
        page = self.pages[page_id]
        page.prefetch_count += 1
        page.last_transfer_kind = "host_to_gpu"
        page.last_transfer_async = async_requested
        page.last_transfer_at = time.time()
        page.resident_tier = GenieResidencyTier.HOT_GPU
        self._ensure_gpu_page(page_id)
        self.hot_page_ids.add(page_id)
        self.gpu_pool_device = "cuda" if self._gpu_available() else "cpu-emulated"
        self.gpu_pool_emulated = not self._gpu_available()

    def _evict_page(self, page_id: int, *, async_requested: bool, reason: str) -> None:
        page = self.pages[page_id]
        page.eviction_count += 1
        page.last_transfer_kind = "gpu_to_host"
        page.last_transfer_async = async_requested
        page.last_transfer_at = time.time()
        page.resident_tier = GenieResidencyTier.WARM_PINNED_CPU
        self._drop_gpu_page(page_id)
        self.hot_page_ids.discard(page_id)

    def _trim_hot_pages(
        self,
        *,
        keep_page_ids: set[int],
        async_requested: bool,
        reason: str,
    ) -> list[int]:
        if self.gpu_hot_page_limit <= 0:
            return []
        evicted: list[int] = []
        while len(self.hot_page_ids) > self.gpu_hot_page_limit:
            candidates = [page_id for page_id in self.hot_page_ids if page_id not in keep_page_ids]
            if not candidates:
                break
            page_id = min(
                candidates,
                key=lambda candidate: self.pages[candidate].last_accessed_at or self.pages[candidate].last_transfer_at,
            )
            self._evict_page(page_id, async_requested=async_requested, reason=reason)
            evicted.append(page_id)
        return evicted

    def _ensure_gpu_page(self, page_id: int) -> None:
        if page_id in self.gpu_page_buffers:
            return
        host_buffer = self.page_buffers[page_id]
        if self._gpu_available():
            assert torch is not None
            gpu_buffer = torch.from_numpy(host_buffer.astype(np.int32, copy=True)).to("cuda", non_blocking=True)
            self.gpu_page_buffers[page_id] = gpu_buffer
            self.gpu_pool_device = "cuda"
            self.gpu_pool_emulated = False
            return
        self.gpu_page_buffers[page_id] = host_buffer.copy()
        self.gpu_pool_device = "cpu-emulated"
        self.gpu_pool_emulated = True

    def _drop_gpu_page(self, page_id: int) -> None:
        self.gpu_page_buffers.pop(page_id, None)

    def _page_numpy(self, page_id: int, *, prefer_gpu: bool = False) -> np.ndarray:
        if prefer_gpu and page_id in self.gpu_page_buffers:
            buffer = self.gpu_page_buffers[page_id]
            if torch is not None and isinstance(buffer, torch.Tensor):
                return buffer.detach().cpu().numpy().astype(np.uint32, copy=False)
            return np.asarray(buffer, dtype=np.uint32)
        return self.page_buffers[page_id]

    def transfer_queue_snapshot(self) -> dict[str, Any]:
        completed = sum(1 for request in self.transfer_history if request.status == "completed")
        reclaimed = sum(1 for request in self.transfer_history if request.status == "reclaimed")
        return {
            "pending": len(self.pending_transfer_ids),
            "completed": completed,
            "reclaimed": reclaimed,
            "history": len(self.transfer_history),
        }

    def materialize(self) -> np.ndarray:
        flat = np.empty(self.total_frames * self.spatial_h * self.spatial_w, dtype=np.uint32)
        for page in self.pages:
            flat[page.token_start:page.token_end] = self._page_numpy(page.page_id, prefer_gpu=True)
        return flat.reshape(self.total_frames, self.spatial_h, self.spatial_w).copy()

    def prefetch_pages(
        self,
        page_ids: list[int],
        *,
        frame_window: tuple[int, int] | None = None,
        async_requested: bool = True,
        reason: str = "prefetch",
    ) -> dict[str, Any]:
        normalized = [page_id for page_id in dict.fromkeys(page_ids) if 0 <= page_id < len(self.pages)]
        for page_id in normalized:
            self._prefetch_page(page_id, async_requested=async_requested, reason=reason)
        evicted = self._trim_hot_pages(
            keep_page_ids=set(normalized),
            async_requested=async_requested,
            reason=f"{reason}:evict",
        )
        transfer = self._record_transfer(
            direction="host_to_gpu",
            page_ids=normalized,
            async_requested=async_requested,
            from_tier=GenieResidencyTier.WARM_PINNED_CPU,
            to_tier=GenieResidencyTier.HOT_GPU,
            frame_window=frame_window,
            reason=reason,
        )
        return {
            "transfer": transfer.as_dict(),
            "prefetched_page_ids": normalized,
            "evicted_page_ids": evicted,
            "hot_page_ids": sorted(self.hot_page_ids),
            "gpu_page_ids": sorted(self.gpu_page_buffers.keys()),
            "hot_page_limit": self.gpu_hot_page_limit,
            "transfer_queue": self.transfer_queue_snapshot(),
        }

    def evict_pages(
        self,
        page_ids: list[int],
        *,
        frame_window: tuple[int, int] | None = None,
        async_requested: bool = True,
        reason: str = "evict",
    ) -> dict[str, Any]:
        normalized = [page_id for page_id in dict.fromkeys(page_ids) if 0 <= page_id < len(self.pages)]
        for page_id in normalized:
            self._evict_page(page_id, async_requested=async_requested, reason=reason)
        transfer = self._record_transfer(
            direction="gpu_to_host",
            page_ids=normalized,
            async_requested=async_requested,
            from_tier=GenieResidencyTier.HOT_GPU,
            to_tier=GenieResidencyTier.WARM_PINNED_CPU,
            frame_window=frame_window,
            reason=reason,
        )
        return {
            "transfer": transfer.as_dict(),
            "evicted_page_ids": normalized,
            "hot_page_ids": sorted(self.hot_page_ids),
            "gpu_page_ids": sorted(self.gpu_page_buffers.keys()),
            "hot_page_limit": self.gpu_hot_page_limit,
            "transfer_queue": self.transfer_queue_snapshot(),
        }

    def page_span_for_frames(self, frame_start: int, frame_end: int) -> tuple[int, int]:
        frame_size = self.spatial_h * self.spatial_w
        token_start = max(frame_start, 0) * frame_size
        token_end = min(frame_end, self.total_frames) * frame_size
        if token_end <= token_start:
            return (0, 0)
        first_page = token_start // self.page_size_tokens
        last_page = (token_end - 1) // self.page_size_tokens
        return (int(first_page), int(last_page))

    def page_ids_for_frames(self, frame_start: int, frame_end: int) -> list[int]:
        safe_start = max(min(int(frame_start), self.total_frames), 0)
        safe_end = max(min(int(frame_end), self.total_frames), safe_start)
        if safe_end <= safe_start:
            return []
        first_page, last_page = self.page_span_for_frames(safe_start, safe_end)
        if last_page < first_page:
            return []
        return list(range(first_page, last_page + 1))

    def prefetch_window(
        self,
        *,
        frame_start: int,
        frame_end: int,
        async_requested: bool = True,
    ) -> dict[str, Any]:
        safe_start = max(min(int(frame_start), self.total_frames), 0)
        safe_end = max(min(int(frame_end), self.total_frames), safe_start)
        page_ids = self.page_ids_for_frames(safe_start, safe_end)
        page_window = self.prefetch_pages(
            page_ids,
            frame_window=(safe_start, safe_end),
            async_requested=async_requested,
            reason="prefetch_window",
        )
        page_window["frame_window"] = [safe_start, safe_end]
        return page_window

    def evict_window(
        self,
        *,
        frame_start: int,
        frame_end: int,
        async_requested: bool = True,
    ) -> dict[str, Any]:
        safe_start = max(min(int(frame_start), self.total_frames), 0)
        safe_end = max(min(int(frame_end), self.total_frames), safe_start)
        page_ids = self.page_ids_for_frames(safe_start, safe_end)
        page_window = self.evict_pages(
            page_ids,
            frame_window=(safe_start, safe_end),
            async_requested=async_requested,
            reason="evict_window",
        )
        page_window["frame_window"] = [safe_start, safe_end]
        return page_window

    def window_tokens(self, frame_start: int, frame_end: int) -> np.ndarray:
        safe_start = max(min(int(frame_start), self.total_frames), 0)
        safe_end = max(min(int(frame_end), self.total_frames), safe_start)
        if safe_end <= safe_start:
            return np.empty((0, self.spatial_h, self.spatial_w), dtype=np.uint32)
        frame_size = self.spatial_h * self.spatial_w
        token_start = safe_start * frame_size
        token_end = safe_end * frame_size
        flat = np.empty(token_end - token_start, dtype=np.uint32)
        offset = 0
        for page_id in self.page_ids_for_frames(safe_start, safe_end):
            page = self.pages[page_id]
            overlap_start = max(token_start, page.token_start)
            overlap_end = min(token_end, page.token_end)
            if overlap_end <= overlap_start:
                continue
            local_start = overlap_start - page.token_start
            local_end = overlap_end - page.token_start
            slice_len = overlap_end - overlap_start
            flat[offset : offset + slice_len] = self._page_numpy(page_id, prefer_gpu=True)[local_start:local_end]
            offset += slice_len
        return flat.reshape(safe_end - safe_start, self.spatial_h, self.spatial_w)

    def touch_window(
        self,
        *,
        frame_start: int,
        frame_end: int,
        resident_tier: GenieResidencyTier,
        mark_dirty: bool = False,
        transfer_fast_path: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        page_ids = self.page_ids_for_frames(frame_start, frame_end)
        safe_start = max(min(int(frame_start), self.total_frames), 0)
        safe_end = max(min(int(frame_end), self.total_frames), safe_start)
        now = time.time()
        async_requested = bool(transfer_fast_path and transfer_fast_path.get("non_blocking_h2d", False))
        if resident_tier == GenieResidencyTier.HOT_GPU:
            page_transfer = self.prefetch_pages(
                page_ids,
                frame_window=(safe_start, safe_end),
                async_requested=async_requested,
                reason="touch_window_prefetch",
            )
        else:
            page_transfer = self.evict_pages(
                page_ids,
                frame_window=(safe_start, safe_end),
                async_requested=async_requested,
                reason="touch_window_evict",
            )
        touched_bytes = 0
        now = time.time()
        for page_id in page_ids:
            page = self.pages[page_id]
            page.resident_tier = resident_tier
            page.access_count += 1
            page.last_accessed_at = now
            touched_bytes += page.bytes_size
            if mark_dirty:
                page.dirty = True
                page.last_updated_at = now
                self.dirty_page_ids.add(page_id)

        self.resident_tier = resident_tier
        self.touch_count += 1
        if mark_dirty:
            self.write_count += 1
        self.last_page_span = (page_ids[0], page_ids[-1]) if page_ids else None
        self.last_frame_window = (safe_start, safe_end)
        self.last_transfer_fast_path = transfer_fast_path
        return {
            "frame_window": [safe_start, safe_end],
            "page_span": list(self.last_page_span) if self.last_page_span is not None else None,
            "page_ids": page_ids,
            "pages_touched": len(page_ids),
            "page_bytes": touched_bytes,
            "resident_tier": resident_tier.value,
            "dirty_pages": sorted(self.dirty_page_ids),
            "hot_pages": sorted(self.hot_page_ids),
            "page_transfer": page_transfer,
        }

    def update_window(
        self,
        *,
        frame_start: int,
        frame_end: int,
        tokens: np.ndarray,
        resident_tier: GenieResidencyTier = GenieResidencyTier.HOT_GPU,
        transfer_fast_path: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        tokens_np = np.asarray(tokens, dtype=np.uint32)
        if tokens_np.ndim != 3:
            raise ValueError("GeniePagedStateStore.update_window requires a [T,H,W] token tensor")
        frame_size = self.spatial_h * self.spatial_w
        token_start = max(frame_start, 0) * frame_size
        token_end = min(frame_end, self.total_frames) * frame_size
        expected_tokens = max(token_end - token_start, 0)
        replacement = tokens_np.reshape(-1)
        if replacement.size != expected_tokens:
            raise ValueError(
                f"GeniePagedStateStore.update_window expected {expected_tokens} tokens, got {replacement.size}"
            )
        replacement_offset = 0
        touched_page_ids = self.page_ids_for_frames(frame_start, frame_end)
        for page_id in touched_page_ids:
            page = self.pages[page_id]
            overlap_start = max(token_start, page.token_start)
            overlap_end = min(token_end, page.token_end)
            if overlap_end <= overlap_start:
                continue
            local_start = overlap_start - page.token_start
            local_end = overlap_end - page.token_start
            replacement_len = overlap_end - overlap_start
            self.page_buffers[page_id][local_start:local_end] = replacement[
                replacement_offset : replacement_offset + replacement_len
            ]
            if page_id in self.gpu_page_buffers:
                gpu_replacement = replacement[replacement_offset : replacement_offset + replacement_len]
                gpu_buffer = self.gpu_page_buffers[page_id]
                if torch is not None and isinstance(gpu_buffer, torch.Tensor):
                    gpu_buffer[local_start:local_end] = torch.from_numpy(gpu_replacement.astype(np.int32, copy=False)).to(
                        gpu_buffer.device,
                        non_blocking=True,
                    )
                else:
                    gpu_buffer[local_start:local_end] = gpu_replacement
            replacement_offset += replacement_len
        return self.touch_window(
            frame_start=frame_start,
            frame_end=frame_end,
            resident_tier=resident_tier,
            mark_dirty=True,
            transfer_fast_path=transfer_fast_path,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "store_id": self.store_id,
            "total_frames": self.total_frames,
            "spatial": [self.spatial_h, self.spatial_w],
            "page_size_tokens": self.page_size_tokens,
            "page_count": self.page_count,
            "bytes_size": self.bytes_size,
            "physical_bytes": self.physical_bytes,
            "dtype": self.dtype,
            "resident_tier": self.resident_tier.value,
            "hot_page_ids": sorted(self.hot_page_ids),
            "dirty_page_ids": sorted(self.dirty_page_ids),
            "touch_count": self.touch_count,
            "write_count": self.write_count,
            "last_page_span": list(self.last_page_span) if self.last_page_span is not None else None,
            "last_frame_window": list(self.last_frame_window) if self.last_frame_window is not None else None,
            "last_transfer_fast_path": self.last_transfer_fast_path,
            "page_pool": {
                "logical_bytes": self.bytes_size,
                "physical_bytes": self.physical_bytes,
                "hot_page_count": len(self.hot_page_ids),
                "dirty_page_count": len(self.dirty_page_ids),
                "hot_page_limit": self.gpu_hot_page_limit,
                "host_pool": {
                    "page_count": len(self.page_buffers),
                    "bytes": self.bytes_size,
                    "resident_tier": GenieResidencyTier.WARM_PINNED_CPU.value,
                },
                "gpu_pool": {
                    "page_count": len(self.gpu_page_buffers),
                    "bytes": sum(self._gpu_buffer_bytes(page_id) for page_id in self.gpu_page_buffers),
                    "device": self.gpu_pool_device,
                    "emulated": self.gpu_pool_emulated,
                },
            },
            "transfer_queue": self.transfer_queue_snapshot(),
            "transfer_history": [request.as_dict() for request in self.transfer_history],
            "pages": [page.as_dict() for page in self.pages],
        }


@dataclass(slots=True)
class GenieExecutionChunk:
    """A stage-local homogeneous chunk sent to a Genie worker."""

    chunk_id: str
    signature: GenieBatchSignature
    entity_ids: list[str]
    runnable_stage: str
    frame_range: tuple[int, int]
    estimated_vram_bytes: int
    estimated_transfer_bytes: int
    estimated_flops: float
    queue_lane: GenieQueueLane
    expected_occupancy: float

    @property
    def size(self) -> int:
        return len(self.entity_ids)

    @property
    def fill_ratio(self) -> float:
        return self.expected_occupancy

    @property
    def scheduler_score(self) -> float:
        return self.expected_occupancy - (self.estimated_transfer_bytes / max(self.estimated_vram_bytes, 1))


@dataclass(slots=True)
class GenieStageRecord:
    """One stage execution sample for runtime introspection."""

    stage: str
    entity_id: str
    queue_lane: str
    started_at: float
    elapsed_ms: float
    chunk_id: str | None = None
    chunk_size: int | None = None
    expected_occupancy: float | None = None
    estimated_transfer_bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenieRuntimeTrace:
    """Per-request runtime trace for stage-local profiling and debugging."""

    records: list[GenieStageRecord] = field(default_factory=list)
    queue_lanes_seen: list[str] = field(default_factory=list)
    chunk_signatures: list[dict[str, Any]] = field(default_factory=list)

    def record(self, record: GenieStageRecord) -> None:
        self.records.append(record)
        if record.queue_lane not in self.queue_lanes_seen:
            self.queue_lanes_seen.append(record.queue_lane)

    def stage_timings_ms(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for record in self.records:
            totals[record.stage] = round(totals.get(record.stage, 0.0) + record.elapsed_ms, 3)
        return totals

    def chunk_summary(self) -> dict[str, Any]:
        chunk_count = 0
        chunk_sizes: list[int] = []
        occupancy: list[float] = []
        transfer_bytes = 0
        for record in self.records:
            if record.chunk_id is None:
                continue
            chunk_count += 1
            if record.chunk_size is not None:
                chunk_sizes.append(record.chunk_size)
            if record.expected_occupancy is not None:
                occupancy.append(record.expected_occupancy)
            if record.estimated_transfer_bytes is not None:
                transfer_bytes += record.estimated_transfer_bytes
        return {
            "chunk_count": chunk_count,
            "avg_chunk_size": (sum(chunk_sizes) / len(chunk_sizes)) if chunk_sizes else 0.0,
            "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
            "avg_expected_occupancy": (sum(occupancy) / len(occupancy)) if occupancy else 0.0,
            "estimated_transfer_bytes": transfer_bytes,
        }


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def build_stage_profile(
    stage_history: list[dict[str, Any]],
    stage_timings_ms: dict[str, float],
) -> dict[str, Any]:
    """Summarize stage execution into a stable profiling payload."""

    stages: dict[str, dict[str, Any]] = {}
    completed_stages: list[str] = []
    for stage in GENIE_STAGE_GRAPH:
        entries = [entry for entry in stage_history if entry.get("stage") == stage]
        queue_lanes = _ordered_unique(
            [str(entry["queue_lane"]) for entry in entries if entry.get("queue_lane") is not None]
        )
        runner_modes = _ordered_unique(
            [str(entry["runner_mode"]) for entry in entries if entry.get("runner_mode") is not None]
        )
        chunk_sizes = [int(entry.get("chunk_size") or 0) for entry in entries if entry.get("chunk_size") is not None]
        frame_ranges = [list(entry["frame_range"]) for entry in entries if entry.get("frame_range") is not None]
        stage_elapsed_ms = round(
            float(stage_timings_ms.get(f"{stage}_ms", sum(float(entry.get("elapsed_ms", 0.0)) for entry in entries))),
            3,
        )
        stages[stage] = {
            "count": len(entries),
            "elapsed_ms": stage_elapsed_ms,
            "queue_lanes": queue_lanes,
            "runner_modes": runner_modes,
            "chunk_count": sum(1 for entry in entries if entry.get("chunk_id") is not None),
            "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
            "frame_ranges": frame_ranges,
        }
        if entries:
            completed_stages.append(stage)
    return {
        "graph": list(GENIE_STAGE_GRAPH),
        "completed_stages": completed_stages,
        "stage_count": len(completed_stages),
        "total_elapsed_ms": round(float(stage_timings_ms.get("total_elapsed_ms", 0.0)), 3),
        "stages": stages,
    }


def build_scheduler_profile(
    *,
    execution_path: str,
    transition_entities: int,
    chunks: list[dict[str, Any]],
    scheduler_inputs: list[dict[str, Any]],
    observed_batch_sizes: list[int],
    batched_across_requests: bool,
    cross_request_batcher: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a scheduler-facing profile payload shared by single and batch paths."""

    queue_lanes = _ordered_unique(
        [str(chunk["queue_lane"]) for chunk in chunks if chunk.get("queue_lane") is not None]
    )
    occupancies = [float(chunk.get("expected_occupancy") or 0.0) for chunk in chunks]
    chunk_sizes = [int(chunk.get("chunk_size") or 0) for chunk in chunks]
    return {
        "execution_path": execution_path,
        "transition_entities": transition_entities,
        "chunk_count": len(chunks),
        "chunks": chunks,
        "queue_lanes": queue_lanes,
        "scheduler_inputs": scheduler_inputs,
        "observed_batch_sizes": observed_batch_sizes,
        "batched_across_requests": batched_across_requests,
        "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
        "max_observed_batch_size": max(observed_batch_sizes) if observed_batch_sizes else 1,
        "avg_expected_occupancy": round(sum(occupancies) / len(occupancies), 4) if occupancies else 0.0,
        "cross_request_batcher": cross_request_batcher,
    }


def build_runtime_state_profile(runtime_state: GenieRuntimeState) -> dict[str, Any]:
    """Expose the hot runtime state in a stable, JSON-friendly shape."""

    return {
        "resident_tier": runtime_state.resident_tier.value,
        "materialized_bytes": runtime_state.materialized_bytes,
        "reuse_hits": runtime_state.reuse_hits,
        "reuse_misses": runtime_state.reuse_misses,
        "last_completed_frame": runtime_state.last_completed_frame,
        "checkpoint_delta_ref": runtime_state.checkpoint_delta_ref,
        "dirty_since_checkpoint": runtime_state.dirty_since_checkpoint,
        "source_cache_key": runtime_state.source_cache_key,
        "layout_key": runtime_state.layout_key,
        "page_size_tokens": runtime_state.page_size_tokens,
        "page_count": runtime_state.page_count,
        "paged_state_key": runtime_state.paged_state_key,
        "paged_state": runtime_state.paged_state_snapshot,
        "transfer_plan": None if runtime_state.transfer_plan is None else runtime_state.transfer_plan.as_dict(),
        "transfer_fast_path": runtime_state.transfer_fast_path,
        "residency": [record.as_dict() for record in runtime_state.residency_records],
    }


def build_benchmark_profile(
    *,
    stage_timings_ms: dict[str, float],
    scheduler_profile: dict[str, Any],
) -> dict[str, Any]:
    """Emit the runtime fields most useful for benchmark-gate comparisons."""

    return {
        "state_token_prep_ms": round(float(stage_timings_ms.get("state_token_prep_ms", 0.0)), 3),
        "transition_ms": round(float(stage_timings_ms.get("transition_ms", 0.0)), 3),
        "checkpoint_ms": round(float(stage_timings_ms.get("checkpoint_ms", 0.0)), 3),
        "artifact_persist_ms": round(float(stage_timings_ms.get("artifact_persist_ms", 0.0)), 3),
        "controlplane_commit_ms": round(float(stage_timings_ms.get("controlplane_commit_ms", 0.0)), 3),
        "total_elapsed_ms": round(float(stage_timings_ms.get("total_elapsed_ms", 0.0)), 3),
        "chunk_count": int(scheduler_profile.get("chunk_count", 0)),
        "max_chunk_size": int(scheduler_profile.get("max_chunk_size", 0)),
        "max_observed_batch_size": int(scheduler_profile.get("max_observed_batch_size", 1)),
        "avg_expected_occupancy": float(scheduler_profile.get("avg_expected_occupancy", 0.0)),
        "batched_across_requests": bool(scheduler_profile.get("batched_across_requests", False)),
    }


@dataclass(slots=True)
class CachedPromptState:
    """Best-effort reusable prompt state entry."""

    cache_key: str
    tokens: np.ndarray
    source_state_handle_id: str | None
    resident_tier: GenieResidencyTier
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    reuse_count: int = 0

    @property
    def memory_bytes(self) -> int:
        return int(self.tokens.nbytes)


def build_genie_execution_family(
    signature: GenieBatchSignature,
    *,
    batch_size: int,
) -> ExecutionFamily:
    return ExecutionFamily(
        backend=signature.backend,
        model=signature.model_name,
        stage=signature.stage,
        device=signature.device,
        dtype=signature.dtype,
        runner_mode=signature.runner_mode,
        batch_size_family=batch_size_family(batch_size),
        width=signature.spatial_w,
        height=signature.spatial_h,
        frame_count=signature.window_num_frames,
        num_steps=signature.maskgit_steps,
        prompt_frames=signature.num_prompt_frames,
        memory_mode="persist" if signature.needs_persist else "runtime_hot",
        layout_key=f"token_frames:{signature.spatial_h}x{signature.spatial_w}",
        execution_kind="temporal_rollout",
        tokenizer_kind=signature.tokenizer_kind,
    )


def generic_residency_tier(tier: GenieResidencyTier) -> ResidencyTier:
    mapping = {
        GenieResidencyTier.HOT_GPU: ResidencyTier.GPU_HOT,
        GenieResidencyTier.WARM_PINNED_CPU: ResidencyTier.CPU_PINNED_WARM,
        GenieResidencyTier.COLD_FILE: ResidencyTier.DURABLE_ONLY,
    }
    return mapping[tier]


def page_layout(token_count: int, page_size_tokens: int = 1024) -> tuple[int, int]:
    page_size = max(int(page_size_tokens), 1)
    pages = int(math.ceil(max(int(token_count), 0) / page_size))
    return page_size, pages


def build_genie_transfer_plan(
    *,
    materialized_bytes: int,
    checkpoint_bytes: int = 0,
    artifact_io_bytes: int = 0,
    prompt_state_hot: bool = False,
    host_staging: dict[str, Any] | None = None,
) -> TransferPlan:
    plan = TransferPlan(
        overlap_h2d_with_compute=not prompt_state_hot,
        overlap_d2h_with_io=True,
        staging_tier=ResidencyTier.CPU_PINNED_WARM,
        notes=[
            "Genie fast path prefers warm pinned prompt state and hot GPU transition windows.",
        ],
    )
    plan.add_h2d(materialized_bytes)
    plan.add_artifact_io(checkpoint_bytes + artifact_io_bytes)
    if host_staging:
        plan.staging_bytes = max(int(host_staging.get("bytes", 0)), 0)
        if host_staging.get("pinned"):
            plan.notes.append("Pinned host staging reserved for prompt/state materialization.")
        if host_staging.get("non_blocking_h2d"):
            plan.notes.append("Prompt H2D path is eligible for non-blocking transfer.")
    return plan


def build_genie_transfer_fast_path(
    *,
    paged_state: GeniePagedStateStore | None,
    host_staging: dict[str, Any] | None,
    frame_start: int | None,
    frame_end: int | None,
    resident_tier: GenieResidencyTier,
) -> dict[str, Any]:
    page_window: dict[str, Any] | None = None
    if paged_state is not None and frame_start is not None and frame_end is not None:
        page_ids = paged_state.page_ids_for_frames(frame_start, frame_end)
        page_bytes = sum(paged_state.pages[page_id].bytes_size for page_id in page_ids)
        async_requested = bool(host_staging and host_staging.get("non_blocking_h2d"))
        keep_page_ids = set(page_ids)
        eviction_candidates = sorted(
            [page_id for page_id in paged_state.hot_page_ids if page_id not in keep_page_ids],
            key=lambda candidate: (
                paged_state.pages[candidate].last_accessed_at or paged_state.pages[candidate].last_transfer_at,
                candidate,
            ),
        )
        evict_page_ids = eviction_candidates[: max(len(paged_state.hot_page_ids) - paged_state.gpu_hot_page_limit, 0)]
        page_window = {
            "frame_window": [max(int(frame_start), 0), max(int(frame_end), 0)],
            "page_span": [page_ids[0], page_ids[-1]] if page_ids else None,
            "page_ids": page_ids,
            "pages_touched": len(page_ids),
            "page_bytes": page_bytes,
            "store_id": paged_state.store_id,
            "hot_page_count": len(paged_state.hot_page_ids),
            "hot_page_limit": paged_state.gpu_hot_page_limit,
            "prefetch_page_ids": page_ids,
            "evict_page_ids": evict_page_ids,
            "transfer_direction": "host_to_gpu" if resident_tier == GenieResidencyTier.HOT_GPU else "gpu_to_host",
            "async_transfer": async_requested,
            "transfer_mode": "scaffolded_async" if async_requested else "inline",
            "transfer_queue": paged_state.transfer_queue_snapshot(),
        }
    return {
        "host_staging": host_staging,
        "page_window": page_window,
        "resident_tier": resident_tier.value,
        "pool_path": "host_to_gpu" if resident_tier == GenieResidencyTier.HOT_GPU else "host_only",
        "non_blocking_h2d": bool(host_staging and host_staging.get("non_blocking_h2d")),
        "pinned_host_staging": bool(host_staging and host_staging.get("pinned")),
        "async_transfer": bool(host_staging and host_staging.get("non_blocking_h2d")),
        "prefetch_strategy": "page_window_lru",
        "transfer_queue": None if paged_state is None else paged_state.transfer_queue_snapshot(),
    }


def build_genie_residency_records(
    *,
    rollout_id: str,
    resident_tier: GenieResidencyTier,
    materialized_bytes: int,
    page_size_tokens: int,
    page_count: int,
    cache_key: str | None,
    host_staging: dict[str, Any] | None = None,
    state_store_key: str | None = None,
) -> list[ResidencyRecord]:
    records = [
        ResidencyRecord(
            object_id=f"{rollout_id}:prompt-state",
            tier=generic_residency_tier(resident_tier),
            bytes_size=max(materialized_bytes, 0),
            layout_key="token_frames_contiguous",
            pinned=resident_tier == GenieResidencyTier.WARM_PINNED_CPU,
            reusable=True,
            source=cache_key,
        ),
        ResidencyRecord(
            object_id=f"{rollout_id}:state-pages",
            tier=generic_residency_tier(resident_tier),
            bytes_size=max(materialized_bytes, 0),
            layout_key=f"paged_tokens:{page_size_tokens}",
            pinned=resident_tier == GenieResidencyTier.WARM_PINNED_CPU,
            reusable=True,
            source=state_store_key or f"pages:{page_count}",
        ),
    ]
    if host_staging and int(host_staging.get("bytes", 0)) > 0:
        records.append(
            ResidencyRecord(
                object_id=f"{rollout_id}:host-staging",
                tier=ResidencyTier.CPU_PINNED_WARM if host_staging.get("pinned") else ResidencyTier.CPU_WARM,
                bytes_size=max(int(host_staging.get("bytes", 0)), 0),
                layout_key="staging_prompt_tokens",
                pinned=bool(host_staging.get("pinned")),
                reusable=False,
                source=f"staging:{host_staging.get('alloc_id', 'ephemeral')}",
            )
        )
    return records


def _stable_token_hash(tokens: np.ndarray) -> str:
    return hashlib.sha256(tokens.tobytes()).hexdigest()


def prompt_cache_key(
    *,
    input_state_handle_id: str | None,
    input_tokens: np.ndarray | None,
    prompt: str,
    seed: int,
) -> str | None:
    """Build a stable cache key for prompt materialization."""

    if input_state_handle_id:
        return f"state_handle:{input_state_handle_id}"
    if input_tokens is not None:
        return f"tokens:{_stable_token_hash(np.asarray(input_tokens, dtype=np.uint32))}"
    if prompt:
        prompt_hash = hashlib.sha256(f"{prompt}:{seed}".encode("utf-8")).hexdigest()
        return f"prompt:{prompt_hash}"
    return None


class GeniePromptStateCache:
    """In-memory prompt state cache used for warm continuation and branch reuse."""

    def __init__(self, max_entries: int = 32):
        self.max_entries = max_entries
        self._entries: dict[str, CachedPromptState] = {}
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "bytes_hot": 0,
        }

    def get(self, cache_key: str | None) -> CachedPromptState | None:
        if cache_key is None:
            self._stats["misses"] += 1
            return None
        entry = self._entries.get(cache_key)
        if entry is None:
            self._stats["misses"] += 1
            return None
        entry.last_accessed = time.time()
        entry.reuse_count += 1
        self._stats["hits"] += 1
        return entry

    def put(
        self,
        cache_key: str | None,
        tokens: np.ndarray,
        *,
        source_state_handle_id: str | None,
        resident_tier: GenieResidencyTier = GenieResidencyTier.WARM_PINNED_CPU,
    ) -> CachedPromptState | None:
        if cache_key is None:
            return None
        if cache_key in self._entries:
            entry = self._entries[cache_key]
            self._stats["bytes_hot"] -= entry.memory_bytes
        else:
            self._evict_if_needed()
        entry = CachedPromptState(
            cache_key=cache_key,
            tokens=np.asarray(tokens, dtype=np.uint32).copy(),
            source_state_handle_id=source_state_handle_id,
            resident_tier=resident_tier,
        )
        self._entries[cache_key] = entry
        self._stats["bytes_hot"] += entry.memory_bytes
        return entry

    def promote(self, cache_key: str | None, tier: GenieResidencyTier) -> None:
        if cache_key is None:
            return
        entry = self._entries.get(cache_key)
        if entry is not None:
            entry.resident_tier = tier
            entry.last_accessed = time.time()

    def snapshot(self) -> dict[str, int]:
        return {
            "entries": len(self._entries),
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "evictions": self._stats["evictions"],
            "bytes_hot": self._stats["bytes_hot"],
        }

    def _evict_if_needed(self) -> None:
        if len(self._entries) < self.max_entries:
            return
        oldest_key = min(self._entries, key=lambda key: self._entries[key].last_accessed)
        oldest = self._entries.pop(oldest_key)
        self._stats["bytes_hot"] -= oldest.memory_bytes
        self._stats["evictions"] += 1


def default_window_num_frames(
    total_frames: int,
    num_prompt_frames: int,
    checkpoint_every_n_frames: int,
) -> int:
    """Choose a bounded frame-group size for one transition window."""

    remaining = max(total_frames - num_prompt_frames, 1)
    if checkpoint_every_n_frames > 0:
        return max(1, min(checkpoint_every_n_frames, remaining))
    return remaining


def frame_windows(
    *,
    total_frames: int,
    num_prompt_frames: int,
    checkpoint_every_n_frames: int,
) -> list[tuple[int, int]]:
    """Split the generation tail into bounded execution windows."""

    start = min(max(num_prompt_frames, 0), max(total_frames - 1, 0))
    if total_frames <= start:
        return []
    window = default_window_num_frames(total_frames, num_prompt_frames, checkpoint_every_n_frames)
    windows: list[tuple[int, int]] = []
    cursor = start
    while cursor < total_frames:
        size = min(window, total_frames - cursor)
        windows.append((cursor, size))
        cursor += size
    return windows


def temperature_bucket(temperature: float) -> str:
    """Bucket temperature for execution signature grouping."""

    if temperature <= 0:
        return "argmax"
    if temperature < 0.5:
        return "low"
    if temperature < 1.0:
        return "medium"
    return "high"


def estimate_transition_flops(
    *,
    spatial_h: int,
    spatial_w: int,
    window_num_frames: int,
    maskgit_steps: int,
) -> float:
    """Very coarse relative flops estimate for scheduling."""

    tokens = spatial_h * spatial_w * max(window_num_frames, 1)
    return float(tokens * max(maskgit_steps, 1))


def estimate_expected_occupancy(chunk_size: int, max_chunk_size: int) -> float:
    """Best-effort occupancy estimate derived from chunk fill."""

    if max_chunk_size <= 0:
        return 0.0
    return min(max(chunk_size / max_chunk_size, 0.0), 1.0)


def lane_priority(lane: GenieQueueLane) -> int:
    """Lower number means higher scheduling priority."""

    if isinstance(lane, str):
        lane = GenieQueueLane(lane)
    priorities = {
        GenieQueueLane.CHECKPOINT_HEAVY: 0,
        GenieQueueLane.HOT_CONTINUATION: 1,
        GenieQueueLane.COLD_MATERIALIZE: 2,
        GenieQueueLane.PERSIST_ONLY: 3,
    }
    return priorities[lane]


def make_stage_signature(
    *,
    backend: str,
    model_name: str,
    stage: str,
    device: str,
    dtype: str,
    tokenizer_kind: str,
    spatial_h: int,
    spatial_w: int,
    window_num_frames: int,
    num_prompt_frames: int,
    maskgit_steps: int,
    temperature: float,
    checkpoint_every_n_frames: int,
    runner_mode: str,
    needs_persist: bool,
) -> GenieBatchSignature:
    """Build a stable stage signature from Genie request shape."""

    return GenieBatchSignature(
        backend=backend,
        model_name=model_name,
        stage=stage,
        device=device,
        dtype=dtype,
        tokenizer_kind=tokenizer_kind,
        spatial_h=spatial_h,
        spatial_w=spatial_w,
        window_num_frames=window_num_frames,
        num_prompt_frames=num_prompt_frames,
        maskgit_steps=maskgit_steps,
        temperature_bucket=temperature_bucket(temperature),
        checkpoint_every_n_frames=checkpoint_every_n_frames,
        runner_mode=runner_mode,
        needs_persist=needs_persist,
    )


def build_transition_entities(root_entity: GenieExecutionEntity) -> list[GenieExecutionEntity]:
    """Split one Genie rollout tail into schedulable frame-window entities."""

    windows = frame_windows(
        total_frames=root_entity.total_frames,
        num_prompt_frames=root_entity.num_prompt_frames,
        checkpoint_every_n_frames=root_entity.checkpoint_every_n_frames,
    )
    if not windows:
        return []

    entities: list[GenieExecutionEntity] = []
    for index, (window_start_frame, window_num_frames) in enumerate(windows):
        window_end_frame = window_start_frame + window_num_frames
        queue_lane = root_entity.queue_lane
        if (
            root_entity.checkpoint_every_n_frames > 0
            and window_end_frame < root_entity.total_frames
            and window_end_frame % root_entity.checkpoint_every_n_frames == 0
        ):
            queue_lane = GenieQueueLane.CHECKPOINT_HEAVY.value
        entities.append(
            GenieExecutionEntity(
                entity_id=f"{root_entity.entity_id}:{window_end_frame:04d}",
                rollout_id=root_entity.rollout_id,
                episode_id=root_entity.episode_id,
                branch_id=root_entity.branch_id,
                sample_id=root_entity.sample_id,
                input_state_handle_id=root_entity.input_state_handle_id,
                current_stage=root_entity.current_stage,
                next_stage=root_entity.next_stage,
                window_start_frame=window_start_frame,
                window_num_frames=window_num_frames,
                total_frames=root_entity.total_frames,
                num_prompt_frames=root_entity.num_prompt_frames,
                checkpoint_every_n_frames=root_entity.checkpoint_every_n_frames,
                priority=root_entity.priority - (index * 0.001),
                deadline_s=root_entity.deadline_s,
                batch_signature=replace(root_entity.batch_signature, window_num_frames=window_num_frames),
                queue_lane=queue_lane,
                stage_attempts=root_entity.stage_attempts,
                last_scheduled_at=root_entity.last_scheduled_at,
            )
        )
    return entities
