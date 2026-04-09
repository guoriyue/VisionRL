"""Continuous batching scheduler for the unified engine.

Implements the core scheduling loop: on each iteration it ejects done
entities, tries to swap-in preempted ones, admits new requests up to the
block budget, builds encode/step batches, and preempts lowest-priority
entities when the block budget is exceeded.

Design modelled on vLLM's ``Scheduler`` but adapted for world-model
latent states rather than KV caches.

Also includes RolloutScheduler for higher-level rollout batching.
"""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Sequence

from wm_infra.config import SchedulerConfig, SchedulerPolicy
from wm_infra.engine.types import (
    DEFAULT_FRAME_COUNT,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    EngineRunConfig,
    EntityRequest,
    Phase,
    RolloutRequest,
    ScheduledBatch,
    SchedulerOutput,
)
from wm_infra.engine.mem_cache.paged_pool import PagedLatentPool
from wm_infra.engine.mem_cache.radix_cache import RadixStateCache


@dataclass(slots=True)
class EntityState:
    """Mutable per-entity bookkeeping inside the scheduler."""

    request: EntityRequest
    phase: Phase = Phase.WAITING
    step_index: int = 0
    blocks_needed: int = 1
    blocks_allocated: int = 0
    swapped_block_ids: tuple[int, ...] = ()
    prefix_matched: int = 0


class ContinuousBatchingScheduler:
    """Block-budget continuous batching scheduler.

    Parameters
    ----------
    config : EngineRunConfig
        Engine configuration (block counts, batch size, etc.).
    pool : PagedLatentPool
        The paged latent pool that backs block allocation.
    cache : RadixStateCache | None
        Optional radix cache for prefix-aware admission ordering.
    """

    def __init__(
        self,
        config: EngineRunConfig,
        pool: PagedLatentPool,
        cache: RadixStateCache | None = None,
    ) -> None:
        self.config = config
        self.pool = pool
        self.cache = cache

        # Ordered dicts preserve insertion order; we iterate front-to-back
        self._waiting: OrderedDict[str, EntityState] = OrderedDict()
        self._running: OrderedDict[str, EntityState] = OrderedDict()
        self._swapped: OrderedDict[str, EntityState] = OrderedDict()
        self._done: OrderedDict[str, EntityState] = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_request(self, request: EntityRequest) -> None:
        """Enqueue a new entity request."""
        state = EntityState(
            request=request,
            blocks_needed=max(1, self._estimate_blocks(request)),
        )

        # Check radix cache for prefix match
        if self.cache is not None and request.prefix_hash is not None:
            try:
                tokens = [int(x) for x in request.prefix_hash.split(",") if x]
                matched, _ = self.cache.match_prefix(tokens)
                state.prefix_matched = matched
            except (ValueError, AttributeError):
                pass

        self._waiting[request.request_id] = state

    def abort_request(self, request_id: str) -> bool:
        """Abort and remove a request from any queue. Returns True if found."""
        for queue in (self._waiting, self._running, self._swapped, self._done):
            if request_id in queue:
                state = queue.pop(request_id)
                if state.phase in (Phase.ENCODING, Phase.STEPPING):
                    self.pool.free(request_id)
                return True
        return False

    def schedule(self) -> SchedulerOutput:
        """Run one scheduling iteration.

        Returns a ``SchedulerOutput`` describing what the engine should do.
        """
        output = SchedulerOutput()

        # (1) Eject DONE entities
        newly_done: list[str] = []
        for rid, state in list(self._running.items()):
            if state.phase == Phase.DONE:
                newly_done.append(rid)
        for rid in newly_done:
            state = self._running.pop(rid)
            self._done[rid] = state
            output.done_ids.append(rid)

        # (2) Try swap-in preempted entities (highest priority first)
        swapped_by_priority = sorted(
            self._swapped.items(),
            key=lambda kv: kv[1].request.priority,
            reverse=True,
        )
        for rid, state in swapped_by_priority:
            needed = len(state.swapped_block_ids)
            if self.pool.num_free_blocks >= needed:
                pt = self.pool.swap_in(rid, state.swapped_block_ids)
                state.phase = Phase.STEPPING
                state.blocks_allocated = pt.num_blocks
                state.swapped_block_ids = ()
                self._swapped.pop(rid)
                self._running[rid] = state
                output.swap_in_ids.append(rid)

        # (3) Admit WAITING requests (cache-aware: prefer those with longer prefix matches)
        waiting_by_priority = sorted(
            self._waiting.items(),
            key=lambda kv: (kv[1].prefix_matched, kv[1].request.priority),
            reverse=True,
        )
        for rid, state in waiting_by_priority:
            if len(self._running) >= self.config.max_batch_size:
                break
            needed = state.blocks_needed
            if self.pool.num_free_blocks >= needed:
                pt = self.pool.alloc(rid, needed)
                state.phase = Phase.ENCODING
                state.blocks_allocated = pt.num_blocks
                self._waiting.pop(rid)
                self._running[rid] = state
                output.encode_ids.append(rid)

        # (4) Build STEPPING batch from running entities in STEPPING phase
        for rid, state in self._running.items():
            if state.phase == Phase.STEPPING:
                output.step_ids.append(rid)

        # (5) Preempt lowest-priority running entities to make room for waiting
        if self._waiting:
            min_needed = min(s.blocks_needed for s in self._waiting.values())
            if self.pool.num_free_blocks < min_needed:
                self._preempt_to_budget(output, min_needed)

        output.num_free_blocks = self.pool.num_free_blocks
        return output

    def on_encode_complete(self, request_ids: Sequence[str]) -> None:
        """Transition entities from ENCODING -> STEPPING after encode finishes."""
        for rid in request_ids:
            state = self._running.get(rid)
            if state is not None and state.phase == Phase.ENCODING:
                state.phase = Phase.STEPPING

    def on_step_complete(self, request_ids: Sequence[str]) -> None:
        """Advance step counter; mark DONE when target steps reached."""
        for rid in request_ids:
            state = self._running.get(rid)
            if state is None:
                continue
            state.step_index += 1
            if state.step_index >= state.request.num_steps:
                state.phase = Phase.DONE

    def drain_done(self) -> list[EntityState]:
        """Pop and return all completed entity states."""
        result = list(self._done.values())
        self._done.clear()
        return result

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def num_waiting(self) -> int:
        return len(self._waiting)

    def num_running(self) -> int:
        return len(self._running)

    def num_swapped(self) -> int:
        return len(self._swapped)

    def num_done(self) -> int:
        return len(self._done)

    def get_state(self, request_id: str) -> EntityState | None:
        for queue in (self._waiting, self._running, self._swapped, self._done):
            if request_id in queue:
                return queue[request_id]
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _estimate_blocks(self, request: EntityRequest) -> int:
        """Estimate the number of blocks an entity will need.

        Simplistic: 1 block per step, clamped to half the pool.
        """
        return min(request.num_steps, self.config.max_num_blocks // 2)

    def _preempt_to_budget(
        self, output: SchedulerOutput, blocks_needed: int
    ) -> None:
        """Preempt lowest-priority running entities until *blocks_needed* are free.

        Only preempts entities whose priority is strictly lower than the
        highest-priority waiting request.
        """
        if not self._waiting:
            return
        highest_waiting_priority = max(
            s.request.priority for s in self._waiting.values()
        )

        running_by_priority = sorted(
            self._running.items(),
            key=lambda kv: kv[1].request.priority,
        )
        for rid, state in running_by_priority:
            if self.pool.num_free_blocks >= blocks_needed:
                break
            if state.phase == Phase.DONE:
                continue
            # Only preempt if running entity has strictly lower priority
            if state.request.priority >= highest_waiting_priority:
                break
            if self.config.swap_enabled:
                try:
                    block_ids = self.pool.swap_out(rid)
                    state.phase = Phase.SWAPPED
                    state.swapped_block_ids = block_ids
                    state.blocks_allocated = 0
                    self._running.pop(rid)
                    self._swapped[rid] = state
                    output.preempt_ids.append(rid)
                except (KeyError, RuntimeError):
                    pass


# ---------------------------------------------------------------------------
# RolloutScheduler (from rollout.py)
# ---------------------------------------------------------------------------


class RolloutScheduler:
    """Schedules world model rollout steps across concurrent requests."""

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self._pending: Deque[RolloutRequest] = deque()
        self._active: dict[str, RolloutRequest] = {}
        self._step_counts: dict[str, int] = {}
        self._waiting_since: dict[str, float] = {}

    @property
    def num_pending(self) -> int:
        return len(self._pending)

    @property
    def num_active(self) -> int:
        return len(self._active)

    def submit(self, request: RolloutRequest) -> None:
        self._pending.append(request)
        self._waiting_since[request.request_id] = time.monotonic()

    def admit(self) -> list[str]:
        admitted = []
        while self._pending and self.num_active < self.config.max_concurrent_rollouts:
            req = self._pending.popleft()
            self._active[req.request_id] = req
            self._step_counts[req.request_id] = 0
            admitted.append(req.request_id)
        return admitted

    def schedule_batch(self) -> ScheduledBatch:
        self.admit()
        if not self._active:
            return ScheduledBatch(request_ids=[], step_indices=[], actions=[])

        candidates = list(self._active.values())
        if self.config.policy == SchedulerPolicy.SJF:
            candidates.sort(key=lambda r: r.num_steps - self._step_counts.get(r.request_id, 0))
        elif self.config.policy == SchedulerPolicy.DEADLINE:
            candidates.sort(key=lambda r: r.deadline or float("inf"))
        elif self.config.policy == SchedulerPolicy.MEMORY_AWARE:
            candidates.sort(key=lambda r: (r.estimate_resource_units(), -r.priority))

        now = time.monotonic()
        urgent = [r for r in candidates if (now - self._waiting_since.get(r.request_id, now)) * 1000 > self.config.max_waiting_time_ms]
        if urgent:
            candidates = urgent + [c for c in candidates if c not in urgent]

        selected = []
        consumed_units = 0.0
        for candidate in candidates:
            if len(selected) >= self.config.max_batch_size:
                break
            units = candidate.estimate_resource_units()
            if self.config.max_batch_resource_units is not None and selected and consumed_units + units > self.config.max_batch_resource_units:
                continue
            selected.append(candidate)
            consumed_units += units

        if not selected:
            selected = candidates[:1]

        return ScheduledBatch(
            request_ids=[r.request_id for r in selected],
            step_indices=[self._step_counts.get(r.request_id, 0) for r in selected],
            actions=[],
        )

    def step_completed(self, request_id: str) -> bool:
        self._step_counts[request_id] = self._step_counts.get(request_id, 0) + 1
        req = self._active.get(request_id)
        return bool(req and self._step_counts[request_id] >= req.num_steps)

    def complete(self, request_id: str) -> None:
        self._active.pop(request_id, None)
        self._step_counts.pop(request_id, None)
        self._waiting_since.pop(request_id, None)

    def cancel(self, request_id: str) -> None:
        self._active.pop(request_id, None)
        self._step_counts.pop(request_id, None)
        self._waiting_since.pop(request_id, None)
        self._pending = deque(r for r in self._pending if r.request_id != request_id)

    def has_work(self) -> bool:
        return bool(self._pending or self._active)
