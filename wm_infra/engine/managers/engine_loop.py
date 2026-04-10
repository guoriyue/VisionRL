"""Engine loop — persistent async loop that drives drain -> schedule -> execute -> output.

The ``EngineLoop`` is the top-level orchestrator. It owns a scheduler and
a worker, pulls requests from a queue, runs the scheduling/execution cycle
in a loop, and resolves per-request futures when entities complete.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from wm_infra.engine.managers.scheduler import ContinuousBatchingScheduler
from wm_infra.engine.mem_cache.paged_pool import PagedLatentPool
from wm_infra.engine.mem_cache.radix_cache import RadixStateCache
from wm_infra.engine.model_executor.worker import RequestQueue, Worker
from wm_infra.engine.types import (
    EngineRunConfig,
    EntityRequest,
    StepResult,
)

logger = logging.getLogger(__name__)


class EngineLoop:
    """Persistent async loop for continuous-batching engine execution.

    Usage::

        loop = EngineLoop(config)
        loop.register_stage(encode_stage)
        loop.register_stage(dynamics_stage)
        await loop.start()
        result = await loop.submit(request)
        await loop.stop()

    Parameters
    ----------
    config : EngineRunConfig
        Engine configuration.
    cache : RadixStateCache | None
        Optional prefix cache for cache-aware scheduling.
    """

    def __init__(
        self,
        config: EngineRunConfig,
        cache: RadixStateCache | None = None,
    ) -> None:
        self.config = config
        self.pool = PagedLatentPool(
            num_blocks=config.max_num_blocks,
            block_size=config.block_size,
            latent_tokens=config.latent_tokens,
            latent_dim=config.latent_dim,
            device=config.device,
        )
        self.cache = cache
        self.scheduler = ContinuousBatchingScheduler(
            config=config,
            pool=self.pool,
            cache=cache,
        )
        self.worker = Worker(pool=self.pool, device=config.device)

        self._request_queue = RequestQueue()
        self._futures: dict[str, asyncio.Future[list[StepResult]]] = {}
        self._results: dict[str, list[StepResult]] = {}
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._has_work = asyncio.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the persistent engine loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop the engine loop gracefully."""
        self._running = False
        self._has_work.set()  # wake the loop so it can exit
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        # Cancel any unresolved futures so callers don't hang
        for _rid, future in self._futures.items():
            if not future.done():
                future.cancel()
        self._futures.clear()
        self._results.clear()

    @property
    def is_running(self) -> bool:
        return self._running

    def register_stage(self, stage: Any) -> None:
        """Register a stage with the worker."""
        self.worker.register_stage(stage)

    # ------------------------------------------------------------------
    # Request submission
    # ------------------------------------------------------------------

    async def submit(self, request: EntityRequest) -> list[StepResult]:
        """Submit a request and await its completion.

        Returns the list of ``StepResult``s collected across all steps.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[StepResult]] = loop.create_future()
        self._futures[request.request_id] = future
        self._results[request.request_id] = []
        self._request_queue.put_nowait(request)
        self._has_work.set()
        return await future

    def submit_nowait(self, request: EntityRequest) -> asyncio.Future[list[StepResult]]:
        """Submit a request without awaiting. Returns the future."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[StepResult]] = loop.create_future()
        self._futures[request.request_id] = future
        self._results[request.request_id] = []
        self._request_queue.put_nowait(request)
        self._has_work.set()
        return future

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        """Persistent loop: drain -> schedule -> execute -> output."""
        while self._running:
            try:
                has_active = await self._iteration()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Engine loop iteration failed")
                has_active = False

            if has_active:
                # Active entities — yield but iterate again immediately
                await asyncio.sleep(0)
            else:
                # Idle — wait for new work instead of busy-spinning
                self._has_work.clear()
                await self._has_work.wait()

    async def _iteration(self) -> bool:
        """One iteration of the engine loop. Returns True if there's active work."""

        # (1) Drain: pull new requests from the queue
        new_requests = self._request_queue.drain()
        for req in new_requests:
            self.scheduler.add_request(req)

        # (2) Schedule
        sched_output = self.scheduler.schedule()

        # (3) Execute encode stage
        if sched_output.encode_ids:
            encode_stage = self.worker.get_stage("encode")
            if encode_stage is not None:
                self.worker.execute_step(sched_output.encode_ids, "encode")
            self.scheduler.on_encode_complete(sched_output.encode_ids)

        # (4) Execute dynamics step
        if sched_output.step_ids:
            dynamics_stage = self.worker.get_stage("dynamics")
            if dynamics_stage is not None:
                step_indices = []
                for rid in sched_output.step_ids:
                    state = self.scheduler.get_state(rid)
                    step_indices.append(state.step_index if state else 0)

                results = self.worker.execute_step(
                    sched_output.step_ids,
                    "dynamics",
                    step_indices=step_indices,
                )
                for result in results:
                    if result.request_id in self._results:
                        self._results[result.request_id].append(result)

            self.scheduler.on_step_complete(sched_output.step_ids)

        # (5) Output: resolve futures for done entities
        done_states = self.scheduler.drain_done()
        for entity_state in done_states:
            rid = entity_state.request.request_id
            future = self._futures.pop(rid, None)
            results = self._results.pop(rid, [])
            self.pool.free(rid)
            if future is not None and not future.done():
                future.set_result(results)

        # Return whether there's still active work
        return bool(
            self.scheduler.num_running()
            or self.scheduler.num_waiting()
            or not self._request_queue.empty()
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def num_pending(self) -> int:
        return self._request_queue.qsize()

    def num_active(self) -> int:
        return self.scheduler.num_running()

    def num_waiting(self) -> int:
        return self.scheduler.num_waiting()
