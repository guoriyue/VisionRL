"""Tests for ContinuousBatchingScheduler — admission, stepping, preemption, swap."""

from __future__ import annotations

import pytest

from wm_infra.engine._types import EngineRunConfig, EntityRequest, Phase
from wm_infra.engine.scheduler import ContinuousBatchingScheduler, EntityState
from wm_infra.engine.state.paged_pool import PagedLatentPool
from wm_infra.engine.state.radix_cache import RadixStateCache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(**overrides: object) -> EngineRunConfig:
    defaults = dict(
        max_num_blocks=32,
        block_size=1,
        latent_tokens=4,
        latent_dim=8,
        max_batch_size=8,
        max_steps_per_entity=16,
    )
    defaults.update(overrides)
    return EngineRunConfig(**defaults)


def _make_pool(config: EngineRunConfig) -> PagedLatentPool:
    return PagedLatentPool(
        num_blocks=config.max_num_blocks,
        block_size=config.block_size,
        latent_tokens=config.latent_tokens,
        latent_dim=config.latent_dim,
        device="cpu",
    )


def _make_request(rid: str, num_steps: int = 3, priority: float = 0.0) -> EntityRequest:
    return EntityRequest(request_id=rid, num_steps=num_steps, priority=priority)


@pytest.fixture
def config() -> EngineRunConfig:
    return _make_config()


@pytest.fixture
def scheduler(config: EngineRunConfig) -> ContinuousBatchingScheduler:
    pool = _make_pool(config)
    return ContinuousBatchingScheduler(config=config, pool=pool)


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------

class TestAdmission:
    def test_add_and_schedule_admits_request(self, scheduler: ContinuousBatchingScheduler) -> None:
        scheduler.add_request(_make_request("r1"))
        out = scheduler.schedule()
        assert "r1" in out.encode_ids
        assert scheduler.num_running() == 1
        assert scheduler.num_waiting() == 0

    def test_multiple_admits(self, scheduler: ContinuousBatchingScheduler) -> None:
        for i in range(4):
            scheduler.add_request(_make_request(f"r{i}"))
        out = scheduler.schedule()
        assert len(out.encode_ids) == 4

    def test_batch_size_limit(self) -> None:
        config = _make_config(max_batch_size=2)
        pool = _make_pool(config)
        sched = ContinuousBatchingScheduler(config=config, pool=pool)
        for i in range(5):
            sched.add_request(_make_request(f"r{i}"))
        out = sched.schedule()
        assert len(out.encode_ids) == 2
        assert sched.num_waiting() == 3

    def test_block_budget_limit(self) -> None:
        """When pool has limited blocks, admit fewer requests."""
        config = _make_config(max_num_blocks=4, max_batch_size=10)
        pool = _make_pool(config)
        sched = ContinuousBatchingScheduler(config=config, pool=pool)
        # Each request needs ~2 blocks (min estimate)
        for i in range(10):
            sched.add_request(_make_request(f"r{i}", num_steps=2))
        out = sched.schedule()
        # Can't admit all 10 with only 4 blocks
        assert len(out.encode_ids) < 10
        assert sched.num_waiting() > 0

    def test_continuous_admission_across_iterations(self, scheduler: ContinuousBatchingScheduler) -> None:
        """New requests are admitted on subsequent schedule() calls."""
        scheduler.add_request(_make_request("r1", num_steps=1))
        out1 = scheduler.schedule()
        assert "r1" in out1.encode_ids

        # Complete r1
        scheduler.on_encode_complete(["r1"])
        scheduler.on_step_complete(["r1"])  # 1 step → done
        out2 = scheduler.schedule()
        assert "r1" in out2.done_ids

        # Now add and admit r2
        scheduler.add_request(_make_request("r2"))
        out3 = scheduler.schedule()
        assert "r2" in out3.encode_ids


# ---------------------------------------------------------------------------
# Phase transitions
# ---------------------------------------------------------------------------

class TestPhaseTransitions:
    def test_encode_to_stepping(self, scheduler: ContinuousBatchingScheduler) -> None:
        scheduler.add_request(_make_request("r1"))
        scheduler.schedule()
        state = scheduler.get_state("r1")
        assert state.phase == Phase.ENCODING

        scheduler.on_encode_complete(["r1"])
        assert state.phase == Phase.STEPPING

    def test_stepping_increments_step_index(self, scheduler: ContinuousBatchingScheduler) -> None:
        scheduler.add_request(_make_request("r1", num_steps=5))
        scheduler.schedule()
        scheduler.on_encode_complete(["r1"])
        assert scheduler.get_state("r1").step_index == 0

        scheduler.on_step_complete(["r1"])
        assert scheduler.get_state("r1").step_index == 1

        scheduler.on_step_complete(["r1"])
        assert scheduler.get_state("r1").step_index == 2

    def test_done_when_steps_reached(self, scheduler: ContinuousBatchingScheduler) -> None:
        scheduler.add_request(_make_request("r1", num_steps=2))
        scheduler.schedule()
        scheduler.on_encode_complete(["r1"])

        scheduler.on_step_complete(["r1"])
        assert scheduler.get_state("r1").phase == Phase.STEPPING

        scheduler.on_step_complete(["r1"])
        assert scheduler.get_state("r1").phase == Phase.DONE

    def test_drain_done(self, scheduler: ContinuousBatchingScheduler) -> None:
        scheduler.add_request(_make_request("r1", num_steps=1))
        scheduler.schedule()
        scheduler.on_encode_complete(["r1"])
        scheduler.on_step_complete(["r1"])

        out = scheduler.schedule()  # ejects done
        assert "r1" in out.done_ids

        done = scheduler.drain_done()
        assert len(done) == 1
        assert done[0].request.request_id == "r1"

    def test_step_ids_in_schedule(self, scheduler: ContinuousBatchingScheduler) -> None:
        scheduler.add_request(_make_request("r1", num_steps=5))
        scheduler.schedule()
        scheduler.on_encode_complete(["r1"])

        out = scheduler.schedule()
        assert "r1" in out.step_ids


# ---------------------------------------------------------------------------
# Abort
# ---------------------------------------------------------------------------

class TestAbort:
    def test_abort_waiting(self, scheduler: ContinuousBatchingScheduler) -> None:
        scheduler.add_request(_make_request("r1"))
        assert scheduler.abort_request("r1")
        assert scheduler.num_waiting() == 0

    def test_abort_running(self, scheduler: ContinuousBatchingScheduler) -> None:
        scheduler.add_request(_make_request("r1"))
        scheduler.schedule()
        assert scheduler.abort_request("r1")
        assert scheduler.num_running() == 0

    def test_abort_nonexistent(self, scheduler: ContinuousBatchingScheduler) -> None:
        assert not scheduler.abort_request("no_such")


# ---------------------------------------------------------------------------
# Preemption / swap
# ---------------------------------------------------------------------------

class TestPreemptionSwap:
    def test_swap_out_and_in(self) -> None:
        config = _make_config(max_num_blocks=6, max_batch_size=10)
        pool = _make_pool(config)
        sched = ContinuousBatchingScheduler(config=config, pool=pool)

        # Admit one request that uses some blocks
        sched.add_request(_make_request("r1", num_steps=3, priority=1.0))
        sched.schedule()
        sched.on_encode_complete(["r1"])

        state = sched.get_state("r1")
        assert state.phase == Phase.STEPPING

        # Manually test swap-out via pool (the scheduler does this internally during preemption)
        block_ids = pool.swap_out("r1")
        assert pool.num_free_blocks == 6  # all blocks freed

        # Swap back in
        pool.swap_in("r1", block_ids)
        assert pool.get_page_table("r1") is not None

    def test_cache_aware_ordering(self) -> None:
        config = _make_config(max_num_blocks=32, max_batch_size=2)
        pool = _make_pool(config)
        cache = RadixStateCache()
        # Insert a prefix for "r1"
        cache.insert([1, 2, 3], entity_id="cache_entry")

        sched = ContinuousBatchingScheduler(config=config, pool=pool, cache=cache)
        sched.add_request(EntityRequest(
            request_id="r1", num_steps=3, prefix_hash="1,2,3",
        ))
        sched.add_request(EntityRequest(
            request_id="r2", num_steps=3, prefix_hash="9,9",
        ))

        out = sched.schedule()
        # r1 has longer prefix match → should be admitted first
        assert out.encode_ids[0] == "r1"


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

class TestIntrospection:
    def test_get_state(self, scheduler: ContinuousBatchingScheduler) -> None:
        scheduler.add_request(_make_request("r1"))
        state = scheduler.get_state("r1")
        assert state is not None
        assert state.phase == Phase.WAITING

    def test_get_state_nonexistent(self, scheduler: ContinuousBatchingScheduler) -> None:
        assert scheduler.get_state("no_such") is None

    def test_counts(self, scheduler: ContinuousBatchingScheduler) -> None:
        assert scheduler.num_waiting() == 0
        assert scheduler.num_running() == 0
        scheduler.add_request(_make_request("r1"))
        assert scheduler.num_waiting() == 1
        scheduler.schedule()
        assert scheduler.num_running() == 1
        assert scheduler.num_waiting() == 0
