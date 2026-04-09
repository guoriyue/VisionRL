"""Tests for PagedLatentPool — block allocation, gather/scatter, COW fork, swap."""

from __future__ import annotations

import pytest
import torch

from wm_infra.engine.state.paged_pool import PagedLatentPool, PageTable


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pool() -> PagedLatentPool:
    """Small pool: 16 blocks, block_size=1, 4 tokens, dim=8."""
    return PagedLatentPool(
        num_blocks=16, block_size=1, latent_tokens=4, latent_dim=8, device="cpu",
    )


@pytest.fixture
def large_pool() -> PagedLatentPool:
    """Larger pool: 64 blocks, block_size=2, 8 tokens, dim=4."""
    return PagedLatentPool(
        num_blocks=64, block_size=2, latent_tokens=8, latent_dim=4, device="cpu",
    )


# ---------------------------------------------------------------------------
# Allocation tests
# ---------------------------------------------------------------------------

class TestAllocation:
    def test_alloc_returns_page_table(self, pool: PagedLatentPool) -> None:
        pt = pool.alloc("e1", 3)
        assert isinstance(pt, PageTable)
        assert pt.entity_id == "e1"
        assert pt.num_blocks == 3

    def test_alloc_decreases_free_count(self, pool: PagedLatentPool) -> None:
        assert pool.num_free_blocks == 16
        pool.alloc("e1", 4)
        assert pool.num_free_blocks == 12

    def test_alloc_too_many_blocks_raises(self, pool: PagedLatentPool) -> None:
        with pytest.raises(RuntimeError, match="Cannot allocate"):
            pool.alloc("e1", 100)

    def test_alloc_duplicate_entity_raises(self, pool: PagedLatentPool) -> None:
        pool.alloc("e1", 1)
        with pytest.raises(ValueError, match="already has a page table"):
            pool.alloc("e1", 1)

    def test_alloc_blocks_extends_existing(self, pool: PagedLatentPool) -> None:
        pt = pool.alloc("e1", 2)
        assert pt.num_blocks == 2
        new_ids = pool.alloc_blocks("e1", 3)
        assert len(new_ids) == 3
        assert pt.num_blocks == 5

    def test_alloc_blocks_nonexistent_raises(self, pool: PagedLatentPool) -> None:
        with pytest.raises(KeyError):
            pool.alloc_blocks("no_such_entity", 1)

    def test_free_returns_blocks(self, pool: PagedLatentPool) -> None:
        pool.alloc("e1", 4)
        pool.alloc("e2", 4)
        assert pool.num_free_blocks == 8
        pool.free("e1")
        assert pool.num_free_blocks == 12
        pool.free("e2")
        assert pool.num_free_blocks == 16

    def test_free_nonexistent_is_noop(self, pool: PagedLatentPool) -> None:
        pool.free("nonexistent")  # should not raise

    def test_entity_ids(self, pool: PagedLatentPool) -> None:
        pool.alloc("a", 1)
        pool.alloc("b", 1)
        assert set(pool.entity_ids()) == {"a", "b"}
        pool.free("a")
        assert pool.entity_ids() == ["b"]


# ---------------------------------------------------------------------------
# Gather / Scatter tests
# ---------------------------------------------------------------------------

class TestGatherScatter:
    def test_gather_empty(self, pool: PagedLatentPool) -> None:
        batch = pool.gather_batch([])
        assert batch.shape[0] == 0

    def test_gather_single_entity(self, pool: PagedLatentPool) -> None:
        pool.alloc("e1", 3)
        batch = pool.gather_batch(["e1"])
        assert batch.shape == (1, 3, 4, 8)  # B=1, steps=3*1, tokens=4, dim=8

    def test_scatter_then_gather_roundtrip(self, pool: PagedLatentPool) -> None:
        pool.alloc("e1", 2)
        data = torch.randn(1, 2, 4, 8)
        pool.scatter_results(["e1"], data)
        gathered = pool.gather_batch(["e1"])
        assert torch.allclose(gathered, data)

    def test_multi_entity_gather(self, pool: PagedLatentPool) -> None:
        pool.alloc("e1", 2)
        pool.alloc("e2", 3)
        batch = pool.gather_batch(["e1", "e2"])
        # Max blocks = 3, so total_steps = 3*1 = 3
        assert batch.shape == (2, 3, 4, 8)

    def test_cross_block_gather_scatter(self, large_pool: PagedLatentPool) -> None:
        """Multi-step blocks: block_size=2 means 2 steps per block."""
        large_pool.alloc("e1", 3)
        # 3 blocks * 2 steps/block = 6 total steps
        data = torch.randn(1, 6, 8, 4)
        large_pool.scatter_results(["e1"], data)
        gathered = large_pool.gather_batch(["e1"])
        assert torch.allclose(gathered, data)


# ---------------------------------------------------------------------------
# COW fork tests
# ---------------------------------------------------------------------------

class TestFork:
    def test_fork_shares_blocks(self, pool: PagedLatentPool) -> None:
        pool.alloc("src", 3)
        data = torch.randn(1, 3, 4, 8)
        pool.scatter_results(["src"], data)

        dst_pt = pool.fork("src", "dst")
        assert dst_pt.num_blocks == 3
        # Shared blocks → same data
        gathered = pool.gather_batch(["dst"])
        assert torch.allclose(gathered, data)

    def test_fork_cow_on_write(self, pool: PagedLatentPool) -> None:
        """After fork, writing to dst should not affect src (COW)."""
        pool.alloc("src", 2)
        orig = torch.ones(1, 2, 4, 8)
        pool.scatter_results(["src"], orig)

        pool.fork("src", "dst")
        new_data = torch.ones(1, 2, 4, 8) * 42.0
        pool.scatter_results(["dst"], new_data)

        src_gathered = pool.gather_batch(["src"])
        dst_gathered = pool.gather_batch(["dst"])
        assert torch.allclose(src_gathered, orig)
        assert torch.allclose(dst_gathered, new_data)

    def test_fork_nonexistent_raises(self, pool: PagedLatentPool) -> None:
        with pytest.raises(KeyError):
            pool.fork("no_such", "dst")

    def test_fork_duplicate_dst_raises(self, pool: PagedLatentPool) -> None:
        pool.alloc("src", 1)
        pool.alloc("dst", 1)
        with pytest.raises(ValueError, match="already has a page table"):
            pool.fork("src", "dst")

    def test_free_shared_blocks_refcount(self, pool: PagedLatentPool) -> None:
        """Freeing one side of a fork should not free shared blocks."""
        pool.alloc("src", 2)
        pool.fork("src", "dst")
        free_before = pool.num_free_blocks
        pool.free("src")
        # Blocks still held by dst, so free count shouldn't increase by 2
        assert pool.num_free_blocks == free_before
        pool.free("dst")
        assert pool.num_free_blocks == free_before + 2


# ---------------------------------------------------------------------------
# Swap tests
# ---------------------------------------------------------------------------

class TestSwap:
    def test_swap_out_frees_blocks(self, pool: PagedLatentPool) -> None:
        pool.alloc("e1", 4)
        assert pool.num_free_blocks == 12
        block_ids = pool.swap_out("e1")
        assert len(block_ids) == 4
        assert pool.num_free_blocks == 16

    def test_swap_roundtrip_preserves_data(self, pool: PagedLatentPool) -> None:
        pool.alloc("e1", 2)
        data = torch.randn(1, 2, 4, 8)
        pool.scatter_results(["e1"], data)

        block_ids = pool.swap_out("e1")
        pt = pool.swap_in("e1", block_ids)
        assert pt.num_blocks == 2

        gathered = pool.gather_batch(["e1"])
        assert torch.allclose(gathered, data)

    def test_swap_out_nonexistent_raises(self, pool: PagedLatentPool) -> None:
        with pytest.raises(KeyError):
            pool.swap_out("no_such")


# ---------------------------------------------------------------------------
# read_block / get_page_table
# ---------------------------------------------------------------------------

class TestIntrospection:
    def test_read_block(self, pool: PagedLatentPool) -> None:
        pt = pool.alloc("e1", 1)
        block = pool.read_block(pt.block_ids[0])
        assert block.shape == (1, 4, 8)

    def test_get_page_table(self, pool: PagedLatentPool) -> None:
        pool.alloc("e1", 2)
        pt = pool.get_page_table("e1")
        assert pt is not None
        assert pt.entity_id == "e1"
        assert pool.get_page_table("no_such") is None

    def test_logical_to_physical(self, pool: PagedLatentPool) -> None:
        pt = pool.alloc("e1", 3)
        for i in range(3):
            assert pt.logical_to_physical(i) == pt.block_ids[i]
