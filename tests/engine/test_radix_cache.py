"""Tests for RadixStateCache — prefix matching, insertion, removal, eviction."""

from __future__ import annotations

import pytest

from wm_infra.engine.state.radix_cache import RadixNode, RadixStateCache


@pytest.fixture
def cache() -> RadixStateCache:
    return RadixStateCache()


# ---------------------------------------------------------------------------
# Basic match / insert
# ---------------------------------------------------------------------------

class TestMatchInsert:
    def test_empty_cache_no_match(self, cache: RadixStateCache) -> None:
        matched, node = cache.match_prefix([1, 2, 3])
        assert matched == 0
        assert node is cache.root

    def test_insert_and_match_full(self, cache: RadixStateCache) -> None:
        cache.insert([1, 2, 3], entity_id="e1", block_ids=[10, 20, 30])
        matched, node = cache.match_prefix([1, 2, 3])
        assert matched == 3
        assert node.value == "e1"
        assert node.block_ids == [10, 20, 30]

    def test_partial_prefix_match(self, cache: RadixStateCache) -> None:
        cache.insert([1, 2, 3])
        matched, node = cache.match_prefix([1, 2, 4])
        assert matched == 2
        # Node at depth 2 has no value (intermediate)
        assert node.value is None

    def test_insert_creates_intermediate_nodes(self, cache: RadixStateCache) -> None:
        cache.insert([1, 2, 3])
        assert cache.size == 3  # three new nodes for tokens 1, 2, 3

    def test_insert_shared_prefix(self, cache: RadixStateCache) -> None:
        cache.insert([1, 2, 3])
        cache.insert([1, 2, 4])
        # tokens 1, 2 shared; 3 and 4 are separate leaves → 4 nodes total
        assert cache.size == 4

    def test_insert_increments_refcount(self, cache: RadixStateCache) -> None:
        node = cache.insert([1, 2], entity_id="e1")
        assert node.ref_count == 1
        node2 = cache.insert([1, 2], entity_id="e2")
        assert node2.ref_count == 2
        assert node is node2  # same physical node


# ---------------------------------------------------------------------------
# Branching
# ---------------------------------------------------------------------------

class TestBranching:
    def test_branching_paths(self, cache: RadixStateCache) -> None:
        cache.insert([1, 2, 3])
        cache.insert([1, 2, 4])
        cache.insert([1, 5, 6])

        m1, _ = cache.match_prefix([1, 2, 3])
        assert m1 == 3
        m2, _ = cache.match_prefix([1, 2, 4])
        assert m2 == 3
        m3, _ = cache.match_prefix([1, 5, 6])
        assert m3 == 3
        m4, _ = cache.match_prefix([1, 5, 7])
        assert m4 == 2

    def test_no_match_at_root(self, cache: RadixStateCache) -> None:
        cache.insert([1, 2, 3])
        matched, _ = cache.match_prefix([9, 9, 9])
        assert matched == 0


# ---------------------------------------------------------------------------
# Remove
# ---------------------------------------------------------------------------

class TestRemove:
    def test_remove_decrements_refcount(self, cache: RadixStateCache) -> None:
        cache.insert([1, 2, 3], entity_id="e1")
        assert cache.remove([1, 2, 3])
        # Node stays in tree (for prefix reuse) but ref=0, value=None
        matched, node = cache.match_prefix([1, 2, 3])
        assert matched == 3
        assert node.ref_count == 0
        assert node.value is None
        # Tree structure preserved — eviction is needed to prune
        assert cache.size == 3

    def test_remove_nonexistent(self, cache: RadixStateCache) -> None:
        assert not cache.remove([1, 2, 3])

    def test_remove_already_zero(self, cache: RadixStateCache) -> None:
        cache.insert([1, 2])
        cache.remove([1, 2])
        # Second remove on ref=0 returns False
        assert not cache.remove([1, 2])

    def test_remove_preserves_sibling(self, cache: RadixStateCache) -> None:
        cache.insert([1, 2, 3])
        cache.insert([1, 2, 4])
        cache.remove([1, 2, 3])
        matched, node = cache.match_prefix([1, 2, 4])
        assert matched == 3

    def test_remove_with_refcount(self, cache: RadixStateCache) -> None:
        """With refcount > 1, first remove just decrements refcount."""
        cache.insert([1, 2], entity_id="e1")
        cache.insert([1, 2], entity_id="e2")
        cache.remove([1, 2])
        matched, node = cache.match_prefix([1, 2])
        assert matched == 2
        assert node.ref_count == 1
        assert node.value == "e2"  # last insert's value

        cache.remove([1, 2])
        matched, node = cache.match_prefix([1, 2])
        assert matched == 2
        assert node.ref_count == 0
        assert node.value is None


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------

class TestEviction:
    def test_evict_unreferenced_leaf(self, cache: RadixStateCache) -> None:
        cache.insert([1, 2, 3])
        cache.remove([1, 2, 3])  # refcount → 0, nodes stay in tree
        assert cache.size == 3

        evicted = cache.evict_lru(num_to_evict=10)
        # Should evict the leaf (1,2,3), then (1,2), then (1) — all unreferenced
        assert len(evicted) == 3
        assert cache.size == 0

    def test_evict_with_shared_prefix(self, cache: RadixStateCache) -> None:
        cache.insert([1, 2, 3])
        cache.insert([1, 2, 4])
        cache.remove([1, 2, 3])
        # [1,2,4] still referenced — only [1,2,3] leaf should be evictable
        evicted = cache.evict_lru(num_to_evict=10)
        assert len(evicted) == 1  # only the (1,2,3) leaf
        evicted_keys = [n.key for n in evicted]
        assert (1, 2, 3) in evicted_keys

    def test_evict_skips_referenced(self, cache: RadixStateCache) -> None:
        cache.insert([1, 2, 3], entity_id="live")  # refcount=1
        evicted = cache.evict_lru(num_to_evict=10)
        assert len(evicted) == 0

    def test_evict_respects_limit(self, cache: RadixStateCache) -> None:
        for i in range(5):
            cache.insert([100 + i])
            cache.remove([100 + i])

        evicted = cache.evict_lru(num_to_evict=2)
        assert len(evicted) == 2

    def test_evict_cascading_cleanup(self, cache: RadixStateCache) -> None:
        """After evicting a leaf, if parent becomes empty leaf, evict it too."""
        cache.insert([1, 2, 3])
        cache.remove([1, 2, 3])
        # All 3 nodes are unreferenced leaves (bottom-up)
        evicted = cache.evict_lru(num_to_evict=10)
        assert cache.size == 0
        assert len(evicted) == 3


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_clear(self, cache: RadixStateCache) -> None:
        cache.insert([1, 2, 3])
        cache.insert([4, 5])
        cache.clear()
        assert cache.size == 0
        assert cache.root.children == {}

    def test_all_nodes(self, cache: RadixStateCache) -> None:
        cache.insert([1, 2])
        nodes = cache.all_nodes()
        # root + node(1) + node(1,2)
        assert len(nodes) == 3
