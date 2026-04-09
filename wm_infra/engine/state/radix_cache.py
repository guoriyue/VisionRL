"""Radix tree cache for prefix-sharing of latent state sequences.

Used by the scheduler for cache-aware admission ordering: requests whose
action prefix already exists in the cache can skip re-encoding those steps.

Design modelled on SGLang's RadixCache but simplified for latent-state
(integer token IDs rather than KV tensors).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(slots=True)
class RadixNode:
    """One node in the radix tree."""

    key: tuple[int, ...] = ()
    children: dict[int, RadixNode] = field(default_factory=dict)
    value: str | None = None        # entity_id owning this prefix
    ref_count: int = 0              # number of live references
    block_ids: list[int] = field(default_factory=list)  # associated pool blocks

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class RadixStateCache:
    """Radix tree mapping integer-token sequences to cached block references.

    Each path from root to a node represents a prefix of action tokens.
    """

    def __init__(self) -> None:
        self.root = RadixNode()
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def match_prefix(self, tokens: Sequence[int]) -> tuple[int, RadixNode]:
        """Return (matched_length, node) for the longest cached prefix.

        Returns ``(0, root)`` when no prefix matches.
        """
        node = self.root
        matched = 0
        for tok in tokens:
            child = node.children.get(tok)
            if child is None:
                break
            node = child
            matched += 1
        return matched, node

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(
        self,
        tokens: Sequence[int],
        entity_id: str | None = None,
        block_ids: list[int] | None = None,
    ) -> RadixNode:
        """Insert a full token sequence into the tree.

        Intermediate nodes are created as needed. The leaf node gets
        ``entity_id`` and ``block_ids`` attached.
        """
        node = self.root
        for tok in tokens:
            if tok not in node.children:
                child = RadixNode(key=node.key + (tok,))
                node.children[tok] = child
                self._size += 1
            node = node.children[tok]
        node.value = entity_id
        node.block_ids = list(block_ids or [])
        node.ref_count += 1
        return node

    # ------------------------------------------------------------------
    # Remove
    # ------------------------------------------------------------------

    def remove(self, tokens: Sequence[int]) -> bool:
        """Decrement ref count for a token sequence.

        The node is NOT pruned from the tree — it stays available for
        prefix matching by future requests.  Use ``evict_lru()`` to
        reclaim unreferenced entries when under memory pressure.

        Returns True if the path existed.
        """
        node = self.root
        for tok in tokens:
            child = node.children.get(tok)
            if child is None:
                return False
            node = child

        if node.ref_count <= 0:
            return False

        node.ref_count -= 1
        if node.ref_count == 0:
            node.value = None
            node.block_ids.clear()

        return True

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def evict_lru(self, num_to_evict: int = 1) -> list[RadixNode]:
        """Evict up to *num_to_evict* unreferenced leaf nodes.

        Uses a bottom-up DFS that collects leaves with ``ref_count == 0``.
        After evicting a leaf, if its parent becomes an empty unreferenced
        leaf, the parent is evicted too (cascading cleanup).

        Returns the evicted nodes so the caller can free associated pool
        blocks.
        """
        evicted: list[RadixNode] = []
        self._evict_dfs(self.root, evicted, num_to_evict)
        return evicted

    def _evict_dfs(
        self,
        node: RadixNode,
        evicted: list[RadixNode],
        remaining: int,
    ) -> int:
        if remaining <= 0:
            return 0

        removed_count = 0
        # Copy keys to avoid mutation during iteration
        for tok in list(node.children.keys()):
            if remaining - removed_count <= 0:
                break
            child = node.children[tok]
            # Recurse first so child's subtree is cleaned bottom-up
            removed_count += self._evict_dfs(
                child, evicted, remaining - removed_count
            )

            # After recursion, if child is an unreferenced leaf → evict it
            if (
                child.is_leaf
                and child.ref_count == 0
                and remaining - removed_count > 0
            ):
                evicted.append(child)
                del node.children[tok]
                self._size -= 1
                removed_count += 1

        return removed_count

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def all_nodes(self) -> list[RadixNode]:
        """Collect all nodes in BFS order (for debugging)."""
        result: list[RadixNode] = []
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            result.append(node)
            queue.extend(node.children.values())
        return result

    def clear(self) -> None:
        """Remove all entries."""
        self.root = RadixNode()
        self._size = 0
