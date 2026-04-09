"""Paged latent pool — vLLM-style block allocator for latent state tensors.

The pool pre-allocates a single large tensor ``[num_blocks, block_size,
latent_tokens, latent_dim]`` at startup and manages it with a free-list.
Entity page-tables map logical block indices to physical block IDs.
Copy-on-write fork uses reference counting to share blocks until mutation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import torch


@dataclass(slots=True)
class PageTable:
    """Per-entity mapping from logical block index to physical block ID."""

    entity_id: str
    block_ids: list[int] = field(default_factory=list)

    @property
    def num_blocks(self) -> int:
        return len(self.block_ids)

    def logical_to_physical(self, logical: int) -> int:
        return self.block_ids[logical]


class PagedLatentPool:
    """Block-managed GPU/CPU tensor pool for latent state.

    Parameters
    ----------
    num_blocks : int
        Total number of physical blocks.
    block_size : int
        Number of steps (frames) per block.
    latent_tokens : int
        Token count per latent state vector.
    latent_dim : int
        Dimension of each latent token.
    device : str
        Torch device for the pool tensor (``"cpu"`` or ``"cuda:N"``).
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        latent_tokens: int,
        latent_dim: int,
        device: str = "cpu",
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.latent_tokens = latent_tokens
        self.latent_dim = latent_dim
        self.device = device

        self.pool = torch.zeros(
            num_blocks, block_size, latent_tokens, latent_dim,
            device=device, dtype=torch.float32,
        )

        self._free: deque[int] = deque(range(num_blocks))
        self._ref_count: dict[int, int] = {}
        self._page_tables: dict[str, PageTable] = {}

        # Per-entity swap storage: entity_id → {original_block_id → cpu tensor}
        # Using per-entity dicts avoids corruption when freed block IDs are
        # reused by another entity before swap-in.
        self._swap_store: dict[str, dict[int, torch.Tensor]] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - self.num_free_blocks

    def get_page_table(self, entity_id: str) -> PageTable | None:
        return self._page_tables.get(entity_id)

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def alloc(self, entity_id: str, num_blocks: int) -> PageTable:
        """Allocate *num_blocks* physical blocks for *entity_id*."""
        if num_blocks > len(self._free):
            raise RuntimeError(
                f"Cannot allocate {num_blocks} blocks — only {len(self._free)} free"
            )
        if entity_id in self._page_tables:
            raise ValueError(f"Entity {entity_id!r} already has a page table")

        block_ids: list[int] = []
        for _ in range(num_blocks):
            bid = self._free.popleft()
            block_ids.append(bid)
            self._ref_count[bid] = 1

        pt = PageTable(entity_id=entity_id, block_ids=block_ids)
        self._page_tables[entity_id] = pt
        return pt

    def alloc_blocks(self, entity_id: str, num_blocks: int) -> list[int]:
        """Append *num_blocks* new physical blocks to an existing entity."""
        if num_blocks > len(self._free):
            raise RuntimeError(
                f"Cannot allocate {num_blocks} blocks — only {len(self._free)} free"
            )
        pt = self._page_tables.get(entity_id)
        if pt is None:
            raise KeyError(f"No page table for entity {entity_id!r}")

        new_ids: list[int] = []
        for _ in range(num_blocks):
            bid = self._free.popleft()
            new_ids.append(bid)
            self._ref_count[bid] = 1
            pt.block_ids.append(bid)
        return new_ids

    def free(self, entity_id: str) -> None:
        """Release all blocks held by *entity_id*."""
        pt = self._page_tables.pop(entity_id, None)
        if pt is None:
            return
        for bid in pt.block_ids:
            rc = self._ref_count.get(bid, 0) - 1
            if rc <= 0:
                self._ref_count.pop(bid, None)
                self.pool[bid].zero_()
                self._free.append(bid)
            else:
                self._ref_count[bid] = rc

    # ------------------------------------------------------------------
    # Gather / Scatter (batched tensor ops)
    # ------------------------------------------------------------------

    def gather_batch(self, entity_ids: Sequence[str]) -> torch.Tensor:
        """Gather the concatenated latent blocks for a batch of entities.

        Returns shape ``[B, total_steps, latent_tokens, latent_dim]`` where
        *total_steps* = max(entity_blocks) * block_size (zero-padded).
        """
        if not entity_ids:
            return torch.empty(0, 0, self.latent_tokens, self.latent_dim,
                               device=self.device)

        max_blocks = max(
            self._page_tables[eid].num_blocks for eid in entity_ids
        )
        total_steps = max_blocks * self.block_size

        batch = torch.zeros(
            len(entity_ids), total_steps, self.latent_tokens, self.latent_dim,
            device=self.device, dtype=self.pool.dtype,
        )
        for i, eid in enumerate(entity_ids):
            pt = self._page_tables[eid]
            for j, bid in enumerate(pt.block_ids):
                start = j * self.block_size
                end = start + self.block_size
                batch[i, start:end] = self.pool[bid]
        return batch

    def scatter_results(
        self,
        entity_ids: Sequence[str],
        data: torch.Tensor,
    ) -> None:
        """Write batched results back into the pool blocks.

        *data* shape: ``[B, total_steps, latent_tokens, latent_dim]``.
        """
        for i, eid in enumerate(entity_ids):
            pt = self._page_tables[eid]
            for j, bid in enumerate(pt.block_ids):
                start = j * self.block_size
                end = start + self.block_size
                self._ensure_writable(eid, j)
                actual_bid = self._page_tables[eid].block_ids[j]
                self.pool[actual_bid] = data[i, start:end]

    # ------------------------------------------------------------------
    # Copy-on-write fork
    # ------------------------------------------------------------------

    def fork(self, src_entity: str, dst_entity: str) -> PageTable:
        """COW-fork: share physical blocks with reference counting."""
        src_pt = self._page_tables.get(src_entity)
        if src_pt is None:
            raise KeyError(f"No page table for entity {src_entity!r}")
        if dst_entity in self._page_tables:
            raise ValueError(f"Entity {dst_entity!r} already has a page table")

        new_block_ids = list(src_pt.block_ids)
        for bid in new_block_ids:
            self._ref_count[bid] = self._ref_count.get(bid, 1) + 1

        dst_pt = PageTable(entity_id=dst_entity, block_ids=new_block_ids)
        self._page_tables[dst_entity] = dst_pt
        return dst_pt

    def _ensure_writable(self, entity_id: str, logical_idx: int) -> None:
        """If block is shared (refcount > 1), copy to a new block first."""
        pt = self._page_tables[entity_id]
        bid = pt.block_ids[logical_idx]
        if self._ref_count.get(bid, 1) <= 1:
            return
        # COW: allocate new block, copy data, decrement old ref
        if not self._free:
            raise RuntimeError("No free blocks for COW copy")
        new_bid = self._free.popleft()
        self.pool[new_bid] = self.pool[bid].clone()
        self._ref_count[new_bid] = 1
        self._ref_count[bid] -= 1
        pt.block_ids[logical_idx] = new_bid

    # ------------------------------------------------------------------
    # Swap (GPU ↔ CPU)
    # ------------------------------------------------------------------

    def swap_out(self, entity_id: str) -> tuple[int, ...]:
        """Copy entity blocks to CPU swap space and free GPU blocks.

        Each entity's data is stored in an isolated dict keyed by original
        block ID, so freed block IDs can be safely reused before swap-in.
        """
        pt = self._page_tables.get(entity_id)
        if pt is None:
            raise KeyError(f"No page table for entity {entity_id!r}")

        block_ids = tuple(pt.block_ids)
        store: dict[int, torch.Tensor] = {}
        for bid in block_ids:
            store[bid] = self.pool[bid].clone().to("cpu")
        self._swap_store[entity_id] = store

        self.free(entity_id)
        return block_ids

    def swap_in(self, entity_id: str, block_ids: tuple[int, ...]) -> PageTable:
        """Restore entity blocks from CPU swap space to GPU pool."""
        store = self._swap_store.pop(entity_id, None)
        if store is None:
            raise RuntimeError(f"No swap data for entity {entity_id!r}")

        # Allocate fresh physical blocks (may differ from original IDs)
        pt = self.alloc(entity_id, len(block_ids))
        for i, old_bid in enumerate(block_ids):
            new_bid = pt.block_ids[i]
            self.pool[new_bid] = store[old_bid].to(self.pool.device)
        return pt

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def read_block(self, block_id: int) -> torch.Tensor:
        """Read a single physical block (for debugging / tests)."""
        return self.pool[block_id]

    def entity_ids(self) -> list[str]:
        """Return all entity IDs that have page tables."""
        return list(self._page_tables.keys())
