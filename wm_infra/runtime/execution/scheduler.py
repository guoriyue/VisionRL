"""Shared schedulers for homogeneous temporal execution."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

from wm_infra.runtime.execution.types import (
    BatchSignature,
    ExecutionBatchPolicy,
    ExecutionChunk,
    ExecutionWorkItem,
)


@dataclass(frozen=True, slots=True)
class SchedulerDecision:
    """Batching decision for one homogeneous signature lane."""

    signature: BatchSignature
    ready_count: int
    chunk_count: int
    policy_mode: str


class HomogeneousChunkScheduler:
    """Group work by signature, then emit fixed-size stage-local chunks.

    This keeps the scheduler substrate separate from env or backend-specific
    orchestration. The current implementation is synchronous and deterministic,
    but the boundary matches a future async send/recv style runtime.
    """

    def schedule(
        self,
        *,
        work_items: list[ExecutionWorkItem],
        policy: ExecutionBatchPolicy,
        chunk_id_prefix: str,
        latent_join: Callable[[list[Any]], Any],
        action_join: Callable[[list[Any]], Any],
    ) -> tuple[list[ExecutionChunk], list[SchedulerDecision]]:
        if not work_items:
            return [], []

        max_chunk_size = max(1, policy.max_chunk_size)
        grouped_items: OrderedDict[BatchSignature, list[ExecutionWorkItem]] = OrderedDict()
        for item in work_items:
            grouped_items.setdefault(item.entity.batch_signature, []).append(item)

        chunks: list[ExecutionChunk] = []
        decisions: list[SchedulerDecision] = []
        chunk_index = 0
        for signature, signature_items in grouped_items.items():
            signature_chunk_count = 0
            for offset in range(0, len(signature_items), max_chunk_size):
                chunk_items = signature_items[offset:offset + max_chunk_size]
                if len(chunk_items) < policy.min_ready_size and not policy.allow_partial_batch:
                    continue
                chunks.append(
                    ExecutionChunk(
                        chunk_id=f"{chunk_id_prefix}:{chunk_index}",
                        signature=signature,
                        entities=[item.entity for item in chunk_items],
                        latent_batch=latent_join([item.latent_item for item in chunk_items]),
                        action_batch=action_join([item.action_item for item in chunk_items]),
                    )
                )
                signature_chunk_count += 1
                chunk_index += 1
            decisions.append(
                SchedulerDecision(
                    signature=signature,
                    ready_count=len(signature_items),
                    chunk_count=signature_chunk_count,
                    policy_mode=policy.mode,
                )
            )
        return chunks, decisions


def build_execution_chunks(
    *,
    signature: BatchSignature,
    entities: list[Any],
    latent_items: list[Any],
    action_items: list[Any],
    policy: ExecutionBatchPolicy,
    chunk_id_prefix: str,
    latent_join: Callable[[list[Any]], Any],
    action_join: Callable[[list[Any]], Any],
) -> list[ExecutionChunk]:
    """Compatibility helper for call sites that already precomputed one signature."""

    if len(entities) != len(latent_items) or len(entities) != len(action_items):
        raise ValueError("entities, latent_items, and action_items must have the same length")

    work_items = [
        ExecutionWorkItem(
            entity=entity,
            latent_item=latent_item,
            action_item=action_item,
        )
        for entity, latent_item, action_item in zip(entities, latent_items, action_items, strict=True)
    ]
    chunks, _ = HomogeneousChunkScheduler().schedule(
        work_items=work_items,
        policy=policy,
        chunk_id_prefix=chunk_id_prefix,
        latent_join=latent_join,
        action_join=action_join,
    )

    if any(chunk.signature != signature for chunk in chunks):
        raise ValueError("build_execution_chunks received entities with mismatched batch signatures")
    return chunks
