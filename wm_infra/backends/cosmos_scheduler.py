"""Chunk scheduler for homogeneous Cosmos sample-production work."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from wm_infra.backends.cosmos_runtime import CosmosExecutionChunk, CosmosExecutionEntity, expected_occupancy


@dataclass(slots=True)
class CosmosSchedulerDecision:
    """Chunk decision plus coarse scheduler metadata."""

    chunk: CosmosExecutionChunk
    scheduler_inputs: dict[str, float | int | str | bool]


class CosmosChunkScheduler:
    """Group homogeneous Cosmos entities into ECS-style chunks."""

    def __init__(self, max_chunk_size: int = 4) -> None:
        self.max_chunk_size = max_chunk_size

    def schedule(self, entities: Iterable[CosmosExecutionEntity], *, estimated_units: float) -> list[CosmosSchedulerDecision]:
        groups: dict[tuple[str, object], list[CosmosExecutionEntity]] = defaultdict(list)
        for entity in entities:
            groups[(entity.queue_lane.value, entity.batch_signature)].append(entity)

        decisions: list[CosmosSchedulerDecision] = []
        for (lane, signature), items in groups.items():
            items.sort(key=lambda item: (-item.priority, item.sample_id))
            for offset in range(0, len(items), self.max_chunk_size):
                chunk_entities = items[offset:offset + self.max_chunk_size]
                occupancy = expected_occupancy(len(chunk_entities), self.max_chunk_size)
                chunk = CosmosExecutionChunk(
                    chunk_id=f"{signature.stage}:{lane}:{offset // self.max_chunk_size}",
                    signature=signature,
                    entity_ids=[entity.entity_id for entity in chunk_entities],
                    expected_occupancy=occupancy,
                    estimated_units=estimated_units,
                )
                decisions.append(
                    CosmosSchedulerDecision(
                        chunk=chunk,
                        scheduler_inputs={
                            "queue_lane": lane,
                            "batch_signature_cardinality": len(groups),
                            "expected_occupancy": occupancy,
                            "estimated_units": estimated_units,
                            "has_reference": signature.has_reference,
                        },
                    )
                )

        decisions.sort(key=lambda item: (-item.chunk.expected_occupancy, item.chunk.chunk_id))
        return decisions
