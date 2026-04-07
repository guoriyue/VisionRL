from __future__ import annotations

import torch

from wm_infra.core.execution import (
    BatchSignature,
    ExecutionBatchPolicy,
    ExecutionEntity,
    build_execution_chunks,
    summarize_execution_chunks,
)


def test_build_execution_chunks_respects_shared_batch_policy() -> None:
    signature = BatchSignature(
        stage="env_step",
        latent_shape=(1, 1),
        action_dim=3,
        dtype="float32",
        device="cpu",
        needs_decode=False,
    )
    policy = ExecutionBatchPolicy(mode="sync", max_chunk_size=2, min_ready_size=1, return_when_ready_count=2)
    entities = [
        ExecutionEntity(
            entity_id=f"env-{index}:env_step:0",
            rollout_id=f"env-{index}",
            stage="env_step",
            step_idx=0,
            batch_signature=signature,
        )
        for index in range(5)
    ]
    latent_items = [torch.full((1, 1, 1), float(index)) for index in range(5)]
    action_items = [torch.eye(3, dtype=torch.float32)[index % 3:index % 3 + 1] for index in range(5)]

    chunks = build_execution_chunks(
        signature=signature,
        entities=entities,
        latent_items=latent_items,
        action_items=action_items,
        policy=policy,
        chunk_id_prefix="env_step",
        latent_join=lambda items: torch.cat(items, dim=0),
        action_join=lambda items: torch.cat(items, dim=0),
    )
    summary = summarize_execution_chunks(chunks, policy=policy)

    assert [chunk.size for chunk in chunks] == [2, 2, 1]
    assert chunks[0].latent_batch.shape == (2, 1, 1)
    assert chunks[0].action_batch.shape == (2, 3)
    assert summary["batch_policy"]["mode"] == "sync"
    assert summary["batch_policy"]["return_when_ready_count"] == 2
    assert summary["avg_chunk_fill_ratio"] == 5 / 6
