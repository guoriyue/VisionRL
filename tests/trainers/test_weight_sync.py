"""Tests for trainer-to-rollout weight sync adapters."""

from __future__ import annotations

import asyncio
from typing import Any

import torch

from vrl.trainers.weight_sync import (
    RayRuntimeWeightSyncer,
    build_runtime_weight_syncer,
)


class _RuntimeWithSync:
    def __init__(self) -> None:
        self.current_policy_version = 0
        self.weight_sync = object()
        self.calls: list[tuple[dict[str, Any], int]] = []

    async def update_weights(self, state_ref: dict[str, Any], policy_version: int) -> None:
        self.calls.append((state_ref, policy_version))
        self.current_policy_version = policy_version


class _RuntimeWithoutSync:
    async def update_weights(self, state_ref: dict[str, Any], policy_version: int) -> None:
        del state_ref, policy_version


def test_ray_runtime_weight_syncer_pushes_cpu_state_with_monotonic_versions() -> None:
    runtime = _RuntimeWithSync()
    syncer = RayRuntimeWeightSyncer(runtime)

    asyncio.run(syncer.push({"weight": torch.ones(2)}))
    asyncio.run(syncer.push({"weight": torch.full((2,), 2.0)}))

    assert [version for _, version in runtime.calls] == [1, 2]
    assert runtime.current_policy_version == 2
    assert all(call[0]["weight"].device.type == "cpu" for call in runtime.calls)
    assert torch.equal(asyncio.run(syncer.pull())["weight"], torch.full((2,), 2.0))


def test_build_runtime_weight_syncer_requires_runtime_weight_sync_handle() -> None:
    assert build_runtime_weight_syncer(_RuntimeWithSync()) is not None
    assert build_runtime_weight_syncer(_RuntimeWithoutSync()) is None
