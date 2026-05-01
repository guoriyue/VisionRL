"""Tests for the P0 Ray collector wrapper."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest


def make_stub_collector(cfg: Any | None = None, family: str | None = None):
    import torch

    from vrl.rollouts.types import ExperienceBatch

    class _StubCollector:
        async def collect(self, prompts: list[str], **kwargs: Any) -> ExperienceBatch:
            group_size = int(kwargs["group_size"])
            prompt = prompts[0]
            seed = kwargs.get("seed")
            value = float(len(prompt) + (seed or 0))
            return ExperienceBatch(
                observations=torch.full((group_size, 1), value),
                actions=torch.full((group_size, 1), value + 1.0),
                rewards=torch.arange(group_size, dtype=torch.float32) + value,
                dones=torch.ones(group_size, dtype=torch.bool),
                group_ids=torch.zeros(group_size, dtype=torch.long),
                extras={
                    "log_probs": torch.full((group_size, 1), value),
                    "family": family or "stub",
                },
                context={"factory_cfg": bool(cfg)},
                prompts=[prompt] * group_size,
            )

    return _StubCollector()


@pytest.fixture()
def ray_local():
    ray = pytest.importorskip("ray")
    ray.init(
        local_mode=True,
        num_cpus=4,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
    )
    try:
        yield ray
    finally:
        ray.shutdown()


def _jobs():
    from vrl.distributed.ray import CollectorJob

    return [
        CollectorJob(prompt_index=10, prompt="aa", group_size=2, seed=1),
        CollectorJob(prompt_index=11, prompt="bbb", group_size=2, seed=2),
        CollectorJob(prompt_index=12, prompt="c", group_size=2, seed=3),
        CollectorJob(prompt_index=13, prompt="dddd", group_size=2, seed=4),
    ]


def test_split_jobs_round_robin() -> None:
    from vrl.distributed.ray import RayCollectorManager

    shards = RayCollectorManager.split_jobs(_jobs(), 2)
    assert [[job.prompt_index for job in shard] for shard in shards] == [
        [10, 12],
        [11, 13],
    ]


def test_collector_actor_collect_jobs_direct() -> None:
    from vrl.distributed.ray import RayCollectorActor

    actor = RayCollectorActor(
        cfg={"collector_factory": make_stub_collector},
        family="stub",
    )
    batches = actor.collect_jobs(_jobs()[:1])

    assert len(batches) == 1
    batch = batches[0]
    assert batch.group_ids.tolist() == [10, 10]
    assert batch.prompts == ["aa", "aa"]
    assert batch.context["ray_prompt_index"] == 10


def test_placement_group_cpu_only_local_mode(ray_local) -> None:
    from vrl.distributed.ray import RayConfig, create_rollout_placement_group

    placement = create_rollout_placement_group(
        RayConfig(
            enable=True,
            num_rollout_workers=2,
            gpus_per_rollout_worker=0.0,
            cpus_per_rollout_worker=1.0,
        ),
    )

    assert len(placement.ordered_bundle_indices) == 2
    assert sorted(placement.ordered_bundle_indices) == [0, 1]


def test_collector_manager_collect_stacks_batches(ray_local) -> None:
    from vrl.distributed.ray import (
        RayCollectorManager,
        RayConfig,
        create_rollout_placement_group,
    )

    config = RayConfig(
        enable=True,
        num_rollout_workers=2,
        gpus_per_rollout_worker=0.0,
        cpus_per_rollout_worker=1.0,
    )
    placement = create_rollout_placement_group(config)
    manager = RayCollectorManager(
        cfg={
            "collector_factory": (
                "tests.distributed.ray.test_collector_manager:make_stub_collector"
            ),
        },
        family="stub",
        config=config,
        placement=placement,
    )

    try:
        batch = asyncio.run(manager.collect(_jobs()))
    finally:
        manager.close()

    assert batch.rewards.shape[0] == 8
    assert sorted(batch.group_ids.tolist()) == [10, 10, 11, 11, 12, 12, 13, 13]
    assert sorted(batch.prompts or []) == ["aa", "aa", "bbb", "bbb", "c", "c", "dddd", "dddd"]
    assert batch.extras["log_probs"].shape == (8, 1)


def test_collector_manager_skips_empty_worker_shards(ray_local) -> None:
    from vrl.distributed.ray import (
        RayCollectorManager,
        RayConfig,
        create_rollout_placement_group,
    )

    config = RayConfig(
        enable=True,
        num_rollout_workers=3,
        gpus_per_rollout_worker=0.0,
        cpus_per_rollout_worker=1.0,
    )
    placement = create_rollout_placement_group(config)
    manager = RayCollectorManager(
        cfg={"collector_factory": make_stub_collector},
        family="stub",
        config=config,
        placement=placement,
    )

    try:
        batch = asyncio.run(manager.collect(_jobs()[:1]))
    finally:
        manager.close()

    assert batch.group_ids.tolist() == [10, 10]


def test_collector_manager_rejects_empty_jobs(ray_local) -> None:
    from vrl.distributed.ray import (
        RayCollectorManager,
        RayConfig,
        create_rollout_placement_group,
    )

    config = RayConfig(
        enable=True,
        num_rollout_workers=1,
        gpus_per_rollout_worker=0.0,
        cpus_per_rollout_worker=1.0,
    )
    placement = create_rollout_placement_group(config)
    manager = RayCollectorManager(
        cfg={"collector_factory": make_stub_collector},
        family="stub",
        config=config,
        placement=placement,
    )

    try:
        with pytest.raises(ValueError, match="at least one job"):
            asyncio.run(manager.collect([]))
    finally:
        manager.close()
