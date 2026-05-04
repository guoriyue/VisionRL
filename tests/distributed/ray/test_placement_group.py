"""Tests for Ray rollout placement group helpers."""

from __future__ import annotations

import pytest


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


def test_placement_group_cpu_only_local_mode(ray_local) -> None:
    from vrl.distributed.ray import create_rollout_placement_group
    from vrl.rollouts.runtime.config import RolloutBackendConfig

    placement = create_rollout_placement_group(
        RolloutBackendConfig(
            backend="ray",
            num_workers=2,
            gpus_per_worker=0.0,
            cpus_per_worker=1.0,
        ),
    )

    assert len(placement.ordered_bundle_indices) == 2
    assert sorted(placement.ordered_bundle_indices) == [0, 1]
