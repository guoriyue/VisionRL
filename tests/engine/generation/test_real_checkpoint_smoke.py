"""Env-gated smoke tests for real generation checkpoints.

These tests are intentionally skipped by default because they require large
local checkpoints and CUDA memory. A clear skip is still useful: CI and local
runs show exactly which real checkpoint DoD remains unverified.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest


class _ZeroReward:
    async def score(self, rollout: Any) -> float:
        return 0.0


def _require_checkpoint(env_var: str) -> str:
    path = os.environ.get(env_var)
    if not path:
        pytest.skip(f"{env_var} is not set; missing real checkpoint path")
    if not Path(path).exists():
        pytest.skip(f"{env_var} points to missing path: {path}")
    if os.environ.get("VRL_REAL_CHECKPOINT_SMOKE") != "1":
        pytest.skip(
            "VRL_REAL_CHECKPOINT_SMOKE=1 is required for expensive real "
            f"checkpoint smoke; {env_var}={path}"
        )
    return path


def _require_cuda() -> Any:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; real checkpoint smoke cannot run")
    return torch


def _assert_experience_batch(batch: Any) -> None:
    assert batch.observations is not None
    assert batch.actions is not None
    assert batch.rewards is not None
    assert batch.dones is not None
    assert batch.group_ids is not None
    assert batch.extras["log_probs"] is not None


@pytest.mark.asyncio
async def test_real_sd3_5_checkpoint_smoke() -> None:
    path = _require_checkpoint("VRL_SD3_5_CHECKPOINT")
    torch = _require_cuda()

    from vrl.models.families.sd3_5.builder import build_sd3_5_runtime_bundle
    from vrl.models.runtime import RuntimeBuildSpec
    from vrl.rollouts.collectors.sd3_5 import SD3_5Collector, SD3_5CollectorConfig

    spec = RuntimeBuildSpec(
        model_name_or_path=path,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        backend_preference=("diffusers",),
        task_variant="t2i",
        use_lora=False,
        scheduler_config={"num_steps": 2},
    )
    bundle = build_sd3_5_runtime_bundle(spec)
    collector = SD3_5Collector(
        bundle.policy,
        _ZeroReward(),
        SD3_5CollectorConfig(
            num_steps=2,
            height=512,
            width=512,
            sample_batch_size=1,
        ),
    )
    try:
        batch = await collector.collect(["a red square"], group_size=1, seed=7)
    finally:
        await collector.shutdown()

    _assert_experience_batch(batch)
    assert batch.videos is not None
    assert batch.extras["timesteps"] is not None


@pytest.mark.asyncio
async def test_real_wan_2_1_checkpoint_smoke() -> None:
    path = _require_checkpoint("VRL_WAN_2_1_CHECKPOINT")
    torch = _require_cuda()

    from vrl.models.families.wan_2_1.builder import build_wan_2_1_runtime_bundle
    from vrl.models.runtime import RuntimeBuildSpec
    from vrl.rollouts.collectors.wan_2_1 import (
        Wan_2_1Collector,
        Wan_2_1CollectorConfig,
    )

    spec = RuntimeBuildSpec(
        model_name_or_path=path,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        backend_preference=("diffusers",),
        task_variant="t2v",
        use_lora=False,
        scheduler_config={"num_steps": 2},
    )
    bundle = build_wan_2_1_runtime_bundle(spec)
    collector = Wan_2_1Collector(
        bundle.policy,
        _ZeroReward(),
        Wan_2_1CollectorConfig(
            num_steps=2,
            height=240,
            width=416,
            num_frames=33,
            sample_batch_size=1,
        ),
    )
    try:
        batch = await collector.collect(["a short video of a red cube"], group_size=1, seed=7)
    finally:
        await collector.shutdown()

    _assert_experience_batch(batch)
    assert batch.videos is not None
    assert batch.extras["timesteps"] is not None


@pytest.mark.asyncio
async def test_real_janus_pro_checkpoint_smoke() -> None:
    path = _require_checkpoint("VRL_JANUS_PRO_CHECKPOINT")
    _require_cuda()

    from vrl.models.families.janus_pro.policy import JanusProConfig, JanusProPolicy
    from vrl.rollouts.collectors.janus_pro import (
        JanusProCollector,
        JanusProCollectorConfig,
    )

    policy = JanusProPolicy(
        JanusProConfig(
            model_path=path,
            use_lora=False,
            device="cuda",
        ),
    )
    collector = JanusProCollector(
        policy,
        _ZeroReward(),
        JanusProCollectorConfig(n_samples_per_prompt=1),
    )
    try:
        batch = await collector.collect(["a red square"], group_size=1, seed=7)
    finally:
        await collector.shutdown()

    _assert_experience_batch(batch)
    assert batch.extras["token_ids"] is not None
    assert batch.extras["token_log_probs"] is not None


@pytest.mark.asyncio
async def test_real_nextstep_1_checkpoint_smoke() -> None:
    path = _require_checkpoint("VRL_NEXTSTEP_1_CHECKPOINT")
    vae_path = _require_checkpoint("VRL_NEXTSTEP_1_VAE_CHECKPOINT")
    _require_cuda()

    from vrl.models.families.nextstep_1.policy import (
        NextStep1Config,
        NextStep1Policy,
    )
    from vrl.rollouts.collectors.nextstep_1 import (
        NextStep1Collector,
        NextStep1CollectorConfig,
    )

    policy = NextStep1Policy(
        NextStep1Config(
            model_path=path,
            vae_path=vae_path,
            use_lora=False,
            device="cuda",
        ),
    )
    collector = NextStep1Collector(
        policy,
        _ZeroReward(),
        NextStep1CollectorConfig(n_samples_per_prompt=1),
    )
    try:
        batch = await collector.collect(["a red square"], group_size=1, seed=7)
    finally:
        await collector.shutdown()

    _assert_experience_batch(batch)
    assert batch.extras["token_log_probs"] is not None
    assert batch.extras["saved_noise"] is not None
