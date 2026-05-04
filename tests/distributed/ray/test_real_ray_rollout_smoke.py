"""Env-gated real checkpoint smoke for Ray rollout workers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _require_real_ray_smoke() -> None:
    if os.environ.get("VRL_RAY_REAL_SMOKE") != "1":
        pytest.skip("VRL_RAY_REAL_SMOKE=1 is required for real Ray rollout smoke")


def _require_checkpoint(env_var: str) -> str:
    path = os.environ.get(env_var)
    if not path:
        pytest.skip(f"{env_var} is not set; missing real checkpoint path")
    if not Path(path).exists():
        pytest.skip(f"{env_var} points to missing path: {path}")
    return path


def _require_cuda_for_workers(num_workers: int):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; real Ray rollout smoke cannot run")
    if torch.cuda.device_count() < num_workers:
        pytest.skip(
            f"real Ray rollout smoke needs {num_workers} CUDA devices, "
            f"found {torch.cuda.device_count()}",
        )
    return torch


@pytest.mark.asyncio
async def test_real_sd3_5_ray_rollout_smoke() -> None:
    _require_real_ray_smoke()
    ray = pytest.importorskip("ray")
    checkpoint = _require_checkpoint("VRL_SD3_5_CHECKPOINT")
    num_workers = int(os.environ.get("VRL_RAY_REAL_NUM_WORKERS", "1"))
    torch = _require_cuda_for_workers(num_workers)

    from vrl.config.loader import load_config
    from vrl.distributed.ray import RayRolloutLauncher
    from vrl.engine.generation import GenerationRequest
    from vrl.rollouts.runtime_inputs import build_rollout_runtime_inputs

    cfg = load_config(
        "experiment/sd3_5_ocr_grpo",
        overrides=[
            "distributed.backend=ray",
            f"distributed.rollout.num_workers={num_workers}",
            "distributed.rollout.gpus_per_worker=1",
            "distributed.rollout.cpus_per_worker=1",
            "distributed.rollout.sync_trainable_state=disabled",
            f"model.path={checkpoint}",
            "model.use_lora=false",
            "model.torch_compile.enable=false",
            "sampling.num_steps=2",
            "sampling.height=512",
            "sampling.width=512",
            "rollout.sample_batch_size=1",
        ],
    )
    inputs = build_rollout_runtime_inputs(
        cfg,
        "sd3_5",
        weight_dtype=torch.bfloat16,
        executor_kwargs={"sample_batch_size": 1},
    )
    assert inputs is not None

    runtime = RayRolloutLauncher(
        ray_init_kwargs={
            "include_dashboard": False,
            "ignore_reinit_error": True,
            "log_to_driver": False,
        },
    ).launch(cfg, inputs.runtime_spec, inputs.gatherer)
    try:
        output = await runtime.generate(
            GenerationRequest(
                request_id="real-sd3-ray-smoke",
                family="sd3_5",
                task="t2i",
                prompts=["a red square"],
                samples_per_prompt=1,
                sampling={
                    "num_steps": 2,
                    "guidance_scale": float(cfg.sampling.guidance_scale),
                    "height": 512,
                    "width": 512,
                    "noise_level": float(cfg.rollout.noise_level),
                    "cfg": bool(cfg.sampling.cfg),
                    "sample_batch_size": 1,
                    "sde_window_size": int(cfg.rollout.sde.window_size),
                    "sde_window_range": list(cfg.rollout.sde.window_range),
                    "same_latent": False,
                    "max_sequence_length": 256,
                    "seed": 7,
                    "return_kl": False,
                },
            ),
        )
    finally:
        await runtime.shutdown()
        ray.shutdown()

    assert output.error is None
    assert output.output is not None
    assert output.metrics is not None
    assert output.metrics.micro_batches == 1
