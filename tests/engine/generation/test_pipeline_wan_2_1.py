"""Unit tests for Wan_2_1PipelineExecutor.

These tests use a stubbed Wan policy that implements the minimal
``encode_prompt``/``prepare_sampling``/``forward_step``/``decode_latents``
contract required by the executor — no diffusers / Wan weights needed.

Wan differs from SD3.5 in two structural ways exercised here:
- 5D latents ``[B, C, T_v, H, W]`` (Wan VAE temporal axis)
- text encoder returns ``prompt_embeds`` only (no pooled CLIP)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
import torch

from vrl.engine.generation import (
    GenerationIdFactory,
    GenerationRequest,
)
from vrl.models.families.wan_2_1.executor import Wan_2_1PipelineExecutor

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubScheduler:
    """Mock scheduler keyed by step index (not by sigma value)."""

    def __init__(self, num_steps: int) -> None:
        self._num_steps = num_steps
        self.sigmas = torch.linspace(1.0, 0.0, num_steps + 2)
        self.timesteps = torch.tensor(
            [self.sigmas[i].item() for i in range(num_steps)],
            dtype=torch.float32,
        )

    def index_for_timestep(self, t: torch.Tensor) -> int:
        diffs = (self.sigmas - t).abs()
        idx = int(diffs.argmin().item())
        return min(idx, len(self.sigmas) - 2)

    def set_timesteps(self, n: int, device: Any = None) -> None:
        del device
        self._num_steps = n
        self.sigmas = torch.linspace(1.0, 0.0, n + 2)
        self.timesteps = torch.tensor(
            [self.sigmas[i].item() for i in range(n)],
            dtype=torch.float32,
        )


@dataclass
class _StubWanState:
    latents: torch.Tensor  # 5D: [B, C, T_v, H, W]
    timesteps: torch.Tensor
    scheduler: _StubScheduler
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    guidance_scale: float
    do_cfg: bool
    seed: int


@dataclass
class _StubWanPolicy:
    """Bare-minimum Wan policy stub.

    - ``encode_prompt`` returns only ``prompt_embeds`` (+ optional neg) —
      no pooled CLIP, matching Wan's text-encoder contract.
    - ``prepare_sampling`` builds 5D latents ``[B, C, T_v, H, W]``.
    - ``forward_step`` returns a deterministic linear function of latents.
    - ``decode_latents`` reduces to a video tensor ``[B, 3, T_v, H, W]``.
    """

    embed_dim: int = 8
    latent_channels: int = 4
    num_frames_latent: int = 3
    family: str = "wan_2_1-stub"
    forward_calls: int = 0
    encode_calls: list[str] = field(default_factory=list)

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del negative_prompt, kwargs
        if isinstance(prompt, list):
            prompt = prompt[0]
        self.encode_calls.append(prompt)
        h = abs(hash(prompt)) % (2**31)
        gen = torch.Generator().manual_seed(h)
        prompt_embeds = torch.randn(1, 4, self.embed_dim, generator=gen)
        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": torch.zeros_like(prompt_embeds),
        }

    def prepare_sampling(
        self,
        request: Any,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> _StubWanState:
        del kwargs
        scheduler = _StubScheduler(request.num_steps)
        bsz = encoded["prompt_embeds"].shape[0]
        H = max(1, request.height // 32)
        W = max(1, request.width // 32)
        seed = request.seed if request.seed is not None else 0
        gen = torch.Generator().manual_seed(int(seed))
        latents = torch.randn(
            bsz,
            self.latent_channels,
            self.num_frames_latent,
            H,
            W,
            generator=gen,
        )
        return _StubWanState(
            latents=latents,
            timesteps=scheduler.timesteps,
            scheduler=scheduler,
            prompt_embeds=encoded["prompt_embeds"],
            negative_prompt_embeds=encoded.get("negative_prompt_embeds"),
            guidance_scale=request.guidance_scale,
            do_cfg=request.guidance_scale > 1.0,
            seed=int(seed),
        )

    def forward_step(
        self,
        state: _StubWanState,
        step_idx: int,
        *,
        model: Any = None,
    ) -> dict[str, torch.Tensor]:
        del model
        self.forward_calls += 1
        noise_pred = torch.sin(state.latents) * (step_idx + 1) * 0.01
        return {
            "noise_pred": noise_pred,
            "noise_pred_cond": noise_pred,
            "noise_pred_uncond": torch.zeros_like(noise_pred),
        }

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        # 5D latents: [B, C, T_v, H, W] -> [B, 3, T_v, H*8, W*8]
        B, C, T_v, H, W = latents.shape
        rgb = latents[:, :3] if C >= 3 else latents.repeat(1, 3, 1, 1, 1)[:, :3]
        # Upsample only the spatial dims, preserve T_v.
        rgb_flat = rgb.reshape(B * T_v, 3, H, W)
        up = torch.nn.functional.interpolate(
            rgb_flat, scale_factor=8, mode="nearest",
        )
        return up.reshape(B, T_v, 3, H * 8, W * 8).permute(0, 2, 1, 3, 4)

    def export_batch_context(self, state: _StubWanState) -> dict[str, Any]:
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
            "model_family": self.family,
        }

    def export_training_extras(self, state: _StubWanState) -> dict[str, Any]:
        return {
            "prompt_embeds": state.prompt_embeds,
            "negative_prompt_embeds": state.negative_prompt_embeds,
        }


def _request(
    *,
    prompts: list[str] | None = None,
    samples_per_prompt: int = 4,
    num_steps: int = 4,
    height: int = 64,
    width: int = 64,
    num_frames: int = 9,
    seed: int | None = 42,
    sample_batch_size: int = 8,
) -> GenerationRequest:
    return GenerationRequest(
        request_id="wan_2_1-test",
        family="wan_2_1",
        task="t2v",
        prompts=prompts or ["a red cube spinning"],
        samples_per_prompt=samples_per_prompt,
        sampling={
            "num_steps": num_steps,
            "guidance_scale": 4.5,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "noise_level": 1.0,
            "cfg": True,
            "sample_batch_size": sample_batch_size,
            "sde_window_size": 0,
            "sde_window_range": [0, num_steps],
            "same_latent": False,
            "max_sequence_length": 512,
            "seed": seed,
        },
        return_artifacts={
            "output", "rollout_trajectory_data", "denoising_env",
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_executor_forward_shapes_for_two_prompts_x_four_samples() -> None:
    """forward(2 prompts x 4 samples) -> OutputBatch with 8 specs + correct shapes."""
    policy = _StubWanPolicy()
    executor = Wan_2_1PipelineExecutor(policy, sample_batch_size=8)
    request = _request(
        prompts=["a red cube spinning", "a blue sphere bouncing"],
        samples_per_prompt=4,
        num_steps=3,
        height=32,
        width=32,
        seed=42,
        sample_batch_size=8,
    )
    specs = GenerationIdFactory().build_sample_specs(request)
    assert len(specs) == 8

    output = executor.forward(request, specs)

    assert output.error is None
    assert output.request_id == request.request_id
    assert output.family == "wan_2_1"
    assert output.task == "t2v"
    assert len(output.sample_specs) == 8

    # output: decoded video [B, 3, T_v, H, W]
    assert isinstance(output.output, torch.Tensor)
    assert output.output.ndim == 5
    assert output.output.shape[0] == 8
    assert output.output.shape[1] == 3

    # rollout_trajectory_data
    rt = output.rollout_trajectory_data
    assert rt is not None
    # log_probs: [B, T] = [8, 3]
    assert rt.rollout_log_probs.shape == (8, 3)
    # dit_trajectory.latents (5D latents stacked on step dim)
    # [B, T_steps, C, T_v, H, W]
    assert rt.dit_trajectory.latents.ndim == 6
    assert rt.dit_trajectory.latents.shape[0] == 8
    assert rt.dit_trajectory.latents.shape[1] == 3
    # timesteps: [B, T]
    assert rt.dit_trajectory.timesteps.shape == (8, 3)

    # denoising_env carries replay context
    env = rt.denoising_env
    assert env is not None
    assert env.extra["actions"].shape == rt.dit_trajectory.latents.shape
    assert env.extra["kl"].shape == (8, 3)
    assert "training_extras" in env.extra
    assert "context" in env.extra
    assert env.extra["context"]["guidance_scale"] == 4.5

    # encode_prompt called once per prompt (not once per chunk).
    assert policy.encode_calls == [
        "a red cube spinning", "a blue sphere bouncing",
    ]


def test_executor_micro_batches_when_group_exceeds_sample_batch_size() -> None:
    """group_size=12 with sample_batch_size=4 → 3 chunks."""
    policy = _StubWanPolicy()
    executor = Wan_2_1PipelineExecutor(policy, sample_batch_size=4)
    request = _request(
        prompts=["a red cube spinning"],
        samples_per_prompt=12,
        num_steps=2,
        height=32,
        width=32,
        seed=7,
        sample_batch_size=4,
    )
    specs = GenerationIdFactory().build_sample_specs(request)
    assert len(specs) == 12

    output = executor.forward(request, specs)
    assert output.metrics is not None
    assert output.metrics.micro_batches == 3
    assert output.metrics.num_samples == 12
    assert output.metrics.num_steps == 2
    # 3 chunks x 2 steps = 6 forward_step calls
    assert policy.forward_calls == 6
    assert output.output.shape[0] == 12


def test_executor_request_id_round_trip() -> None:
    """request_id flows through unchanged."""
    policy = _StubWanPolicy()
    executor = Wan_2_1PipelineExecutor(policy)
    request = _request(num_steps=2, height=32, width=32)
    specs = GenerationIdFactory().build_sample_specs(request)
    output = executor.forward(request, specs)
    assert output.request_id == "wan_2_1-test"


def test_executor_workload_signature_matches_request() -> None:
    """workload_signature derives from the request sampling dict."""
    policy = _StubWanPolicy()
    executor = Wan_2_1PipelineExecutor(policy)
    request = _request(num_steps=4, height=128, width=64, num_frames=17)
    sig = executor.workload_signature(request)
    assert sig.family == "wan_2_1"
    assert sig.task == "t2v"
    assert sig.height == 128
    assert sig.width == 64
    assert sig.num_steps == 4
    assert sig.num_frames == 17


def test_executor_seed_reproducibility() -> None:
    """Same seed + same prompts → bitwise-identical log_probs and latents."""
    policy_a = _StubWanPolicy()
    policy_b = _StubWanPolicy()
    executor_a = Wan_2_1PipelineExecutor(policy_a)
    executor_b = Wan_2_1PipelineExecutor(policy_b)
    req_a = _request(num_steps=3, height=32, width=32, seed=1234)
    req_b = _request(num_steps=3, height=32, width=32, seed=1234)
    specs_a = GenerationIdFactory().build_sample_specs(req_a)
    specs_b = GenerationIdFactory().build_sample_specs(req_b)

    out_a = executor_a.forward(req_a, specs_a)
    out_b = executor_b.forward(req_b, specs_b)

    assert torch.equal(out_a.output, out_b.output)
    assert torch.equal(
        out_a.rollout_trajectory_data.rollout_log_probs,
        out_b.rollout_trajectory_data.rollout_log_probs,
    )
    assert torch.equal(
        out_a.rollout_trajectory_data.dit_trajectory.latents,
        out_b.rollout_trajectory_data.dit_trajectory.latents,
    )


def test_executor_kl_window_zeroes_kl_when_disabled() -> None:
    """When sampling.return_kl=False, kl tensor is zero."""
    policy = _StubWanPolicy()
    executor = Wan_2_1PipelineExecutor(policy)
    request = _request(num_steps=3, height=32, width=32, samples_per_prompt=2)
    request.sampling["return_kl"] = False
    specs = GenerationIdFactory().build_sample_specs(request)

    output = executor.forward(request, specs)
    kl = output.rollout_trajectory_data.denoising_env.extra["kl"]
    assert torch.all(kl == 0)


def test_executor_kl_window_populated_when_enabled() -> None:
    """When sampling.return_kl=True, kl tensor matches |log_prob|."""
    policy = _StubWanPolicy()
    executor = Wan_2_1PipelineExecutor(policy)
    request = _request(num_steps=3, height=32, width=32, samples_per_prompt=2)
    request.sampling["return_kl"] = True
    specs = GenerationIdFactory().build_sample_specs(request)

    output = executor.forward(request, specs)
    kl = output.rollout_trajectory_data.denoising_env.extra["kl"]
    log_probs = output.rollout_trajectory_data.rollout_log_probs
    assert torch.equal(kl, log_probs.abs())


@pytest.mark.parametrize("samples_per_prompt", [1, 3, 5])
def test_executor_arbitrary_group_sizes(samples_per_prompt: int) -> None:
    """Various group sizes produce correct B dimension."""
    policy = _StubWanPolicy()
    executor = Wan_2_1PipelineExecutor(policy, sample_batch_size=4)
    request = _request(
        samples_per_prompt=samples_per_prompt,
        num_steps=2,
        height=32,
        width=32,
    )
    specs = GenerationIdFactory().build_sample_specs(request)
    output = executor.forward(request, specs)
    assert output.output.shape[0] == samples_per_prompt
    assert output.rollout_trajectory_data.rollout_log_probs.shape[0] == samples_per_prompt
