"""Unit tests for CosmosPipelineExecutor.

These tests use a stubbed CosmosPredict2Policy that implements the minimal
``encode_prompt``/``prepare_sampling``/``forward_step``/``decode_latents``
contract required by the executor — no diffusers / Cosmos weights needed.

Cosmos differs from SD3.5 in three ways the stubs reflect:

- 5D latents ``[B, C, T, H, W]`` (Video2World)
- ``reference_image`` flows through ``encode_prompt`` and
  ``prepare_sampling``
- Decoded output is a 5D video tensor ``[B, C, T, H, W]``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from vrl.engine import (
    GenerationIdFactory,
    GenerationRequest,
)
from vrl.models.families.cosmos.executor import CosmosPipelineExecutor

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
class _StubCosmosState:
    latents: torch.Tensor
    timesteps: torch.Tensor
    scheduler: _StubScheduler
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    init_latents: torch.Tensor
    guidance_scale: float
    do_cfg: bool
    fps: int
    seed: int
    reference_image_seen: Any = None


@dataclass
class _StubCosmosPolicy:
    """Bare-minimum CosmosPredict2Policy stub.

    - ``encode_prompt`` returns deterministic fake embeds keyed off the
      prompt string and records the reference image seen.
    - ``prepare_sampling`` builds 5D latents from the per-chunk seed.
    - ``forward_step`` returns a deterministic linear function of the
      latents → ``sde_step_with_logprob`` math runs end-to-end.
    - ``decode_latents`` projects 5D latents → 3-channel video.
    """

    embed_dim: int = 8
    latent_channels: int = 4
    num_frames: int = 2
    family: str = "cosmos-stub"
    forward_calls: int = 0
    encode_calls: list[str] = field(default_factory=list)
    encode_reference_images: list[Any] = field(default_factory=list)
    prepare_reference_images: list[Any] = field(default_factory=list)

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del negative_prompt
        if isinstance(prompt, list):
            prompt = prompt[0]
        self.encode_calls.append(prompt)
        self.encode_reference_images.append(kwargs.get("reference_image"))
        h = abs(hash(prompt)) % (2**31)
        gen = torch.Generator().manual_seed(h)
        prompt_embeds = torch.randn(
            1,
            4,
            self.embed_dim,
            generator=gen,
        )
        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": torch.zeros_like(prompt_embeds),
            "reference_image": kwargs.get("reference_image"),
        }

    def prepare_sampling(
        self,
        request: Any,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> _StubCosmosState:
        ref_img = kwargs.get("reference_image", encoded.get("reference_image"))
        self.prepare_reference_images.append(ref_img)
        scheduler = _StubScheduler(request.num_steps)
        bsz = encoded["prompt_embeds"].shape[0]
        # 5D latent shape [B, C, T, H, W] (small for tests)
        H = max(1, request.height // 32)
        W = max(1, request.width // 32)
        T = self.num_frames
        seed = request.seed if request.seed is not None else 0
        gen = torch.Generator().manual_seed(int(seed))
        latents = torch.randn(
            bsz,
            self.latent_channels,
            T,
            H,
            W,
            generator=gen,
        )
        init_latents = torch.zeros_like(latents)
        return _StubCosmosState(
            latents=latents,
            timesteps=scheduler.timesteps,
            scheduler=scheduler,
            prompt_embeds=encoded["prompt_embeds"],
            negative_prompt_embeds=encoded.get("negative_prompt_embeds"),
            init_latents=init_latents,
            guidance_scale=request.guidance_scale,
            do_cfg=request.guidance_scale > 1.0,
            fps=request.fps or 16,
            seed=int(seed),
            reference_image_seen=ref_img,
        )

    def forward_step(
        self,
        state: _StubCosmosState,
        step_idx: int,
    ) -> dict[str, torch.Tensor]:
        self.forward_calls += 1
        # Deterministic: noise_pred = sin(latents) * (step_idx + 1) * 0.01
        noise_pred = torch.sin(state.latents) * (step_idx + 1) * 0.01
        return {
            "noise_pred": noise_pred,
            "noise_pred_cond": noise_pred,
            "noise_pred_uncond": torch.zeros_like(noise_pred),
        }

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        # 5D [B, C, T, H, W] → tile to 3 channels.
        B, C, T, H, W = latents.shape
        rgb = latents[:, :3] if C >= 3 else latents.repeat(1, 3, 1, 1, 1)[:, :3]
        # Upscale spatial dims 8x
        rgb = rgb.reshape(B * T, 3, H, W)
        rgb = torch.nn.functional.interpolate(
            rgb,
            scale_factor=8,
            mode="nearest",
        )
        H8, W8 = rgb.shape[-2], rgb.shape[-1]
        return rgb.reshape(B, 3, T, H8, W8)

    def export_batch_context(self, state: _StubCosmosState) -> dict[str, Any]:
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
            "model_family": self.family,
            "fps": state.fps,
        }

    def export_training_extras(self, state: _StubCosmosState) -> dict[str, Any]:
        return {
            "prompt_embeds": state.prompt_embeds,
            "negative_prompt_embeds": state.negative_prompt_embeds,
            "init_latents": state.init_latents,
        }


def _request(
    *,
    prompts: list[str] | None = None,
    samples_per_prompt: int = 4,
    num_steps: int = 4,
    height: int = 64,
    width: int = 64,
    num_frames: int = 2,
    fps: int = 16,
    seed: int | None = 42,
    sample_batch_size: int = 8,
    metadata: dict[str, Any] | None = None,
) -> GenerationRequest:
    return GenerationRequest(
        request_id="cosmos-test",
        family="cosmos",
        task="v2w",
        prompts=prompts or ["a red cube"],
        samples_per_prompt=samples_per_prompt,
        sampling={
            "num_steps": num_steps,
            "guidance_scale": 7.0,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "fps": fps,
            "cfg": True,
            "sample_batch_size": sample_batch_size,
            "sde_window_size": 0,
            "sde_window_range": [0, num_steps],
            "same_latent": False,
            "max_sequence_length": 512,
            "seed": seed,
        },
        return_artifacts={
            "output",
            "rollout_trajectory_data",
            "denoising_env",
        },
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_executor_forward_shapes_for_two_prompts_x_four_samples() -> None:
    """forward(2 prompts x 4 samples) -> OutputBatch with 8 specs + correct shapes."""
    policy = _StubCosmosPolicy()
    executor = CosmosPipelineExecutor(policy, sample_batch_size=8)
    request = _request(
        prompts=["a red cube", "a blue sphere"],
        samples_per_prompt=4,
        num_steps=3,
        height=32,
        width=32,
        num_frames=2,
        seed=42,
        sample_batch_size=8,
    )
    specs = GenerationIdFactory().build_sample_specs(request)
    assert len(specs) == 8

    output = executor.forward(request, specs)

    assert output.error is None
    assert output.request_id == request.request_id
    assert output.family == "cosmos"
    assert output.task == "v2w"
    assert len(output.sample_specs) == 8

    # output: decoded video [B, C, T, H, W]
    assert isinstance(output.output, torch.Tensor)
    assert output.output.ndim == 5
    assert output.output.shape[0] == 8
    assert output.output.shape[1] == 3
    assert output.output.shape[2] == 2  # num_frames

    # rollout_trajectory_data
    rt = output.rollout_trajectory_data
    assert rt is not None
    # log_probs: [B, T] = [8, 3]
    assert rt.rollout_log_probs.shape == (8, 3)
    # dit_trajectory.latents: [B, T_steps, C, T, H, W]
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
    assert env.extra["context"]["guidance_scale"] == 7.0
    assert env.extra["context"]["fps"] == 16

    # encode_prompt called once per prompt (not once per chunk).
    assert policy.encode_calls == ["a red cube", "a blue sphere"]


def test_executor_micro_batches_when_group_exceeds_sample_batch_size() -> None:
    """group_size=12 with sample_batch_size=4 → 3 chunks."""
    policy = _StubCosmosPolicy()
    executor = CosmosPipelineExecutor(policy, sample_batch_size=4)
    request = _request(
        prompts=["a red cube"],
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


def test_executor_workload_signature_matches_request() -> None:
    """workload_signature derives from the request sampling dict."""
    policy = _StubCosmosPolicy()
    executor = CosmosPipelineExecutor(policy)
    request = _request(
        num_steps=4,
        height=128,
        width=64,
        num_frames=8,
    )
    sig = executor.workload_signature(request)
    assert sig.family == "cosmos"
    assert sig.task == "v2w"
    assert sig.height == 128
    assert sig.width == 64
    assert sig.num_steps == 4
    assert sig.num_frames == 8


def test_executor_seed_reproducibility() -> None:
    """Same seed + same prompts → bitwise-identical log_probs and latents."""
    policy_a = _StubCosmosPolicy()
    policy_b = _StubCosmosPolicy()
    executor_a = CosmosPipelineExecutor(policy_a)
    executor_b = CosmosPipelineExecutor(policy_b)
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


def test_executor_kl_disabled_zero_kl() -> None:
    """When sampling.return_kl=False, kl tensor is zero."""
    policy = _StubCosmosPolicy()
    executor = CosmosPipelineExecutor(policy)
    request = _request(num_steps=3, height=32, width=32, samples_per_prompt=2)
    request.sampling["return_kl"] = False
    specs = GenerationIdFactory().build_sample_specs(request)

    output = executor.forward(request, specs)
    kl = output.rollout_trajectory_data.denoising_env.extra["kl"]
    assert torch.all(kl == 0)


def test_executor_kl_enabled_matches_log_prob_abs() -> None:
    """When sampling.return_kl=True, kl tensor matches |log_prob|."""
    policy = _StubCosmosPolicy()
    executor = CosmosPipelineExecutor(policy)
    request = _request(num_steps=3, height=32, width=32, samples_per_prompt=2)
    request.sampling["return_kl"] = True
    specs = GenerationIdFactory().build_sample_specs(request)

    output = executor.forward(request, specs)
    kl = output.rollout_trajectory_data.denoising_env.extra["kl"]
    log_probs = output.rollout_trajectory_data.rollout_log_probs
    assert torch.equal(kl, log_probs.abs())


def test_executor_uses_constructor_reference_image() -> None:
    """``reference_image`` from __init__ is forwarded to the policy."""
    sentinel = object()
    policy = _StubCosmosPolicy()
    executor = CosmosPipelineExecutor(policy, reference_image=sentinel)
    request = _request(num_steps=2, height=32, width=32, samples_per_prompt=1)
    specs = GenerationIdFactory().build_sample_specs(request)
    executor.forward(request, specs)
    # Both the encode and the prepare paths see the constructor-provided
    # reference image.
    assert policy.encode_reference_images == [sentinel]
    assert policy.prepare_reference_images == [sentinel]


def test_executor_metadata_reference_image_overrides_constructor() -> None:
    """Per-request metadata['reference_image'] takes precedence."""
    ctor_sentinel = object()
    meta_sentinel = object()
    policy = _StubCosmosPolicy()
    executor = CosmosPipelineExecutor(policy, reference_image=ctor_sentinel)
    request = _request(
        num_steps=2,
        height=32,
        width=32,
        samples_per_prompt=1,
        metadata={"reference_image": meta_sentinel},
    )
    specs = GenerationIdFactory().build_sample_specs(request)
    executor.forward(request, specs)
    assert policy.encode_reference_images == [meta_sentinel]
    assert policy.prepare_reference_images == [meta_sentinel]
