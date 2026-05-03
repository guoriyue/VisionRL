"""Unit tests for SD3_5PipelineExecutor.

These tests use a stubbed SD3_5Policy that implements the minimal
``encode_prompt``/``prepare_sampling``/``forward_step``/``decode_latents``
contract required by the executor — no diffusers / SD3.5 weights needed.
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
from vrl.models.families.sd3_5.executor import SD3_5PipelineExecutor

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubScheduler:
    """Mock scheduler keyed by step index (not by sigma value).

    ``timesteps[i]`` is encoded as ``i / num_steps`` so
    ``index_for_timestep(t)`` recovers ``i`` exactly. Sigmas have length
    ``num_steps + 2`` so ``prev_step_index = step + 1`` always stays in
    bounds even at the last step.
    """

    def __init__(self, num_steps: int) -> None:
        self._num_steps = num_steps
        # Sigmas length num_steps + 2, monotone decreasing 1 → 0.
        self.sigmas = torch.linspace(1.0, 0.0, num_steps + 2)
        # Timesteps[i] = (i + 1) / (num_steps + 2) so each timestep
        # uniquely maps to a sigma index.
        self.timesteps = torch.tensor(
            [self.sigmas[i].item() for i in range(num_steps)],
            dtype=torch.float32,
        )

    def index_for_timestep(self, t: torch.Tensor) -> int:
        diffs = (self.sigmas - t).abs()
        idx = int(diffs.argmin().item())
        # Clamp to never produce a prev_step_index out of bounds.
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
class _StubSamplingState:
    latents: torch.Tensor
    timesteps: torch.Tensor
    scheduler: _StubScheduler
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    negative_pooled_prompt_embeds: torch.Tensor | None
    guidance_scale: float
    do_cfg: bool
    seed: int


@dataclass
class _StubPolicy:
    """Bare-minimum SD3_5Policy stub.

    - ``encode_prompt`` returns deterministic fake embeds keyed off the
      prompt string so different prompts produce different states.
    - ``prepare_sampling`` builds latents from the per-chunk seed for
      reproducibility.
    - ``forward_step`` returns a deterministic linear function of the
      latents → ``sde_step_with_logprob`` math runs end-to-end.
    - ``decode_latents`` projects latents → 3-channel image.
    """

    embed_dim: int = 8
    pooled_dim: int = 4
    latent_channels: int = 4
    family: str = "sd3_5-stub"
    forward_calls: int = 0
    encode_calls: list[str] = field(default_factory=list)

    # SD3 returns these in a dict
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
        # Hash prompt → seed for deterministic-but-prompt-specific embeds.
        h = abs(hash(prompt)) % (2**31)
        gen = torch.Generator().manual_seed(h)
        prompt_embeds = torch.randn(
            1, 4, self.embed_dim, generator=gen,
        )
        pooled = torch.randn(1, self.pooled_dim, generator=gen)
        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled,
            "negative_prompt_embeds": torch.zeros_like(prompt_embeds),
            "negative_pooled_prompt_embeds": torch.zeros_like(pooled),
        }

    def prepare_sampling(
        self,
        request: Any,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> _StubSamplingState:
        del kwargs
        scheduler = _StubScheduler(request.num_steps)
        bsz = encoded["prompt_embeds"].shape[0]
        # Latent shape: [B, C, H/8, W/8] — small for tests.
        H = max(1, request.height // 32)
        W = max(1, request.width // 32)
        seed = request.seed if request.seed is not None else 0
        gen = torch.Generator().manual_seed(int(seed))
        latents = torch.randn(
            bsz, self.latent_channels, H, W, generator=gen,
        )
        return _StubSamplingState(
            latents=latents,
            timesteps=scheduler.timesteps,
            scheduler=scheduler,
            prompt_embeds=encoded["prompt_embeds"],
            pooled_prompt_embeds=encoded["pooled_prompt_embeds"],
            negative_prompt_embeds=encoded.get("negative_prompt_embeds"),
            negative_pooled_prompt_embeds=encoded.get("negative_pooled_prompt_embeds"),
            guidance_scale=request.guidance_scale,
            do_cfg=request.guidance_scale > 1.0,
            seed=int(seed),
        )

    def forward_step(
        self,
        state: _StubSamplingState,
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
        # Tile to 3 channels and upscale 8x for shape [B, 3, H*8, W*8].
        _B, C, _H, _W = latents.shape
        rgb = latents[:, :3] if C >= 3 else latents.repeat(1, 3, 1, 1)[:, :3]
        return torch.nn.functional.interpolate(
            rgb, scale_factor=8, mode="nearest",
        )

    def export_batch_context(self, state: _StubSamplingState) -> dict[str, Any]:
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
            "model_family": self.family,
        }

    def export_training_extras(self, state: _StubSamplingState) -> dict[str, Any]:
        return {
            "prompt_embeds": state.prompt_embeds,
            "pooled_prompt_embeds": state.pooled_prompt_embeds,
            "negative_prompt_embeds": state.negative_prompt_embeds,
            "negative_pooled_prompt_embeds": state.negative_pooled_prompt_embeds,
        }


def _request(
    *,
    prompts: list[str] | None = None,
    samples_per_prompt: int = 4,
    num_steps: int = 4,
    height: int = 64,
    width: int = 64,
    seed: int | None = 42,
    sample_batch_size: int = 8,
) -> GenerationRequest:
    return GenerationRequest(
        request_id="sd3_5-test",
        family="sd3_5",
        task="t2i",
        prompts=prompts or ["a red cube"],
        samples_per_prompt=samples_per_prompt,
        sampling={
            "num_steps": num_steps,
            "guidance_scale": 4.5,
            "height": height,
            "width": width,
            "noise_level": 0.7,
            "cfg": True,
            "sample_batch_size": sample_batch_size,
            "sde_window_size": 0,
            "sde_window_range": [0, num_steps],
            "same_latent": False,
            "max_sequence_length": 256,
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
    policy = _StubPolicy()
    executor = SD3_5PipelineExecutor(policy, sample_batch_size=8)
    request = _request(
        prompts=["a red cube", "a blue sphere"],
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
    assert output.family == "sd3_5"
    assert output.task == "t2i"
    assert len(output.sample_specs) == 8

    # output: decoded images [B, 3, H, W]
    assert isinstance(output.output, torch.Tensor)
    assert output.output.shape[0] == 8
    assert output.output.shape[1] == 3

    # rollout_trajectory_data
    rt = output.rollout_trajectory_data
    assert rt is not None
    # log_probs: [B, T] = [8, 3]
    assert rt.rollout_log_probs.shape == (8, 3)
    # dit_trajectory.latents: [B, T, C, H, W]
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
    assert policy.encode_calls == ["a red cube", "a blue sphere"]


def test_same_latent_requires_explicit_seed() -> None:
    policy = _StubPolicy()
    executor = SD3_5PipelineExecutor(policy, sample_batch_size=8)
    request = _request(seed=None)
    request.sampling["same_latent"] = True
    specs = GenerationIdFactory().build_sample_specs(request)

    with pytest.raises(ValueError, match="same_latent=True requires"):
        executor.forward(request, specs)


def test_executor_micro_batches_when_group_exceeds_sample_batch_size() -> None:
    """group_size=12 with sample_batch_size=4 → 3 chunks."""
    policy = _StubPolicy()
    executor = SD3_5PipelineExecutor(policy, sample_batch_size=4)
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


def test_executor_request_id_round_trip() -> None:
    """request_id flows through unchanged; failure to match is a hard error."""
    policy = _StubPolicy()
    executor = SD3_5PipelineExecutor(policy)
    request = _request(num_steps=2, height=32, width=32)
    specs = GenerationIdFactory().build_sample_specs(request)
    output = executor.forward(request, specs)
    assert output.request_id == "sd3_5-test"


def test_executor_workload_signature_matches_request() -> None:
    """workload_signature derives from the request sampling dict."""
    policy = _StubPolicy()
    executor = SD3_5PipelineExecutor(policy)
    request = _request(num_steps=4, height=128, width=64)
    sig = executor.workload_signature(request)
    assert sig.family == "sd3_5"
    assert sig.task == "t2i"
    assert sig.height == 128
    assert sig.width == 64
    assert sig.num_steps == 4


def test_executor_seed_reproducibility() -> None:
    """Same seed + same prompts → bitwise-identical log_probs and latents."""
    policy_a = _StubPolicy()
    policy_b = _StubPolicy()
    executor_a = SD3_5PipelineExecutor(policy_a)
    executor_b = SD3_5PipelineExecutor(policy_b)
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
    policy = _StubPolicy()
    executor = SD3_5PipelineExecutor(policy)
    request = _request(num_steps=3, height=32, width=32, samples_per_prompt=2)
    request.sampling["return_kl"] = False
    specs = GenerationIdFactory().build_sample_specs(request)

    output = executor.forward(request, specs)
    kl = output.rollout_trajectory_data.denoising_env.extra["kl"]
    assert torch.all(kl == 0)


def test_executor_kl_window_populated_when_enabled() -> None:
    """When sampling.return_kl=True, kl tensor matches |log_prob|."""
    policy = _StubPolicy()
    executor = SD3_5PipelineExecutor(policy)
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
    policy = _StubPolicy()
    executor = SD3_5PipelineExecutor(policy, sample_batch_size=4)
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
