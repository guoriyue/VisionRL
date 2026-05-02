"""Unit tests for NextStep1PipelineExecutor.

These tests use a stubbed NextStep1Policy that implements the minimal
``processor``/``language_model``/``sample_image_tokens``/``decode_image_tokens``
contract required by the executor — no upstream NextStep-1 weights needed.

NextStep-1 differs from a diffusion executor in three ways the stubs reflect:

- AR image sampling is one black-box call (``sample_image_tokens``) that
  returns three tensors: ``tokens``/``saved_noise``/``log_probs``.
- Tokens are continuous embeddings ``[B, L_img, D_token]``, not int ids.
- There is no DiT trajectory; the executor sets
  ``rollout_trajectory_data.dit_trajectory = None`` and
  ``rollout_trajectory_data.denoising_env = None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
import torch
import torch.nn as nn

from vrl.engine.generation import (
    GenerationIdFactory,
    GenerationRequest,
)
from vrl.models.families.nextstep_1.executor import NextStep1PipelineExecutor

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubTokenizer:
    """Bare-minimum HF-style tokenizer.

    Returns deterministic int ids derived from prompt characters, padded
    to ``max_length``. Every prompt produces a different ids tensor so we
    can assert routing + alignment.
    """

    pad_token_id: int = 0

    def __call__(
        self,
        prompts: list[str],
        *,
        return_tensors: str = "pt",
        padding: str = "max_length",
        truncation: bool = True,
        max_length: int = 16,
    ) -> dict[str, torch.Tensor]:
        del return_tensors, padding, truncation
        B = len(prompts)
        ids = torch.zeros(B, max_length, dtype=torch.long)
        mask = torch.zeros(B, max_length, dtype=torch.long)
        for i, p in enumerate(prompts):
            if len(p) == 0:
                # Empty prompt (uncond): emit a single non-pad id at pos 0
                # so attention mask is non-trivial but tokens differ from
                # the conditional prompt.
                ids[i, 0] = 1
                mask[i, 0] = 1
                continue
            n = min(max_length, len(p))
            for j in range(n):
                # Map char to a non-pad id (1..250).
                ids[i, j] = (ord(p[j]) % 250) + 1
            mask[i, :n] = 1
        return {"input_ids": ids, "attention_mask": mask}


@dataclass
class _StubLanguageModel:
    """Stub LLM exposing ``get_input_embeddings``.

    The executor only needs ``self.model.language_model.get_input_embeddings()``
    so the LLM can stay a simple ``nn.Embedding`` wrapper.
    """

    vocab_size: int = 256
    hidden_size: int = 8

    def __post_init__(self) -> None:
        gen = torch.Generator().manual_seed(0)
        weight = torch.randn(self.vocab_size, self.hidden_size, generator=gen)
        self._embed = nn.Embedding.from_pretrained(weight, freeze=True)

    def get_input_embeddings(self) -> nn.Module:
        return self._embed


@dataclass
class _StubPolicy:
    """Bare-minimum NextStep1Policy stub.

    - ``processor`` is the stub HF tokenizer.
    - ``language_model.get_input_embeddings()`` returns a deterministic
      embedding so we can assert prompt routing.
    - ``sample_image_tokens`` returns the canonical 3-tuple
      ``(tokens, saved_noise, log_probs)``. Outputs are derived from the
      generator state so seed parity is exact.
    - ``decode_image_tokens`` projects tokens → 3-channel image.
    """

    image_token_num: int = 16
    token_dim: int = 4
    hidden_dim: int = 8
    image_size_default: int = 64
    sample_calls: int = 0
    decode_calls: int = 0
    last_sample_kwargs: dict[str, Any] = field(default_factory=dict)
    family: str = "nextstep_1-stub"
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def __post_init__(self) -> None:
        self.processor = _StubTokenizer()
        self.language_model = _StubLanguageModel(hidden_size=self.hidden_dim)

    @torch.no_grad()
    def sample_image_tokens(
        self,
        prompt_embeds: torch.Tensor,
        uncond_embeds: torch.Tensor | None,
        prompt_mask: torch.Tensor,
        uncond_mask: torch.Tensor | None,
        *,
        cfg_scale: float | None = None,
        num_flow_steps: int | None = None,
        noise_level: float | None = None,
        image_token_num: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del uncond_embeds, prompt_mask, uncond_mask
        del cfg_scale, num_flow_steps, noise_level
        self.sample_calls += 1
        self.last_sample_kwargs = {
            "image_token_num": image_token_num,
            "generator_is_set": generator is not None,
        }
        B = prompt_embeds.shape[0]
        L = int(image_token_num or self.image_token_num)
        D = self.token_dim

        # Derive deterministic outputs from generator if available; fall
        # back to a fresh generator otherwise so unit tests without a seed
        # also work.
        gen = generator if generator is not None else torch.Generator().manual_seed(0)
        tokens = torch.randn(B, L, D, generator=gen)
        saved_noise = torch.randn(B, L, D, generator=gen)
        # log_probs negative-real, derived from tokens so non-trivial.
        log_probs = -(tokens.float() ** 2).mean(dim=-1)  # [B, L]
        return tokens, saved_noise, log_probs

    @torch.no_grad()
    def decode_image_tokens(
        self,
        tokens: torch.Tensor,
        image_size: int | None = None,
    ) -> torch.Tensor:
        self.decode_calls += 1
        H = W = int(image_size or self.image_size_default)
        B = tokens.shape[0]
        # Sum tokens to a scalar per sample, broadcast to a 3-channel image
        # so different inputs produce different decoded images.
        scalar = tokens.float().mean(dim=(1, 2)).view(B, 1, 1, 1)
        base = torch.linspace(-1.0, 1.0, H * W).view(1, 1, H, W)
        rgb = (scalar * base).expand(B, 3, H, W).contiguous()
        return rgb.clamp(-1.0, 1.0)


def _request(
    *,
    prompts: list[str] | None = None,
    samples_per_prompt: int = 4,
    image_token_num: int = 16,
    image_size: int = 64,
    seed: int | None = 42,
) -> GenerationRequest:
    return GenerationRequest(
        request_id="nextstep_1-test",
        family="nextstep_1",
        task="ar_t2i",
        prompts=prompts or ["a red cube"],
        samples_per_prompt=samples_per_prompt,
        sampling={
            "cfg_scale": 4.5,
            "num_flow_steps": 4,
            "noise_level": 1.0,
            "image_token_num": image_token_num,
            "image_size": image_size,
            "max_text_length": 16,
            "rescale_to_unit": True,
            "seed": seed,
        },
        return_artifacts={"output", "rollout_trajectory_data"},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_executor_forward_shapes_for_two_prompts_x_four_samples() -> None:
    """forward(2 prompts x 4 samples) -> OutputBatch with 8 specs + correct shapes."""
    policy = _StubPolicy()
    executor = NextStep1PipelineExecutor(policy)
    request = _request(
        prompts=["a red cube", "a blue sphere"],
        samples_per_prompt=4,
        image_token_num=16,
        image_size=32,
        seed=42,
    )
    specs = GenerationIdFactory().build_sample_specs(request)
    assert len(specs) == 8

    output = executor.forward(request, specs)

    assert output.error is None
    assert output.request_id == request.request_id
    assert output.family == "nextstep_1"
    assert output.task == "ar_t2i"
    assert len(output.sample_specs) == 8

    # output: decoded images [B, 3, H, W]
    assert isinstance(output.output, torch.Tensor)
    assert output.output.shape == (8, 3, 32, 32)

    # rollout_trajectory_data: AR has no dit_trajectory / denoising_env
    rt = output.rollout_trajectory_data
    assert rt is not None
    assert rt.dit_trajectory is None
    assert rt.denoising_env is None
    # log_probs: [B, L_img] = [8, 16]
    assert rt.rollout_log_probs.shape == (8, 16)

    # extra payload: AR-specific replay artifacts
    extra = output.extra
    assert extra["tokens"].shape == (8, 16, policy.token_dim)
    assert extra["saved_noise"].shape == (8, 16, policy.token_dim)
    assert extra["log_probs"].shape == (8, 16)
    assert extra["prompt_input_ids"].shape == (8, 16)
    assert extra["prompt_attention_mask"].shape == (8, 16)
    assert extra["uncond_input_ids"].shape == (8, 16)
    assert extra["uncond_attention_mask"].shape == (8, 16)
    assert extra["images_for_reward"].shape == output.output.shape

    # context carries replay-time scalars
    assert extra["context"]["cfg_scale"] == 4.5
    assert extra["context"]["num_flow_steps"] == 4
    assert extra["context"]["image_token_num"] == 16


def test_executor_request_id_round_trip() -> None:
    """request_id flows through unchanged."""
    policy = _StubPolicy()
    executor = NextStep1PipelineExecutor(policy)
    request = _request()
    specs = GenerationIdFactory().build_sample_specs(request)
    output = executor.forward(request, specs)
    assert output.request_id == "nextstep_1-test"


def test_executor_workload_signature_matches_request() -> None:
    """workload_signature derives from the request sampling dict."""
    policy = _StubPolicy()
    executor = NextStep1PipelineExecutor(policy)
    request = _request(image_token_num=64, image_size=128)
    sig = executor.workload_signature(request)
    assert sig.family == "nextstep_1"
    assert sig.task == "ar_t2i"


def test_executor_seed_reproducibility() -> None:
    """Same seed + same prompts → bitwise-identical tokens, saved_noise, log_probs."""
    policy_a = _StubPolicy()
    policy_b = _StubPolicy()
    executor_a = NextStep1PipelineExecutor(policy_a)
    executor_b = NextStep1PipelineExecutor(policy_b)

    req_a = _request(seed=1234)
    req_b = _request(seed=1234)
    specs_a = GenerationIdFactory().build_sample_specs(req_a)
    specs_b = GenerationIdFactory().build_sample_specs(req_b)

    out_a = executor_a.forward(req_a, specs_a)
    out_b = executor_b.forward(req_b, specs_b)

    assert torch.equal(out_a.extra["tokens"], out_b.extra["tokens"])
    assert torch.equal(out_a.extra["saved_noise"], out_b.extra["saved_noise"])
    assert torch.equal(out_a.extra["log_probs"], out_b.extra["log_probs"])
    assert torch.equal(out_a.output, out_b.output)


def test_executor_different_seeds_produce_different_tokens() -> None:
    """Different seeds → different tokens (sanity check on stub determinism)."""
    policy = _StubPolicy()
    executor = NextStep1PipelineExecutor(policy)
    req_a = _request(seed=11)
    req_b = _request(seed=22)
    out_a = executor.forward(req_a, GenerationIdFactory().build_sample_specs(req_a))
    out_b = executor.forward(req_b, GenerationIdFactory().build_sample_specs(req_b))
    assert not torch.equal(out_a.extra["tokens"], out_b.extra["tokens"])


def test_executor_metrics_recorded() -> None:
    """OutputBatch.metrics carries num_prompts/num_samples/num_steps."""
    policy = _StubPolicy()
    executor = NextStep1PipelineExecutor(policy)
    request = _request(samples_per_prompt=3, image_token_num=8)
    specs = GenerationIdFactory().build_sample_specs(request)
    output = executor.forward(request, specs)
    assert output.metrics is not None
    assert output.metrics.num_prompts == 1
    assert output.metrics.num_samples == 3
    assert output.metrics.num_steps == 8
    assert output.metrics.micro_batches == 1


@pytest.mark.parametrize("samples_per_prompt", [1, 3, 5])
def test_executor_arbitrary_group_sizes(samples_per_prompt: int) -> None:
    """Various group sizes produce correct B dimension."""
    policy = _StubPolicy()
    executor = NextStep1PipelineExecutor(policy)
    request = _request(samples_per_prompt=samples_per_prompt)
    specs = GenerationIdFactory().build_sample_specs(request)
    output = executor.forward(request, specs)
    assert output.output.shape[0] == samples_per_prompt
    assert output.extra["tokens"].shape[0] == samples_per_prompt
    assert output.extra["saved_noise"].shape[0] == samples_per_prompt
    assert output.extra["log_probs"].shape[0] == samples_per_prompt


def test_executor_rescale_to_unit_clamps_decoded_images() -> None:
    """When rescale_to_unit=True, images_for_reward lives in [0, 1]."""
    policy = _StubPolicy()
    executor = NextStep1PipelineExecutor(policy)
    request = _request()
    request.sampling["rescale_to_unit"] = True
    specs = GenerationIdFactory().build_sample_specs(request)
    output = executor.forward(request, specs)
    imgs = output.extra["images_for_reward"]
    assert imgs.min().item() >= 0.0 - 1e-6
    assert imgs.max().item() <= 1.0 + 1e-6


def test_executor_rescale_to_unit_off_keeps_native_range() -> None:
    """When rescale_to_unit=False, images_for_reward IS the decoded image."""
    policy = _StubPolicy()
    executor = NextStep1PipelineExecutor(policy)
    request = _request()
    request.sampling["rescale_to_unit"] = False
    specs = GenerationIdFactory().build_sample_specs(request)
    output = executor.forward(request, specs)
    assert torch.equal(output.extra["images_for_reward"], output.output)


def test_executor_sample_call_count_is_one() -> None:
    """AR sample_image_tokens is a single black-box call per request."""
    policy = _StubPolicy()
    executor = NextStep1PipelineExecutor(policy)
    request = _request(samples_per_prompt=4)
    specs = GenerationIdFactory().build_sample_specs(request)
    executor.forward(request, specs)
    assert policy.sample_calls == 1
    assert policy.decode_calls == 1
