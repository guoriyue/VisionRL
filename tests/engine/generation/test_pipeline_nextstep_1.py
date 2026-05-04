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

import torch
import torch.nn as nn

from vrl.engine import (
    GenerationIdFactory,
    GenerationRequest,
)
from vrl.models.ar import ARStepResult
from vrl.models.families.nextstep_1.executor import NextStep1PipelineExecutor
from vrl.models.families.nextstep_1.flow_step import (
    flow_logprob_at,
    flow_sample_with_logprob,
)

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


class _ZeroVelocityHead:
    input_dim: int = 4

    def net(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        del t, cond
        return torch.zeros_like(x)


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
    ar_init_calls: int = 0
    ar_step_calls: int = 0
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
        self.sample_calls += 1
        self.last_sample_kwargs = {
            "image_token_num": image_token_num,
            "generator_is_set": generator is not None,
        }
        state = self.init_ar_state(
            prompt_embeds,
            uncond_embeds,
            prompt_mask,
            uncond_mask,
            cfg_scale=cfg_scale,
            num_flow_steps=num_flow_steps,
            noise_level=noise_level,
            image_token_num=image_token_num,
            generator=generator,
        )
        while state["position"] < state["image_token_num"]:
            self._sample_ar_step(state)
        return self.finalize_ar_state(state)

    def init_ar_state(
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
    ) -> dict[str, Any]:
        del uncond_embeds, prompt_mask, uncond_mask
        del cfg_scale, num_flow_steps, noise_level
        self.ar_init_calls += 1
        gen = generator if generator is not None else torch.Generator().manual_seed(0)
        batch_size = prompt_embeds.shape[0]
        image_token_num = int(image_token_num or self.image_token_num)
        return {
            "tokens": torch.zeros(batch_size, image_token_num, self.token_dim),
            "saved_noise": torch.zeros(batch_size, image_token_num, self.token_dim),
            "log_probs": torch.zeros(batch_size, image_token_num),
            "image_token_num": image_token_num,
            "generator": gen,
            "position": 0,
            "positions": torch.zeros(batch_size, dtype=torch.long),
        }

    def step_ar(
        self,
        state: dict[str, Any],
        sequences: list[Any],
        *,
        generator: torch.Generator | None = None,
    ) -> ARStepResult:
        del generator
        self.ar_step_calls += 1
        row_indices = [sequence.metadata["row_index"] for sequence in sequences]
        positions = [sequence.position for sequence in sequences]
        assert len(set(positions)) == 1
        assert positions == [int(state["positions"][row].item()) for row in row_indices]
        token, saved_noise, log_prob = self._sample_ar_step(
            state,
            row_indices=row_indices,
            position=positions[0],
        )
        return ARStepResult(
            sequence_ids=[sequence.sample_id for sequence in sequences],
            positions=positions,
            token=token,
            log_prob=log_prob,
            replay_extras={"saved_noise": saved_noise},
        )

    def finalize_ar_state(
        self,
        state: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return state["tokens"], state["saved_noise"], state["log_probs"]

    def _sample_ar_step(
        self,
        state: dict[str, Any],
        *,
        row_indices: list[int] | None = None,
        position: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if row_indices is None:
            row_indices = list(range(state["tokens"].shape[0]))
        rows = torch.tensor(row_indices, dtype=torch.long)
        row_positions = [int(state["positions"][row].item()) for row in row_indices]
        assert len(set(row_positions)) == 1
        position = row_positions[0] if position is None else position
        assert all(row_position == position for row_position in row_positions)

        batch_size = len(row_indices)
        gen = state["generator"]
        token = torch.randn(batch_size, self.token_dim, generator=gen)
        saved_noise = torch.randn(batch_size, self.token_dim, generator=gen)
        log_prob = -(token.float() ** 2).mean(dim=-1)
        state["tokens"][rows, position] = token
        state["saved_noise"][rows, position] = saved_noise
        state["log_probs"][rows, position] = log_prob
        state["positions"][rows] += 1
        state["position"] = int(state["positions"].min().item())
        return token, saved_noise, log_prob

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


def test_executor_scheduled_ar_matches_black_box_path_bitwise() -> None:
    """Scheduled full-row AR path preserves the black-box output contract."""
    policy_black_box = _StubPolicy()
    executor_black_box = NextStep1PipelineExecutor(policy_black_box)
    request_black_box = _request(seed=1234)
    specs_black_box = GenerationIdFactory().build_sample_specs(request_black_box)
    out_black_box = executor_black_box.forward(request_black_box, specs_black_box)

    policy_scheduled = _StubPolicy()
    executor_scheduled = NextStep1PipelineExecutor(policy_scheduled)
    request_scheduled = _request(seed=1234)
    request_scheduled.sampling["use_ar_scheduler"] = True
    specs_scheduled = GenerationIdFactory().build_sample_specs(request_scheduled)
    out_scheduled = executor_scheduled.forward(request_scheduled, specs_scheduled)

    assert policy_scheduled.sample_calls == 0
    assert policy_scheduled.ar_init_calls == 1
    assert policy_scheduled.ar_step_calls == request_scheduled.sampling["image_token_num"]
    assert torch.equal(out_black_box.extra["tokens"], out_scheduled.extra["tokens"])
    assert torch.equal(out_black_box.extra["saved_noise"], out_scheduled.extra["saved_noise"])
    assert torch.equal(out_black_box.extra["log_probs"], out_scheduled.extra["log_probs"])
    assert torch.equal(out_black_box.output, out_scheduled.output)


def test_executor_partial_scheduled_ar_is_deterministic_and_complete() -> None:
    """Partial scheduled AR is reproducible and keeps replay artifacts aligned."""
    policy_a = _StubPolicy()
    executor_a = NextStep1PipelineExecutor(policy_a)
    request_a = _request(seed=1234, samples_per_prompt=5)
    request_a.sampling["use_ar_scheduler"] = True
    request_a.sampling["ar_scheduler_batch_size"] = 2
    specs_a = GenerationIdFactory().build_sample_specs(request_a)
    out_a = executor_a.forward(request_a, specs_a)

    policy_b = _StubPolicy()
    executor_b = NextStep1PipelineExecutor(policy_b)
    request_b = _request(seed=1234, samples_per_prompt=5)
    request_b.sampling["use_ar_scheduler"] = True
    request_b.sampling["ar_scheduler_batch_size"] = 2
    specs_b = GenerationIdFactory().build_sample_specs(request_b)
    out_b = executor_b.forward(request_b, specs_b)

    assert policy_a.sample_calls == 0
    assert policy_a.ar_init_calls == 1
    assert policy_a.ar_step_calls > request_a.sampling["image_token_num"]
    assert [spec.sample_id for spec in out_a.sample_specs] == [spec.sample_id for spec in specs_a]
    assert out_a.extra["tokens"].shape == (5, request_a.sampling["image_token_num"], 4)
    assert out_a.extra["saved_noise"].shape == (
        5,
        request_a.sampling["image_token_num"],
        4,
    )
    assert out_a.extra["log_probs"].shape == (5, request_a.sampling["image_token_num"])
    assert torch.equal(out_a.extra["tokens"], out_b.extra["tokens"])
    assert torch.equal(
        out_a.extra["saved_noise"],
        out_b.extra["saved_noise"],
    )
    assert torch.equal(
        out_a.extra["log_probs"],
        out_b.extra["log_probs"],
    )
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


def test_flow_sample_saved_noise_replays_sample_logprob() -> None:
    """Explicit initial_noise is the replay artifact used by flow_logprob_at."""
    head = _ZeroVelocityHead()
    cond = torch.zeros(2, 8)
    initial_noise = torch.full((2, head.input_dim), 0.25)
    generator = torch.Generator().manual_seed(7)

    sample = flow_sample_with_logprob(
        head,
        cond,
        num_flow_steps=4,
        noise_level=1.0,
        generator=generator,
        initial_noise=initial_noise,
    )
    replay_log_prob = flow_logprob_at(
        head,
        cond,
        target_token=sample.token,
        saved_noise=sample.initial_noise,
        num_flow_steps=4,
        noise_level=1.0,
    )

    assert torch.equal(sample.initial_noise, initial_noise)
    assert torch.allclose(sample.log_prob, replay_log_prob)
