"""Unit tests for JanusProPipelineExecutor.

These tests use a stubbed JanusProPolicy that implements the minimal
``processor``/``language_model``/``sample_image_tokens``/
``decode_image_tokens`` contract required by the executor — no
DeepSeek-Janus weights are loaded.
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
from vrl.models.families.janus_pro.executor import JanusProPipelineExecutor

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


HIDDEN = 16
TEXT_VOCAB = 64
IMG_VOCAB = 1024


class _StubTokenizer:
    """Tokeniser that maps each character to ``ord(c) % TEXT_VOCAB``."""

    pad_token_id: int = 0

    def __call__(
        self,
        formatted: list[str],
        return_tensors: str = "pt",
        padding: str | bool = True,
        truncation: bool = True,
        max_length: int = 256,
    ) -> dict[str, torch.Tensor]:
        del return_tensors, truncation
        seqs = [
            torch.tensor(
                [ord(c) % TEXT_VOCAB for c in s[:max_length]], dtype=torch.long,
            )
            for s in formatted
        ]
        L = max_length if padding == "max_length" else max(s.numel() for s in seqs)
        ids = torch.zeros(len(seqs), L, dtype=torch.long)
        mask = torch.zeros(len(seqs), L, dtype=torch.long)
        for i, s in enumerate(seqs):
            n = min(s.numel(), L)
            ids[i, :n] = s[:n]
            mask[i, :n] = 1
        return {"input_ids": ids, "attention_mask": mask}


class _StubProcessor:
    def __init__(self) -> None:
        self.tokenizer = _StubTokenizer()


@dataclass
class _StubPolicy:
    """Bare-minimum JanusProPolicy stub.

    - ``sample_image_tokens`` returns deterministic-but-torch-RNG-aware
      tokens + log-probs so ``torch.manual_seed(seed)`` makes parity
      tests reproducible.
    - ``decode_image_tokens`` projects the tokens to a 3-channel image
      tensor in ``[-1, 1]``.
    - ``language_model.get_input_embeddings()`` is a small ``nn.Embedding``
      so the executor's ``_embed`` step type-checks.
    """

    image_token_num: int = 4
    sample_calls: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._processor = _StubProcessor()
        self._embed = nn.Embedding(TEXT_VOCAB, HIDDEN)
        self._lm = type("_LM", (), {
            "get_input_embeddings": lambda _self=self: self._embed,
        })()

    @property
    def processor(self) -> _StubProcessor:
        return self._processor

    @property
    def language_model(self) -> Any:
        return self._lm

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def model_family(self) -> str:
        return "janus_pro-stub"

    def sample_image_tokens(
        self,
        cond_inputs_embeds: torch.Tensor,
        uncond_inputs_embeds: torch.Tensor,
        cond_attention_mask: torch.Tensor,
        uncond_attention_mask: torch.Tensor,
        *,
        cfg_weight: float | None = None,
        temperature: float | None = None,
        image_token_num: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        L_img = image_token_num or self.image_token_num
        B = cond_inputs_embeds.shape[0]
        # Record the call signature for shape sanity assertions.
        self.sample_calls.append(
            {
                "B": B,
                "L_img": L_img,
                "cfg_weight": cfg_weight,
                "temperature": temperature,
                "L_text": cond_inputs_embeds.shape[1],
            },
        )
        # Use the global RNG so torch.manual_seed in the executor governs
        # reproducibility — exactly what the parity test relies on.
        token_ids = torch.randint(0, IMG_VOCAB, (B, L_img), dtype=torch.long)
        log_probs = -torch.rand(B, L_img, dtype=torch.float32)
        return token_ids, log_probs

    def decode_image_tokens(
        self,
        image_token_ids: torch.Tensor,
        *,
        image_size: int = 384,
    ) -> torch.Tensor:
        B = image_token_ids.shape[0]
        # Tile a content-derived signal to [-1, 1].
        sig = image_token_ids.float().mean(dim=-1, keepdim=True).view(B, 1, 1, 1)
        # Small image tensor — keep tests fast.
        H = W = max(8, image_size // 8)
        return torch.zeros(B, 3, H, W) + sig.tanh().expand(-1, 3, H, W) * 0.5


def _request(
    *,
    prompts: list[str] | None = None,
    samples_per_prompt: int = 4,
    image_token_num: int = 4,
    image_size: int = 64,
    max_text_length: int = 8,
    seed: int | None = 42,
    cfg_weight: float = 5.0,
    temperature: float = 1.0,
) -> GenerationRequest:
    sampling: dict[str, Any] = {
        "cfg_weight": cfg_weight,
        "temperature": temperature,
        "image_token_num": image_token_num,
        "image_size": image_size,
        "max_text_length": max_text_length,
    }
    if seed is not None:
        sampling["seed"] = seed
    return GenerationRequest(
        request_id="janus_pro-test",
        family="janus_pro",
        task="ar_t2i",
        prompts=prompts or ["a red cube"],
        samples_per_prompt=samples_per_prompt,
        sampling=sampling,
        return_artifacts={"output", "token_ids", "token_log_probs"},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_executor_forward_shapes_for_two_prompts_x_four_samples() -> None:
    """forward(2 prompts x 4 samples) -> OutputBatch with 8 specs + correct shapes."""
    policy = _StubPolicy(image_token_num=4)
    executor = JanusProPipelineExecutor(policy)
    request = _request(
        prompts=["a red cube", "a blue sphere"],
        samples_per_prompt=4,
        image_token_num=4,
        image_size=64,
        max_text_length=8,
        seed=42,
    )
    specs = GenerationIdFactory().build_sample_specs(request)
    assert len(specs) == 8

    output = executor.forward(request, specs)

    assert output.error is None
    assert output.request_id == request.request_id
    assert output.family == "janus_pro"
    assert output.task == "ar_t2i"
    assert len(output.sample_specs) == 8

    # output: decoded images [B, 3, H, W]
    assert isinstance(output.output, torch.Tensor)
    assert output.output.shape[0] == 8
    assert output.output.shape[1] == 3

    # extra payload
    assert output.extra["token_ids"].shape == (8, 4)
    assert output.extra["token_log_probs"].shape == (8, 4)
    assert output.extra["token_mask"].shape == (8, 4)
    assert output.extra["prompt_input_ids"].shape == (8, 8)  # max_text_length
    assert output.extra["prompt_attention_mask"].shape == (8, 8)
    assert output.extra["uncond_input_ids"].shape == (8, 8)
    assert output.extra["uncond_attention_mask"].shape == (8, 8)
    assert output.extra["context"]["cfg_weight"] == 5.0
    assert output.extra["context"]["image_token_num"] == 4

    # AR has no DiT trajectory.
    assert output.rollout_trajectory_data is None


def test_executor_request_id_round_trip() -> None:
    """request_id flows through unchanged."""
    policy = _StubPolicy(image_token_num=4)
    executor = JanusProPipelineExecutor(policy)
    request = _request(image_token_num=4, image_size=64, max_text_length=8)
    specs = GenerationIdFactory().build_sample_specs(request)
    output = executor.forward(request, specs)
    assert output.request_id == "janus_pro-test"


def test_executor_workload_signature_matches_request() -> None:
    """workload_signature derives from the request sampling dict."""
    policy = _StubPolicy(image_token_num=8)
    executor = JanusProPipelineExecutor(policy)
    # Surface the AR token count under the canonical sampling key so the
    # P1 batch planner can group same-shape AR requests.
    request = _request(image_token_num=8, image_size=128, max_text_length=8)
    request.sampling["max_new_image_tokens"] = 8
    sig = executor.workload_signature(request)
    assert sig.family == "janus_pro"
    assert sig.task == "ar_t2i"
    assert sig.max_new_tokens == 8


def test_executor_seed_reproducibility() -> None:
    """Same seed + same prompts → bitwise-identical token_ids and log_probs."""
    policy_a = _StubPolicy(image_token_num=4)
    policy_b = _StubPolicy(image_token_num=4)
    executor_a = JanusProPipelineExecutor(policy_a)
    executor_b = JanusProPipelineExecutor(policy_b)
    req_a = _request(image_token_num=4, image_size=64, max_text_length=8, seed=1234)
    req_b = _request(image_token_num=4, image_size=64, max_text_length=8, seed=1234)
    specs_a = GenerationIdFactory().build_sample_specs(req_a)
    specs_b = GenerationIdFactory().build_sample_specs(req_b)

    out_a = executor_a.forward(req_a, specs_a)
    out_b = executor_b.forward(req_b, specs_b)

    assert torch.equal(out_a.extra["token_ids"], out_b.extra["token_ids"])
    assert torch.equal(
        out_a.extra["token_log_probs"], out_b.extra["token_log_probs"],
    )
    assert torch.equal(out_a.output, out_b.output)


def test_executor_passes_cfg_and_temperature_to_model() -> None:
    """sample_image_tokens receives the cfg_weight + temperature from sampling."""
    policy = _StubPolicy(image_token_num=4)
    executor = JanusProPipelineExecutor(policy)
    request = _request(
        cfg_weight=3.5, temperature=0.7,
        image_token_num=4, image_size=64, max_text_length=8,
    )
    specs = GenerationIdFactory().build_sample_specs(request)
    executor.forward(request, specs)
    assert len(policy.sample_calls) == 1
    assert policy.sample_calls[0]["cfg_weight"] == pytest.approx(3.5)
    assert policy.sample_calls[0]["temperature"] == pytest.approx(0.7)


@pytest.mark.parametrize("samples_per_prompt", [1, 3, 5])
def test_executor_arbitrary_group_sizes(samples_per_prompt: int) -> None:
    """Various group sizes produce correct B dimension."""
    policy = _StubPolicy(image_token_num=4)
    executor = JanusProPipelineExecutor(policy)
    request = _request(
        samples_per_prompt=samples_per_prompt,
        image_token_num=4, image_size=64, max_text_length=8,
    )
    specs = GenerationIdFactory().build_sample_specs(request)
    output = executor.forward(request, specs)
    assert output.output.shape[0] == samples_per_prompt
    assert output.extra["token_ids"].shape == (samples_per_prompt, 4)
    assert output.extra["token_log_probs"].shape == (samples_per_prompt, 4)


def test_executor_metrics_populated() -> None:
    """GenerationMetrics carries num_prompts/num_samples/num_steps."""
    policy = _StubPolicy(image_token_num=4)
    executor = JanusProPipelineExecutor(policy)
    request = _request(
        prompts=["x", "y"], samples_per_prompt=3,
        image_token_num=4, image_size=64, max_text_length=8,
    )
    specs = GenerationIdFactory().build_sample_specs(request)
    output = executor.forward(request, specs)
    assert output.metrics is not None
    assert output.metrics.num_prompts == 2
    assert output.metrics.num_samples == 6
    assert output.metrics.num_steps == 4
    assert output.metrics.micro_batches == 1


def test_executor_token_mask_is_ones() -> None:
    """token_mask is all-ones (Janus has no per-position padding)."""
    policy = _StubPolicy(image_token_num=4)
    executor = JanusProPipelineExecutor(policy)
    request = _request(
        image_token_num=4, image_size=64, max_text_length=8,
        samples_per_prompt=2,
    )
    specs = GenerationIdFactory().build_sample_specs(request)
    output = executor.forward(request, specs)
    assert torch.all(output.extra["token_mask"] == 1)
