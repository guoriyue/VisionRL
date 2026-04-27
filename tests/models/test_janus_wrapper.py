"""Tests for vrl.models.families.janus_pro.JanusProT2I.

Uses a stub MultiModalityCausalLM (identity trunk + tiny gen_head) so
we can validate the shape contracts of the wrapper without downloading
the 3 GB Janus checkpoint.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vrl.models.families.janus_pro.model import (
    JANUS_IMAGE_PIXEL_SIZE,
    JANUS_IMAGE_TOKEN_NUM,
    JANUS_IMAGE_VOCAB_SIZE,
    JanusProConfig,
    JanusProT2I,
    image_token_logits_from_hidden,
)


HIDDEN = 32
TEXT_VOCAB = 64
IMG_VOCAB = JANUS_IMAGE_VOCAB_SIZE


class _StubLM(nn.Module):
    """Identity trunk: output last_hidden_state == input embeddings."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(TEXT_VOCAB, HIDDEN)

    @property
    def model(self) -> "_StubLM":
        # Property (not attribute) so nn.Module does not register self as a
        # submodule of itself, which would make ``train()`` recurse forever.
        return self

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(
        self,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: object = None,
        output_hidden_states: bool = False,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            last_hidden_state=inputs_embeds,
            past_key_values=past_key_values,
        )


class _StubVQ(nn.Module):
    def decode_code(self, ids: torch.Tensor, shape: list[int]) -> torch.Tensor:
        B, _, h, w = shape
        return torch.zeros(B, 3, h * 16, w * 16)


class _StubMMGPT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _StubLM()
        self.gen_vision_model = _StubVQ()
        self.gen_head = nn.Linear(HIDDEN, IMG_VOCAB)
        self.gen_aligner = nn.Identity()
        self.gen_embed = nn.Embedding(IMG_VOCAB, HIDDEN)

    def prepare_gen_img_embeds(self, ids: torch.Tensor) -> torch.Tensor:
        return self.gen_embed(ids)


@pytest.fixture
def stub_model() -> JanusProT2I:
    cfg = JanusProConfig(use_lora=False)
    return JanusProT2I(config=cfg, mmgpt=_StubMMGPT(), processor=object())


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


class TestSurface:
    def test_constants(self) -> None:
        assert JANUS_IMAGE_TOKEN_NUM == 576
        assert JANUS_IMAGE_VOCAB_SIZE == 16_384
        assert JANUS_IMAGE_PIXEL_SIZE == 384

    def test_loads_with_stub(self, stub_model: JanusProT2I) -> None:
        # Frozen base, no LoRA → 0 trainable
        assert stub_model.trainable_param_count() == 0

    def test_missing_attr_raises(self) -> None:
        broken = nn.Module()  # no language_model / gen_head / gen_vision_model
        with pytest.raises(RuntimeError, match="Janus"):
            JanusProT2I(JanusProConfig(use_lora=False), mmgpt=broken, processor=object())


# ---------------------------------------------------------------------------
# Train-time forward
# ---------------------------------------------------------------------------


class TestForwardImageLogits:
    def test_shape(self, stub_model: JanusProT2I) -> None:
        B, L_text, L_img = 2, 5, 8
        text_emb = torch.randn(B, L_text, HIDDEN)
        text_mask = torch.ones(B, L_text)
        img_ids = torch.randint(0, IMG_VOCAB, (B, L_img))
        out = stub_model.forward_image_logits(text_emb, text_mask, img_ids)
        assert out.shape == (B, L_img, IMG_VOCAB)

    def test_uses_gen_head_not_lm_head(self, stub_model: JanusProT2I) -> None:
        """Catches the silent bug where someone replaces gen_head with lm_head."""
        B, L_text, L_img = 1, 3, 2
        text_emb = torch.randn(B, L_text, HIDDEN)
        text_mask = torch.ones(B, L_text)
        img_ids = torch.zeros(B, L_img, dtype=torch.long)
        out = stub_model.forward_image_logits(text_emb, text_mask, img_ids)
        # gen_head emits IMG_VOCAB (16384), text vocab is TEXT_VOCAB (64)
        assert out.shape[-1] == IMG_VOCAB
        assert out.shape[-1] != TEXT_VOCAB


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class TestSampleImageTokens:
    def test_shapes_and_dtypes(self, stub_model: JanusProT2I) -> None:
        B, L_text = 2, 4
        cond = torch.randn(B, L_text, HIDDEN)
        uncond = torch.randn(B, L_text, HIDDEN)
        mask = torch.ones(B, L_text)
        toks, lps = stub_model.sample_image_tokens(
            cond, uncond, mask, mask, image_token_num=6,
        )
        assert toks.shape == (B, 6)
        assert lps.shape == (B, 6)
        assert toks.dtype == torch.long
        assert lps.dtype == torch.float32

    def test_logprobs_are_negative(self, stub_model: JanusProT2I) -> None:
        """log_softmax entries are always <= 0."""
        B = 1
        cond = torch.randn(B, 2, HIDDEN)
        uncond = torch.randn(B, 2, HIDDEN)
        mask = torch.ones(B, 2)
        _, lps = stub_model.sample_image_tokens(
            cond, uncond, mask, mask, image_token_num=4,
        )
        assert (lps <= 0.0).all()

    def test_token_ids_in_range(self, stub_model: JanusProT2I) -> None:
        B = 2
        cond = torch.randn(B, 2, HIDDEN)
        uncond = torch.randn(B, 2, HIDDEN)
        mask = torch.ones(B, 2)
        toks, _ = stub_model.sample_image_tokens(
            cond, uncond, mask, mask, image_token_num=4,
        )
        assert (toks >= 0).all() and (toks < IMG_VOCAB).all()


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


class TestDecode:
    def test_pixel_shape(self, stub_model: JanusProT2I) -> None:
        B, L = 2, 16  # 4x4 grid
        out = stub_model.decode_image_tokens(torch.randint(0, IMG_VOCAB, (B, L)))
        assert out.shape == (B, 3, 64, 64)

    def test_pixel_range_clipped(self, stub_model: JanusProT2I) -> None:
        out = stub_model.decode_image_tokens(torch.zeros(1, 4, dtype=torch.long))
        assert out.min() >= -1.0 and out.max() <= 1.0


# ---------------------------------------------------------------------------
# VQ latent-channel resolution — protects 7B/larger variants
# ---------------------------------------------------------------------------


class _ConfigStubVQ(nn.Module):
    """Stub VQ that records the latent_channels it was called with."""

    def __init__(self, config_attrs: dict[str, object] | None = None) -> None:
        super().__init__()
        self.last_shape: list[int] | None = None
        if config_attrs is not None:
            self.config = SimpleNamespace(**config_attrs)
        # else: no .config attribute at all — exercises the fallback branch

    def decode_code(self, ids: torch.Tensor, shape: list[int]) -> torch.Tensor:
        self.last_shape = list(shape)
        B, _, h, w = shape
        return torch.zeros(B, 3, h * 16, w * 16)


def _build_with_vq(vq: nn.Module) -> JanusProT2I:
    mmgpt = _StubMMGPT()
    mmgpt.gen_vision_model = vq
    return JanusProT2I(JanusProConfig(use_lora=False), mmgpt=mmgpt, processor=object())


class TestVQLatentChannels:
    def test_explicit_override_wins(self) -> None:
        vq = _ConfigStubVQ({"z_channels": 8})
        m = _build_with_vq(vq)
        m.config.vq_latent_channels = 17
        m.decode_image_tokens(torch.zeros(1, 4, dtype=torch.long))
        assert vq.last_shape is not None and vq.last_shape[1] == 17

    def test_quantize_embedding_autodetect(self) -> None:
        """Live probe of ``quantize.embedding.weight`` is the authoritative
        source — this is what Janus-Pro-1B actually hits at runtime."""
        vq = _ConfigStubVQ(None)
        # Give the stub a real quantizer with a 5-dim codebook.
        vq.quantize = nn.Module()
        vq.quantize.embedding = nn.Embedding(64, 5)
        m = _build_with_vq(vq)
        m.decode_image_tokens(torch.zeros(1, 4, dtype=torch.long))
        assert vq.last_shape == [1, 5, 2, 2]

    def test_z_channels_intentionally_skipped(self) -> None:
        """Janus-Pro-1B has config.z_channels=256 but codebook dim=8.
        z_channels is encoder-hidden-dim and MUST NOT be used as shape[1]."""
        vq = _ConfigStubVQ({"z_channels": 256})  # trap value — must be ignored
        m = _build_with_vq(vq)
        m.decode_image_tokens(torch.zeros(1, 4, dtype=torch.long))
        # Falls through to the 8-dim constant, NOT 256.
        assert vq.last_shape == [1, 8, 2, 2]

    def test_embed_dim_autodetect(self) -> None:
        """Config ``embed_dim`` is the fallback when live probe misses."""
        vq = _ConfigStubVQ({"embed_dim": 12})
        m = _build_with_vq(vq)
        m.decode_image_tokens(torch.zeros(1, 4, dtype=torch.long))
        assert vq.last_shape == [1, 12, 2, 2]

    def test_latent_channels_attr(self) -> None:
        vq = _ConfigStubVQ({"latent_channels": 6})
        m = _build_with_vq(vq)
        m.decode_image_tokens(torch.zeros(1, 4, dtype=torch.long))
        assert vq.last_shape == [1, 6, 2, 2]

    def test_fallback_to_8(self) -> None:
        """No config, no quantizer probe — last-resort hard-coded default."""
        vq = _ConfigStubVQ(None)
        m = _build_with_vq(vq)
        m.decode_image_tokens(torch.zeros(1, 4, dtype=torch.long))
        assert vq.last_shape == [1, 8, 2, 2]

    def test_garbage_config_raises(self) -> None:
        vq = _ConfigStubVQ({"embed_dim": "twelve"})  # str — invalid
        m = _build_with_vq(vq)
        with pytest.raises(RuntimeError, match="embed_dim"):
            m.decode_image_tokens(torch.zeros(1, 4, dtype=torch.long))

    def test_negative_override_raises(self) -> None:
        vq = _ConfigStubVQ({"embed_dim": 8})
        m = _build_with_vq(vq)
        m.config.vq_latent_channels = -1
        with pytest.raises(RuntimeError, match="positive int"):
            m.decode_image_tokens(torch.zeros(1, 4, dtype=torch.long))


# ---------------------------------------------------------------------------
# disable_adapter
# ---------------------------------------------------------------------------


class TestDisableAdapter:
    def test_raises_when_no_lora_adapter(self, stub_model: JanusProT2I) -> None:
        """Silent-failure red-line: yielding a no-op would make ref == policy."""
        assert stub_model.has_lora_adapter is False
        with pytest.raises(RuntimeError, match="no PEFT adapter"):
            with stub_model.disable_adapter():
                pass

    def test_has_lora_adapter_flag(self, stub_model: JanusProT2I) -> None:
        # Stub LM has no ``disable_adapter`` attr → flag must be False.
        assert stub_model.has_lora_adapter is False


# ---------------------------------------------------------------------------
# Gen-head helper
# ---------------------------------------------------------------------------


def test_image_token_logits_helper(stub_model: JanusProT2I) -> None:
    h = torch.randn(2, 7, HIDDEN)
    out = image_token_logits_from_hidden(stub_model.mmgpt, h)
    assert out.shape == (2, 7, IMG_VOCAB)
