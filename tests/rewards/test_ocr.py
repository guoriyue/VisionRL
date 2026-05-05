"""Tests for vrl.rewards.ocr (OCRReward)."""

from __future__ import annotations

import pytest

from vrl.rewards.ocr import OCRReward, _normalize_text, _normalized_edit_distance

# ---------------------------------------------------------------------------
# Unit tests for text helpers
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_basic(self) -> None:
        assert _normalize_text("  Hello, World!  ") == "hello world"

    def test_collapses_whitespace(self) -> None:
        assert _normalize_text("a   b\tc") == "a b c"

    def test_strips_punctuation(self) -> None:
        assert _normalize_text("EXIT-42.") == "exit42"

    def test_empty(self) -> None:
        assert _normalize_text("") == ""


class TestNormalizedEditDistance:
    def test_identical(self) -> None:
        assert _normalized_edit_distance("hello", "hello") == pytest.approx(0.0)

    def test_completely_different(self) -> None:
        dist = _normalized_edit_distance("abc", "xyz")
        assert dist > 0.5

    def test_empty_both(self) -> None:
        assert _normalized_edit_distance("", "") == pytest.approx(0.0)

    def test_partial_match(self) -> None:
        dist = _normalized_edit_distance("hello", "helo")
        assert 0.0 < dist < 1.0

    def test_symmetry(self) -> None:
        d1 = _normalized_edit_distance("abc", "abcd")
        d2 = _normalized_edit_distance("abcd", "abc")
        assert d1 == pytest.approx(d2, abs=0.01)


class _FakePaddleOCR:
    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)

    def ocr(self, frame, cls=False):
        del frame, cls
        text = self.texts.pop(0)
        return [[(None, (text, 1.0))]]


# ---------------------------------------------------------------------------
# OCRReward scoring tests (require rapidocr_onnxruntime)
# ---------------------------------------------------------------------------

def _has_rapidocr() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


_skip_no_rapidocr = pytest.mark.skipif(
    not _has_rapidocr(), reason="rapidocr_onnxruntime not installed"
)


def _make_ocr_rollout(target_text: str, video_tensor=None):
    """Build a minimal Rollout with target_text metadata and a video tensor."""
    import torch

    from vrl.algorithms.types import Rollout, Trajectory

    if video_tensor is None:
        # Black frames — no OCR text expected
        video_tensor = torch.zeros(3, 8, 64, 64)

    traj = Trajectory(prompt="test", seed=0, steps=[], output=video_tensor)
    return Rollout(request=None, trajectory=traj, metadata={"target_text": target_text})


@_skip_no_rapidocr
@pytest.mark.asyncio
class TestOCRRewardScoring:
    async def test_no_target_text_returns_zero(self) -> None:
        reward = OCRReward(device="cpu")
        rollout = _make_ocr_rollout("")
        score = await reward.score(rollout)
        assert score == pytest.approx(0.0)

    async def test_black_frames_low_score(self) -> None:
        """Black frames should have no readable text → low score."""
        reward = OCRReward(device="cpu")
        rollout = _make_ocr_rollout("HELLO")
        score = await reward.score(rollout)
        assert score <= 0.5

    async def test_score_batch_length(self) -> None:
        reward = OCRReward(device="cpu")
        rollouts = [_make_ocr_rollout("A"), _make_ocr_rollout("B")]
        scores = await reward.score_batch(rollouts)
        assert len(scores) == 2


@pytest.mark.asyncio
async def test_image_ocr_substring_match_gets_full_credit() -> None:
    import torch

    reward = OCRReward(device="cpu")
    reward._engine = _FakePaddleOCR(["Cafe Free WiFi Open"])
    rollout = _make_ocr_rollout(
        "Free WiFi",
        video_tensor=torch.zeros(3, 64, 64),
    )

    score = await reward.score(rollout)

    assert score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_video_ocr_keeps_flow_grpo_video_edit_distance_behavior() -> None:
    import torch

    reward = OCRReward(device="cpu")
    reward._engine = _FakePaddleOCR(["Free WiFiX", "Free WiFiX"])
    rollout = _make_ocr_rollout(
        "Free WiFi",
        video_tensor=torch.zeros(3, 8, 64, 64),
    )

    score = await reward.score(rollout)

    assert 0.0 < score < 1.0
