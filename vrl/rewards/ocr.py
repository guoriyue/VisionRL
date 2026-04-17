"""OCR reward function using rapidocr_onnxruntime.

Scores generated images/video by how well OCR-detected text matches a
target string provided in rollout metadata.
"""

from __future__ import annotations

import re
import string
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from vrl.algorithms.types import Rollout
from vrl.rewards.base import RewardFunction


def _normalize_text(text: str) -> str:
    """Lowercase, strip, collapse whitespace, remove punctuation."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalized_edit_distance(a: str, b: str) -> float:
    """Normalized edit distance via SequenceMatcher.

    Returns a value in [0, 1] where 0 means identical strings and 1 means
    completely different.
    """
    if not a and not b:
        return 0.0
    return 1.0 - SequenceMatcher(None, a, b).ratio()


def _safe_filename_fragment(text: str, max_len: int = 24) -> str:
    """Sanitize arbitrary text for use inside a filename."""
    return re.sub(r"[^A-Za-z0-9]+", "_", text)[:max_len].strip("_") or "empty"


class OCRReward(RewardFunction):
    """OCR-based text matching reward.

    Uses ``rapidocr_onnxruntime`` to detect text in generated frames and
    computes a similarity score against a target string from rollout metadata.
    The OCR engine is loaded lazily on first call.

    When ``debug_dir`` is set, dumps the best-scoring frame along with the
    OCR-detected text and target to disk — crucial sanity check for
    catching reward hacking (e.g. PPT-style white-on-black degeneration).
    """

    def __init__(self, device: str = "cuda", debug_dir: str | None = None) -> None:
        self._device = device
        self._engine: Any = None
        self._debug_dir = Path(debug_dir) if debug_dir else None
        self._debug_counter = 0
        if self._debug_dir is not None:
            self._debug_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_loaded(self) -> None:
        if self._engine is not None:
            return
        from rapidocr_onnxruntime import RapidOCR

        self._engine = RapidOCR()

    async def score(self, rollout: Rollout) -> float:
        self._ensure_loaded()
        import numpy as np
        import torch

        target_text_raw = rollout.metadata.get("target_text", "")
        if not target_text_raw:
            return 0.0

        target_text = _normalize_text(target_text_raw)

        output = rollout.trajectory.output

        # ---- extract frames as list of numpy uint8 [H, W, C] ----
        frames: list[np.ndarray] = []

        if isinstance(output, torch.Tensor):
            raw = (output * 255).round().clamp(0, 255).to(torch.uint8)

            if raw.ndim == 4 and raw.shape[0] <= 4:
                # [C, T, H, W] video
                t = raw.shape[1]
                indices = [t // 4, t // 2, 3 * t // 4]
                for idx in indices:
                    frame = raw[:, idx, :, :]  # [C, H, W]
                    frames.append(frame.cpu().numpy().transpose(1, 2, 0))
            elif raw.ndim == 4 and raw.shape[0] > 4:
                # [B, C, H, W] image batch — sample 3 evenly spaced
                b = raw.shape[0]
                indices = [b // 4, b // 2, 3 * b // 4]
                for idx in indices:
                    frames.append(raw[idx].cpu().numpy().transpose(1, 2, 0))
            elif raw.ndim == 3:
                # [C, H, W] single image
                frames.append(raw.cpu().numpy().transpose(1, 2, 0))
            else:
                # [T, C, H, W] or other — sample 3 frames
                t = raw.shape[0]
                indices = [t // 4, t // 2, 3 * t // 4]
                for idx in indices:
                    frames.append(raw[idx].cpu().numpy().transpose(1, 2, 0))
        else:
            # Assume PIL or numpy already
            frames = [np.asarray(output)]

        # ---- run OCR on each frame and find best match ----
        best_dist = 1.0
        best_frame: np.ndarray | None = None
        best_ocr_text: str = ""

        for frame in frames:
            result, _ = self._engine(frame)
            if result is None:
                continue
            # result is list of [bbox, text, confidence]
            ocr_text_raw = " ".join(item[1] for item in result)
            ocr_text = _normalize_text(ocr_text_raw)

            if len(ocr_text) < 2:
                continue

            dist = _normalized_edit_distance(ocr_text, target_text)
            if dist < best_dist:
                best_dist = dist
                best_frame = frame
                best_ocr_text = ocr_text_raw

        score_value = 1.0 - best_dist

        if self._debug_dir is not None and best_frame is not None:
            self._dump_debug_frame(best_frame, target_text_raw, best_ocr_text, score_value)

        return score_value

    def _dump_debug_frame(
        self,
        frame: Any,
        target: str,
        ocr_text: str,
        score_value: float,
    ) -> None:
        """Save best frame + metadata to debug_dir. Failure is non-fatal."""
        try:
            from PIL import Image

            idx = self._debug_counter
            self._debug_counter += 1
            tag = _safe_filename_fragment(target)
            img_path = self._debug_dir / f"{idx:06d}_{tag}_score{score_value:.3f}.png"
            meta_path = self._debug_dir / f"{idx:06d}_{tag}_score{score_value:.3f}.txt"

            Image.fromarray(frame).save(img_path)
            meta_path.write_text(
                f"target: {target}\nocr:    {ocr_text}\nscore:  {score_value:.4f}\n",
                encoding="utf-8",
            )
        except Exception:
            # Debug dump is best-effort; never fail the training step.
            pass

    async def score_batch(self, rollouts: list[Rollout]) -> list[float]:
        return [await self.score(r) for r in rollouts]
