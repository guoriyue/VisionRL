"""Janus-Pro family — DeepSeek's autoregressive multimodal model.

Currently supports Janus-Pro-1B for text-to-image generation under
the visual-rl GRPO pipeline.

Requires the upstream package ``deepseek-ai/Janus`` (not on PyPI):
    git clone https://github.com/deepseek-ai/Janus
    cd Janus && pip install -e .
"""

from __future__ import annotations

from vrl.models.families.janus.model import (
    JanusProConfig,
    JanusProT2I,
    image_token_logits_from_hidden,
)

__all__ = [
    "JanusProConfig",
    "JanusProT2I",
    "image_token_logits_from_hidden",
]
