"""AutoregressivePolicy protocol for AR image-generation models.

Janus-Pro and NextStep-1 structurally satisfy this protocol but DO NOT
inherit from a common ABC — their replay log-prob math is different
(categorical for Janus, Gaussian/flow for NextStep). The Protocol
captures only the minimum shared ownership: device, LoRA toggle, and
the replay-forward call.

The ``replay_forward`` return dict schema is intentionally NOT typed —
Janus returns ``{"logits", "target_tokens"}``, NextStep returns
``{"log_probs", "target_tokens"}``. Evaluators dispatch on dict keys.
See ``SPRINT_ar_support.md`` §5 for the rationale on not unifying.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class AutoregressivePolicy(Protocol):
    """Minimal AR policy protocol shared by Janus-Pro and NextStep-1."""

    @property
    def device(self) -> torch.device:
        """Device the policy currently lives on."""
        ...

    def disable_adapter(self) -> AbstractContextManager[None]:
        """Context manager that disables LoRA / adapter weights.

        Used by evaluators to compute the reference-policy log-prob
        (LoRA-off forward) without unloading weights.
        """
        ...

    def replay_forward(self, batch: Any, timestep_idx: int = 0) -> dict[str, Any]:
        """Recompute the training-time forward to produce logits / log-probs.

        The return-dict schema is family-specific:
          - Janus  → ``{"logits": Tensor[B, L, V], "target_tokens": Tensor[B, L]}``
          - NextStep → ``{"log_probs": Tensor[B, L], "target_tokens": Tensor[B, L, D]}``

        Evaluators dispatch on dict keys; do not unify the schema here.
        """
        ...
