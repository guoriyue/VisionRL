"""Token-level log-probability evaluator for AR policies.

Wraps ``model.replay_forward`` (which returns full logits) and gathers the
log-probabilities of the *sampled* tokens — both under the current policy
and, when needed, under the LoRA-off reference policy.

The returned ``SignalBatch`` is fed to ``TokenGRPO.compute_signal_loss``.

Distribution family is ``"categorical"`` — flow-matching latent-space KL
intermediates are unused and stay ``None``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from vrl.rollouts.collectors.base import Collector
from vrl.rollouts.evaluators.base import Evaluator
from vrl.rollouts.evaluators.types import SignalBatch, SignalRequest
from vrl.rollouts.types import ExperienceBatch


def _has_active_adapter(model: Any) -> bool:
    """Detect whether ``model`` carries a real PEFT adapter we can disable.

    PEFT injects ``disable_adapter`` onto the wrapped sub-module
    (``model.language_model`` for ``JanusProPolicy``) — the *outer*
    JanusProPolicy always exposes a ``disable_adapter`` method, but its
    no-op fallback fires silently when no LoRA was attached. We probe the
    sub-module directly to avoid that false positive.
    """
    sub = getattr(model, "language_model", None)
    if sub is None:
        sub = model
    return hasattr(sub, "disable_adapter") and callable(sub.disable_adapter)


class TokenLogProbEvaluator(Evaluator):
    """Recompute per-token log-probs of sampled tokens under the policy.

    Two-pass when ``need_ref=True``:
      1. forward through ``model`` with the LoRA adapter ON  → log_prob
      2. forward through ``model`` with the LoRA adapter OFF (or
         through ``ref_model`` if provided) → ref_log_prob

    Refuses-to-fail-silently contract:
      If the caller asks for ``need_ref=True`` but neither (a) provides
      an explicit ``ref_model`` nor (b) supplies a model with a real
      PEFT adapter, we raise — never silently produce ``ref_lp == lp``,
      because that yields KL ≡ 0 and the trainer would happily report
      sane-looking metrics on a broken loss.
    """

    def __init__(self, mask_key: str = "token_mask") -> None:
        self.mask_key = mask_key

    def evaluate(
        self,
        collector: Collector,  # TODO: remove unused collector arg
        model: Any,
        batch: ExperienceBatch,
        timestep_idx: int = 0,
        ref_model: Any | None = None,
        signal_request: SignalRequest | None = None,
    ) -> SignalBatch:
        # ``collector`` retained for Trainer-interface compatibility — replay
        # ownership now lives on the model (``model.replay_forward``).
        del collector
        request = signal_request or SignalRequest()
        action_ids: torch.Tensor = batch.actions  # [B, L_img]

        new_lp = self._compute_logprobs(model, batch, action_ids)

        ref_lp = None
        if request.need_ref:
            if ref_model is not None:
                ref_lp = self._compute_logprobs(
                    ref_model, batch, action_ids,
                )
            else:
                if not _has_active_adapter(model):
                    raise RuntimeError(
                        "TokenLogProbEvaluator: signal_request.need_ref=True "
                        "but the model has no PEFT adapter to disable AND no "
                        "ref_model was provided. With use_lora=False you must "
                        "pass a separate frozen ref_model — otherwise "
                        "ref_log_prob would silently equal log_prob and the "
                        "KL penalty would be identically zero, which the "
                        "trainer cannot detect."
                    )
                # LoRA-off pass on the same module — only safe when the
                # adapter is real (verified above).
                with torch.no_grad(), model.disable_adapter():
                    ref_lp = self._compute_logprobs(
                        model, batch, action_ids,
                    )

        aux: dict[str, Any] = {}
        if self.mask_key in batch.extras:
            aux[self.mask_key] = batch.extras[self.mask_key]

        return SignalBatch(
            log_prob=new_lp,
            ref_log_prob=ref_lp,
            entropy=None,
            dist_family="categorical",
            aux=aux,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_logprobs(
        model: Any,
        batch: ExperienceBatch,
        action_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward + gather. Always returns ``[B, L]`` float32 log-probs."""
        out = model.replay_forward(batch, timestep_idx=0)
        logits: torch.Tensor = out["logits"]   # [B, L, V_img]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        gathered = log_probs.gather(-1, action_ids.unsqueeze(-1)).squeeze(-1)
        return gathered  # [B, L]
