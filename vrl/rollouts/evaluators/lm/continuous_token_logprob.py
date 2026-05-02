"""Continuous-token log-probability evaluator for NextStep-1.

Drop-in replacement for ``TokenLogProbEvaluator`` when tokens are
continuous (NextStep-1 family). The LLM produces hidden states; the
flow-matching head turns those into per-token Gaussian log-probabilities
of the *previously sampled* continuous tokens.

Distribution family is ``"continuous_flow"`` — algorithms see the same
``[B, L]`` log-prob tensor as the categorical path, so ``TokenGRPO``
needs no change.
"""

from __future__ import annotations

from typing import Any

import torch

from vrl.rollouts.collectors.base import Collector
from vrl.rollouts.evaluators.base import Evaluator
from vrl.rollouts.evaluators.types import SignalBatch, SignalRequest
from vrl.rollouts.types import ExperienceBatch


def _has_active_adapter(model: Any) -> bool:
    sub = getattr(model, "language_model", None) or model
    return hasattr(sub, "disable_adapter") and callable(sub.disable_adapter)


class ContinuousTokenLogProbEvaluator(Evaluator):
    """Recompute Gaussian log-probs of sampled continuous tokens.

    Two-pass when ``need_ref=True`` — same LoRA-on / LoRA-off contract as
    ``TokenLogProbEvaluator``. Refuses to silently degenerate when the
    caller asks for KL but no real adapter / ref_model is available.
    """

    def __init__(self, mask_key: str = "token_mask") -> None:
        self.mask_key = mask_key

    def evaluate(
        self,
        collector: Collector,
        model: Any,
        batch: ExperienceBatch,
        timestep_idx: int = 0,
        ref_model: Any | None = None,
        signal_request: SignalRequest | None = None,
    ) -> SignalBatch:
        # ``collector`` retained for trainer-interface compatibility — replay
        # ownership now lives on the model (``model.replay_forward``).
        del collector
        request = signal_request or SignalRequest()

        new_lp = self._compute_logprobs(model, batch)

        ref_lp: torch.Tensor | None = None
        if request.need_ref:
            if ref_model is not None:
                ref_lp = self._compute_logprobs(ref_model, batch)
            else:
                if not _has_active_adapter(model):
                    raise RuntimeError(
                        "ContinuousTokenLogProbEvaluator: signal_request.need_ref=True "
                        "but the model has no PEFT adapter to disable AND no ref_model "
                        "was provided. With use_lora=False you must pass a separate "
                        "frozen ref_model — otherwise ref_log_prob would silently "
                        "equal log_prob and KL would be identically zero."
                    )
                with torch.no_grad(), model.disable_adapter():
                    ref_lp = self._compute_logprobs(model, batch)

        aux: dict[str, Any] = {}
        if self.mask_key in batch.extras:
            aux[self.mask_key] = batch.extras[self.mask_key]

        return SignalBatch(
            log_prob=new_lp,
            ref_log_prob=ref_lp,
            entropy=None,
            dist_family="continuous_flow",
            aux=aux,
        )

    @staticmethod
    def _compute_logprobs(
        model: Any,
        batch: ExperienceBatch,
    ) -> torch.Tensor:
        """Forward through ``model.replay_forward`` — return ``[B, L]`` float32 log-probs.

        The model's ``replay_forward`` re-primes the LLM with text, replays
        the AR loop with stashed noise, and re-evaluates the flow head's
        Gaussian density at the sampled tokens. This evaluator just unwraps
        the returned ``log_probs`` and casts to float32.
        """
        out = model.replay_forward(batch, timestep_idx=0)
        log_probs: torch.Tensor = out["log_probs"]   # [B, L]
        return log_probs.float()
