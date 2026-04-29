"""NextStep-1 family — StepFun's continuous-token autoregressive model.

NextStep-1 is a 14B AR transformer paired with a 157M flow-matching head
for continuous image tokens. ICLR 2026 Oral.

Requires the upstream package ``stepfun-ai/NextStep-1`` (not on PyPI):
    git clone https://github.com/stepfun-ai/NextStep-1
    cd NextStep-1 && pip install -e .

The wrapper here mirrors ``vrl.models.families.janus_pro`` so the same
``OnlineTrainer + TokenGRPO`` machinery works without changes — the only
substantive difference is that "logits" become per-token Gaussian
log-probabilities (continuous tokens, no codebook).
"""

from __future__ import annotations

from vrl.models.families.nextstep_1.policy import (
    NextStep1Config,
    NextStep1Policy,
)

__all__ = [
    "NextStep1Config",
    "NextStep1Policy",
]
