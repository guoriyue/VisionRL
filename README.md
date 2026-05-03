# visual-rl

> visual-rl is a visual generation RL infrastructure stack spanning
> autoregressive image generation, diffusion models, and world models.
> It focuses on training/rollout interfaces that remain honest to the
> compute regime of each family, instead of pretending one scheduler
> story fits all.

`visual-rl` (package import name: `vrl`, console entry point: `vrl-serve`)
is a research codebase for reinforcement-learning-style post-training of
visual generative models. It treats AR token generators, flow / diffusion
samplers, and video world models as three distinct compute regimes, and
provides shared trainer / collector / evaluator abstractions across them
without forcing a single scheduler narrative on top.

The repo was previously named `wm-infra` → `vision-rl` → `visual-rl`.

---

## Positioning

What this codebase actually is, today:

- A **multi-paradigm visual RL stack**: AR image generation (Janus-Pro,
  NextStep), flow / diffusion (SD3.5, Wan 2.1), and world models
  (Cosmos Predict 1 / 2 / 2.5).
- A **clean trainer / collector / evaluator boundary** so the same GRPO /
  DPO loop can drive token-level RL and step-level RL from one codebase.
- An **engine skeleton** (`vrl/engine/`) — scheduler, batch planner, IPC,
  artifact store, FastAPI gateway — staged for AR continuous batching
  and multi-GPU rollout/train separation.

What this codebase is **not** trying to be:

- Not a Flow-Factory replacement. Flow-Factory wins on supported model
  count and algorithm checklist coverage; this repo does not compete on
  that axis.
- Not a "vLLM-for-RL paradigm applied to diffusion" story. Single-GPU
  diffusion is GPU-saturated under our 2026-04-27 profiles, so there is
  no batching headroom to harvest there.
- Not a "diffusion engine moat" project. Diffusion family code is a
  baseline coverage asset, not the marketing front line.

We do not currently claim **single-GPU continuous batching speedups for
diffusion**, **engine-as-moat for diffusion**, or **SKU parity with
Flow-Factory**. If those land, they will land with benchmarks, not with
README copy.

---

## Three real bets

### Bet A — AR image generation RL (Janus-Pro, primary)

Token-level GRPO over an AR image generator. This is the line closest to
"already tells a story":

- `vrl/algorithms/grpo_token.py` — `TokenGRPO` (token-level RL, distinct
  from diffusion step-level RL)
- `vrl/rollouts/evaluators/ar/token_logprob.py` — token logprob
  evaluator
- `vrl/rollouts/collectors/janus_pro.py` — Janus-Pro rollout collector
- `vrl/models/families/janus_pro/model.py` — Janus-Pro model wrapper
- `vrl/scripts/train.py` plus `configs/experiment/janus_pro_1b_ocr_grpo.yaml`
  — first end-to-end task (OCR reward)
- `vrl/scripts/janus_pro/train.py` — Janus-Pro family training implementation
- `vrl/scripts/nextstep_1/train.py` — second AR family (NextStep) under the
  same TokenGRPO contract
- `tests/algorithms/test_grpo_token.py`,
  `tests/rollouts/test_janus_pro_collector.py`,
  `tests/models/test_janus_wrapper.py` — covering the path

Goal of this line is **not** "also support a Janus model"; it is to
make AR visual RL a first-class path with a clean separation between
token-level and step-level RL.

### Bet B — Video2World / World Model RL (Cosmos, early-stage)

World-model-style RL on top of NVIDIA Cosmos Predict 2. This line is
real but early — there is one experiment script today, and it is staged
as a research thrust, not a polished product:

- `vrl/models/families/cosmos/{model,predict1,predict2,predict25,variants}.py`
- `vrl/rollouts/collectors/cosmos_predict2.py`
- `vrl/scripts/train.py` plus `configs/experiment/cosmos_predict2_2b_grpo.yaml`
  — single GRPO recipe
- `tests/models/test_cosmos_predict2_step.py`,
  `tests/rollouts/test_cosmos_predict2_collector.py`,
  `tests/e2e/test_serving_cosmos.py`

The intent is to push visual RL from image / video aesthetics toward
world-model control. The honest current state: skeleton + one recipe.
Calling it production-ready would be lying.

### Bet C — Selective throughput infra (only where it actually pays)

Throughput work is scoped to workloads where it can produce a real
number, not painted across the whole codebase:

- AR continuous batching (Janus-Pro token generation) — engine
  skeleton at `vrl/engine/batch_planner.py`
  (`ContinuousBatchPlanner`, currently a FIFO baseline, not a moat).
- Multi-GPU rollout / train physical separation — IPC plumbing at
  `vrl/ipc/` (`server.py`, `client.py`, `artifacts.py`,
  `protocol.py`).
- Request-level scheduling and FastAPI serving — `vrl/engine/`
  (`loop.py`, `scheduler.py`, `batch_planner.py`),
  `vrl/gateway/` (FastAPI app, routes), console script `vrl-serve`.

Until these are backed by a measured benchmark, the engine is a
**thesis asset**, not a marketing asset.

---

## Repository layout

```
vrl/
├── algorithms/        GRPO, TokenGRPO, DPO, flow-matching, stat tracking
├── rewards/           aesthetic, CLIP, OCR, PickScore, composite, multi, remote
├── trainers/          online, offline DPO, EMA, FSDP, weight sync, K-repeat data
├── rollouts/
│   ├── collectors/    janus_pro, nextstep_1, sd3_5, wan_2_1, cosmos_predict2
│   └── evaluators/    lm/{token_logprob, continuous_token_logprob},
│                      diffusion/flow_matching
├── models/families/   janus_pro, nextstep_1, sd3_5, wan_2_1, cosmos
├── engine/            loop, scheduler, batch planner, generation runtime
├── gateway/           FastAPI app, bootstrap, routes (vrl-serve)
└── scripts/           per-family training entry points
configs/               base, model, sampling, experiment
tests/                 algorithms, rollouts, models, e2e
```

---

## Differentiation vs Flow-Factory

Flow-Factory is "Easy RL for Diffusion and Flow-Matching Models." It
wins on diffusion SKU breadth, algorithm checklist (DPO / GRPO /
DiffusionNFT / AWM / DGPO / GRPO-Guard), and verified configs.

`visual-rl` does not try to out-cover it on that axis. The intended
delta is:

| Axis                              | Flow-Factory | visual-rl                            |
| --------------------------------- | ------------ | ------------------------------------ |
| Diffusion / flow model SKUs       | broader      | smaller, baseline coverage           |
| Diffusion algorithm checklist     | broader      | GRPO + DPO baseline                  |
| AR image generation RL            | not in scope | first-class (Janus-Pro, NextStep)    |
| World model / Video2World RL      | not in scope | early-stage (Cosmos Predict 2)       |
| Engine / serving / multi-GPU infra| not in scope | skeleton, benchmark-gated as it ships|

If you want the most diffusion SKUs and the most algorithm parity, use
Flow-Factory. If you want AR visual RL or world-model RL with a shared
trainer abstraction, use this repo.

---

## Install

```bash
pip install -e .
# optional extras
pip install -e ".[cosmos]"   # diffusers + transformers + accelerate
pip install -e ".[ocr]"      # rapidocr-onnxruntime for OCR rewards
pip install -e ".[bench]"    # matplotlib + tabulate
pip install -e ".[dev]"      # pytest, ruff, httpx
```

## Run

```bash
# AR image RL (Janus-Pro, OCR reward)
python -m vrl.scripts.train --config experiment/janus_pro_1b_ocr_grpo

# AR image RL (NextStep, OCR reward)
python -m vrl.scripts.train --config experiment/nextstep_1_ocr_grpo

# World model RL (Cosmos Predict 2, GRPO)
python -m vrl.scripts.train --config experiment/cosmos_predict2_2b_grpo

# Diffusion baselines
python -m vrl.scripts.train --config experiment/sd3_5_ocr_grpo
python -m vrl.scripts.train --config experiment/wan_2_1_1_3b_grpo

# Serving gateway
vrl-serve
```

End-to-end serving tests are gated on real model weights via
`WM_RUN_REAL_MODEL_TESTS=1`.

---

## Honesty checklist (what every claim above is grounded in)

This README is **fact-driven**: every feature claim is verifiable with
`grep` or `ls` against the current tree. If you cloned this repo today,
the following commands all return non-empty output:

```bash
# Bet A — AR image RL
ls vrl/algorithms/grpo_token.py
ls vrl/rollouts/evaluators/ar/token_logprob.py
ls vrl/rollouts/collectors/janus_pro.py
ls vrl/models/families/janus_pro/model.py
ls vrl/scripts/train.py
ls vrl/scripts/janus_pro/train.py
ls vrl/scripts/nextstep_1/train.py
ls tests/algorithms/test_grpo_token.py
grep -nE "^class TokenGRPO" vrl/algorithms/grpo_token.py

# Bet B — World model RL (early-stage)
ls vrl/models/families/cosmos/predict2.py
ls vrl/rollouts/collectors/cosmos_predict2.py
ls vrl/scripts/cosmos/train.py
ls configs/experiment/cosmos_predict2_2b_grpo.yaml
ls tests/e2e/test_serving_cosmos.py

# Bet C — Engine skeleton (benchmark-gated)
ls vrl/engine/protocols.py
ls vrl/engine/loop.py
ls vrl/engine/batch_planner.py
ls vrl/ipc/server.py
grep -nE "^class ContinuousBatchPlanner" vrl/engine/batch_planner.py

# Diffusion baseline coverage
ls vrl/models/families/sd3_5 vrl/models/families/wan_2_1
ls vrl/algorithms/grpo.py vrl/algorithms/dpo.py vrl/algorithms/flow_matching.py

# Console entry point
grep -nE "vrl-serve" pyproject.toml
```

If any of these stop returning, the README is wrong and should be
updated before the code.

---

## License

Apache-2.0. See `pyproject.toml`.
