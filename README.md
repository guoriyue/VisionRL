# visual-rl

`visual-rl` (Python package: `vrl`) is a research infrastructure stack for
RL-style post-training of visual generative models.

The codebase covers three rollout regimes:

- Autoregressive image generation: Janus-Pro and NextStep-1.
- Diffusion / flow generation: SD3.5 and Wan 2.1.
- Video2World generation: Cosmos Predict2.

The main design goal is to keep the training loop shared while keeping each
family's execution semantics explicit. AR token rollout, continuous-token AR
rollout, diffusion denoising, and Video2World generation do not pretend to be
the same runtime problem.

---

## Current Positioning

This codebase currently is:

- A multi-family visual RL stack with GRPO, TokenGRPO, DPO, flow-matching
  replay utilities, reward composition, rollout collection, and online/offline
  trainers.
- A YAML-driven training entrypoint. `vrl.scripts.train` loads one experiment
  config, reads `trainer.entrypoint`, imports that callable, and runs it.
- A registry-driven rollout adapter. `vrl.rollouts.families.specs` binds each
  family to its collector config, request sampling fields, runtime builder,
  executor class, and chunk gatherer.
- A Ray-only online rollout backend. Single-GPU and multi-GPU online rollout
  jobs both use the same Ray launcher and actor path; only `num_workers` and
  resource config change.
- A small generation engine layer. `vrl.engine` owns typed requests, output
  payloads, executor protocols, microbatching, chunk gather, AR scheduling
  helpers, and diffusion denoising helpers. It is not a standalone serving
  engine.

This codebase currently is not:

- A vLLM or SGLang replacement.
- A Flow-Factory replacement with broad diffusion model coverage.
- A benchmarked diffusion throughput engine. Do not claim diffusion speedups
  here without measured results.
- A local-backend rollout stack. Online rollout is Ray-only; the in-process
  `GenerationRuntime` is still useful as the worker facade inside Ray actors
  and for narrow runtime tests.

---

## Main Workflows

### Autoregressive Image RL

AR rollout is the path for Janus-Pro and NextStep-1. It uses TokenGRPO and
family-local executors for model-specific stepping.

Key files:

- `vrl/algorithms/grpo_token.py`
- `vrl/models/families/janus_pro/{builder.py,policy.py,executor.py}`
- `vrl/models/families/nextstep_1/{builder.py,policy.py,executor.py,flow_step.py}`
- `vrl/engine/ar/{sequence.py,spec.py,token_scheduler.py,executor_base.py}`
- `vrl/rollouts/packers/{ar_discrete.py,ar_continuous.py}`
- `vrl/rollouts/evaluators/ar/{token_logprob.py,continuous_token_logprob.py}`
- `vrl/scripts/{janus_pro,nextstep_1}/train.py`

Active experiment configs:

- `configs/experiment/janus_pro_1b_grpo.yaml`
- `configs/experiment/janus_pro_1b_ocr_grpo.yaml`
- `configs/experiment/nextstep_1_ocr_grpo.yaml`

### Diffusion / Flow RL

Diffusion rollout covers SD3.5 and Wan 2.1. It uses diffusion policy modules,
family-local executors, flow-matching replay evaluation, and diffusion rollout
packers.

Key files:

- `vrl/algorithms/{grpo.py,dpo.py,flow_matching.py}`
- `vrl/models/diffusion.py`
- `vrl/models/families/sd3_5/{builder.py,policy.py,executor.py}`
- `vrl/models/families/wan_2_1/{builder.py,diffusers_policy.py,official_policy.py,executor.py}`
- `vrl/engine/diffusion/{spec.py,denoise.py,executor_base.py}`
- `vrl/rollouts/packers/diffusion.py`
- `vrl/rollouts/evaluators/diffusion/flow_matching.py`
- `vrl/scripts/{sd3_5,wan_2_1}/train.py`
- `vrl/scripts/wan_2_1/train_dpo.py`

Active experiment configs:

- `configs/experiment/sd3_5_ocr_grpo.yaml`
- `configs/experiment/wan_2_1_1_3b_grpo.yaml`
- `configs/experiment/wan_2_1_1_3b_ocr_grpo.yaml`
- `configs/experiment/wan_2_1_1_3b_multi_reward_grpo.yaml`
- `configs/experiment/wan_2_1_14b_grpo.yaml`
- `configs/experiment/wan_2_1_1_3b_dpo.yaml`

### Video2World RL

Cosmos Predict2 is wired as a Video2World diffusion-style family. The current
scope is Predict2, not Predict1 or Predict2.5.

Key files:

- `vrl/models/families/cosmos/{builder.py,policy.py,executor.py}`
- `configs/model/cosmos/predict2_2b.yaml`
- `configs/sampling/cosmos_v2w_704p_93f.yaml`
- `configs/experiment/cosmos_predict2_2b_grpo.yaml`
- `vrl/scripts/cosmos/train.py`

### Ray Online Rollout

Online rollout uses Ray for both one GPU and many GPUs. The trainer constructs
serializable runtime inputs, launches Ray rollout workers, sends generation
requests, gathers chunked outputs, and optionally syncs LoRA trainable state.

Key files:

- `vrl/rollouts/runtime/{config.py,backend.py,launch_inputs.py}`
- `vrl/distributed/ray/placement/{group.py,network.py}`
- `vrl/distributed/ray/rollout/{launcher.py,worker.py,executor.py,planner.py,runtime.py,types.py,weight_sync.py}`
- `configs/base/distributed/{ray_rollout_single_gpu.yaml,ray_rollout.yaml}`

Use `ray_rollout_single_gpu` for local single-GPU smoke/debug jobs and
`ray_rollout` or config overrides for multi-worker rollout. The launch path is
the same in both cases.

---

## Repository Layout

```text
vrl/
+-- algorithms/          GRPO, TokenGRPO, DPO, flow matching, stat tracking
+-- config/              YAML loading, validation, CLI helpers
+-- distributed/ray/     Ray placement, rollout actors, train actor primitives
+-- engine/              Generation request contracts and executor primitives
|   +-- ar/              AR sequence scheduling and executor base classes
|   +-- core/            Runtime, worker, registry, protocols, typed payloads
|   +-- diffusion/       Diffusion executor base and denoising helpers
+-- models/              Shared policy interfaces and family implementations
|   +-- families/        cosmos, janus_pro, nextstep_1, sd3_5, wan_2_1
+-- rewards/             Aesthetic, CLIP, OCR, PickScore, composite rewards
+-- rollouts/            Collector, request adapter, evaluators, packers, runtime
|   +-- collector/
|   +-- evaluators/
|   +-- families/
|   +-- packers/
|   +-- runtime/
+-- scripts/             YAML-selected per-family training entrypoints
+-- trainers/            Online trainer, offline DPO trainer, weight sync helpers

configs/
+-- base/                Shared algorithm, actor, rollout, distributed, trainer
+-- dataset/             Offline preference datasets
+-- experiment/          Fully composed experiment recipes
+-- model/               Family/model-specific config slices
+-- sampling/            Sampling shape and rollout sampling config

tests/
+-- algorithms/
+-- config/
+-- distributed/ray/
+-- engine/generation/
+-- models/
+-- rewards/
+-- rollouts/
+-- trainers/
```

---

## Install

```bash
pip install -e .

# Online rollout jobs require Ray. It is imported lazily and is not part of
# the base package dependency set.
pip install ray

# Optional extras
pip install -e ".[cosmos]"   # diffusers, transformers, accelerate
pip install -e ".[ocr]"      # rapidocr-onnxruntime for OCR rewards
pip install -e ".[bench]"    # matplotlib, tabulate
pip install -e ".[dev]"      # pytest, ruff, httpx
```

---

## Run

Training jobs are selected by YAML. The unified CLI does not branch on
model family or algorithm kind; it imports the callable declared by
`trainer.entrypoint`.

```bash
# Janus-Pro TokenGRPO
python -m vrl.scripts.train --config experiment/janus_pro_1b_grpo
python -m vrl.scripts.train --config experiment/janus_pro_1b_ocr_grpo

# NextStep-1 TokenGRPO
python -m vrl.scripts.train --config experiment/nextstep_1_ocr_grpo

# SD3.5 and Wan 2.1 GRPO
python -m vrl.scripts.train --config experiment/sd3_5_ocr_grpo
python -m vrl.scripts.train --config experiment/wan_2_1_1_3b_grpo
python -m vrl.scripts.train --config experiment/wan_2_1_1_3b_ocr_grpo
python -m vrl.scripts.train --config experiment/wan_2_1_1_3b_multi_reward_grpo
python -m vrl.scripts.train --config experiment/wan_2_1_14b_grpo

# Wan 2.1 offline DPO
python -m vrl.scripts.train --config experiment/wan_2_1_1_3b_dpo

# Cosmos Predict2 Video2World GRPO
python -m vrl.scripts.train --config experiment/cosmos_predict2_2b_grpo
```

Equivalent console entrypoint:

```bash
vrl-train --config experiment/janus_pro_1b_ocr_grpo
```

Multi-worker rollout is a config change, not a different launch path:

```bash
python -m vrl.scripts.train \
  --config experiment/wan_2_1_1_3b_grpo \
  distributed.rollout.num_workers=4 \
  distributed.rollout.placement_strategy=SPREAD \
  distributed.rollout.allow_driver_gpu_overlap=false \
  distributed.rollout.release_after_collect=false
```

---

## Verification

Fast structural checks:

```bash
python -m pytest -q tests/config/test_load_all_experiments.py
python -m pytest -q tests/rollouts/test_family_registry.py
python -m pytest -q tests/distributed/ray/test_rollout_launcher.py
```

Broader local checks:

```bash
python -m pytest -q tests/algorithms tests/config tests/rollouts tests/models tests/rewards tests/trainers
python -m pytest -q tests/engine/generation
```

Real model / GPU smoke tests are opt-in and may require model weights:

```bash
WM_RUN_REAL_MODEL_TESTS=1 python -m pytest -q tests/distributed/ray/test_real_ray_rollout_smoke.py
```

---

## Honesty Checklist

The claims above should stay tied to real files. If any command below stops
returning output, update the README or the code in the same change.

```bash
# YAML-driven entrypoint
grep -nE "trainer.entrypoint|resolve_train_target" vrl/scripts/train.py
grep -nE "entrypoint:" configs/experiment/*.yaml

# Family registry and rollout request adapter
grep -nE "^FAMILY_REGISTRY" vrl/rollouts/families/specs.py
grep -nE "^class RolloutEngineRequestBuilder" vrl/rollouts/collector/requests.py
ls vrl/rollouts/collector/{core.py,factory.py,configs.py,requests.py,rewards.py}
ls vrl/rollouts/packers/{ar_discrete.py,ar_continuous.py,diffusion.py}

# AR rollout
grep -nE "^class TokenGRPO" vrl/algorithms/grpo_token.py
grep -nE "^class JanusProPolicy" vrl/models/families/janus_pro/policy.py
grep -nE "^class NextStep1Policy" vrl/models/families/nextstep_1/policy.py
ls vrl/engine/ar/{sequence.py,spec.py,token_scheduler.py,executor_base.py}
ls vrl/rollouts/evaluators/ar/{token_logprob.py,continuous_token_logprob.py}

# Diffusion and Video2World rollout
grep -nE "^class SD3_5Policy" vrl/models/families/sd3_5/policy.py
grep -nE "^class WanT2V" vrl/models/families/wan_2_1/{diffusers_policy.py,official_policy.py}
grep -nE "^class CosmosPredict2Policy" vrl/models/families/cosmos/policy.py
grep -nE "^class FlowMatchingEvaluator" vrl/rollouts/evaluators/diffusion/flow_matching.py
ls vrl/engine/diffusion/{spec.py,denoise.py,executor_base.py}

# Ray-only online rollout
grep -nE "backend must be 'ray'" vrl/rollouts/runtime/config.py
grep -nE "^class RayRolloutLauncher" vrl/distributed/ray/rollout/launcher.py
grep -nE "^class RayRolloutWorker" vrl/distributed/ray/rollout/worker.py
ls configs/base/distributed/{ray_rollout_single_gpu.yaml,ray_rollout.yaml}
```

---

## License

Apache-2.0. See `pyproject.toml`.
