# DESIGN: World Model RL Phase 1 — Cosmos Predict2-2B Video2World GRPO

Status: ACTIVE SPEC, NEEDS REAL GPU RUN, updated 2026-05-02
Owner: visual-rl
Scope: one concrete Phase-1 recipe, not a sweep.

---

## Current Verdict

The experiment intent is still applicable:

- Train Cosmos Predict2-2B-Video2World with GRPO.
- Use one fixed reference image for all prompts.
- Optimize the single `aesthetic` reward.
- Evaluate paired `LoRA - base` reward delta on the same prompts and seeds.
- Do not add a new model, reward, evaluator, or collector for Phase 1.

The old operational content has been removed. This document now tracks only the
current YAML-driven runbook and the blockers that still need resolution before a
defensible Phase-1 result.

---

## Current Implementation Map

The current runnable path is:

- Entry point:
  `vrl/scripts/cosmos/cosmos_predict2_2b_grpo.py`
- Training driver:
  `vrl/scripts/cosmos/train.py`
- Experiment config:
  `configs/experiment/cosmos_predict2_2b_grpo.yaml`
- Model config:
  `configs/model/cosmos/predict2_2b.yaml`
- Sampling config:
  `configs/sampling/cosmos_v2w_704p_93f.yaml`
- Cosmos policy adapter:
  `vrl/models/families/cosmos/predict2_policy.py`
- Cosmos generation executor:
  `vrl/models/families/cosmos/executor.py`
- RL collector:
  `vrl/rollouts/collectors/cosmos_predict2.py`
- Diffusion evaluator:
  `vrl/rollouts/evaluators/diffusion/flow_matching.py`
- Reward registry:
  `vrl/rewards/multi.py`
- Aesthetic reward:
  `vrl/rewards/aesthetic.py`

The entry point accepts `--config` plus OmegaConf dotlist overrides only.
Experiment values must be passed as config overrides or committed into YAML.

---

## Phase 1 Recipe

### Model

Use:

```text
nvidia/Cosmos-Predict2-2B-Video2World
```

Current config location:

```text
configs/model/cosmos/predict2_2b.yaml
```

This recipe uses LoRA by default:

```text
rank=64
alpha=32
target_modules=[
  "to_k", "to_q", "to_v", "to_out.0",
  "add_k_proj", "add_q_proj", "add_v_proj", "to_add_out"
]
```

### Conditioning

Use exactly one fixed reference image for all prompts. This is still the right
Phase-1 choice because it keeps the measured variable to prompt-conditioned
generation quality rather than mixing prompt effects with reference-image
variation.

Important: the current default `model.reference_image` is empty. That path is
acceptable for unit tests only. A real Phase-1 run must set a real image:

```text
model.reference_image=/absolute/path/to/reference.jpg
```

If this is left empty, the training driver logs that Video2World will use zero
conditioning. That run should not be used as a thesis result.

### Sampling

Use the current Cosmos sampling preset:

```text
width=1280
height=704
num_frames=93
num_steps=35
guidance_scale=7.0
fps=16
cfg=true
```

Config location:

```text
configs/sampling/cosmos_v2w_704p_93f.yaml
```

Only lower `num_frames` or resolution for a smoke/debug run if the real run OOMs;
document that as a deviation from the Phase-1 recipe.

### Reward

Use a single reward:

```text
reward.components.aesthetic=1.0
reward.kwargs.aesthetic.model_name=openai/clip-vit-large-patch14
```

`AestheticReward` is still the best current Phase-1 reward because it scores
three frames from a `[C, T, H, W]` video. This is a frame-aesthetic reward, not a
temporal-consistency reward and not an action-following reward.

### Train/Eval Prompts

Use `datasets/drawbench/test.txt` as the deterministic prompt source. The
deterministic split is committed in the repo:

```text
datasets/drawbench/eval_64.txt   = test.txt lines 1-64
datasets/drawbench/train_192.txt = test.txt lines 65-256
```

The Cosmos experiment config now points at these files by default:

```text
data.manifest=datasets/drawbench/train_192.txt
eval.prompts_file=datasets/drawbench/eval_64.txt
```

You only need `data.manifest=...` or `eval.prompts_file=...` overrides when
intentionally running a different split.

### Training Hyperparameters

Use the current YAML defaults unless a smoke run requires a temporary override:

```text
actor.optim.lr=1e-5
actor.ppo_epochs=1
actor.bf16=true
actor.gradient_checkpointing=true
actor.ema.enable=true
actor.ema.decay=0.9
actor.ema.update_interval=8

algorithm.eps_clip=1e-3
algorithm.init_kl_coef=0.004
algorithm.kl_reward=0.0
algorithm.adv_clip_max=5.0
algorithm.global_std=false
algorithm.per_prompt_stat_tracking=true

rollout.n=4
rollout.rollout_batch_size=1
rollout.same_latent=false
rollout.sde.window_size=0
```

For the Phase-1 result run:

```text
trainer.total_epochs=200
trainer.save_freq=100
trainer.log_freq=1
```

---

## Updated Runbook

### 1. Unit-Level Verification

This checks the current engine/executor/config path without loading real Cosmos
weights:

```bash
pytest \
  tests/engine/generation/test_pipeline_cosmos.py \
  tests/engine/generation/test_cosmos_executor_parity.py \
  tests/rollouts/test_evaluators.py \
  tests/config/test_load_all_experiments.py \
  -q
```

Expected status as of this update:

```text
94 passed
```

### 2. Real-Model Smoke Run

Requires real model access, a GPU-capable environment, and a real reference
image.

```bash
python -m vrl.scripts.cosmos.cosmos_predict2_2b_grpo \
  model.reference_image=/absolute/path/to/reference.jpg \
  trainer.total_epochs=8 \
  trainer.save_freq=4 \
  trainer.log_freq=1 \
  trainer.output_dir=outputs/cosmos_pred2_2b_mve_smoke \
  trainer.debug.first_step=true
```

Smoke pass criteria:

- `metrics.csv` is written.
- `checkpoint-4`, `checkpoint-8`, and `checkpoint-final` are written.
- Eval middle-frame PNGs are non-empty and not obviously collapsed.
- Epoch-0 `clip_fraction` and `approx_kl` do not trigger the configured sanity
  thresholds.

Current caveat: the code logs sanity-threshold failures as warnings; it does not
hard-abort. Treat those warnings as a manual stop for Phase 1 unless the code is
changed to raise.

### 3. Phase-1 Training Run

```bash
python -m vrl.scripts.cosmos.cosmos_predict2_2b_grpo \
  model.reference_image=/absolute/path/to/reference.jpg \
  trainer.total_epochs=200 \
  trainer.save_freq=100 \
  trainer.log_freq=1 \
  trainer.output_dir=outputs/cosmos_pred2_2b_mve_day2
```

Track:

- `outputs/cosmos_pred2_2b_mve_day2/metrics.csv`
- `checkpoint-100/eval_samples/*.png`
- `checkpoint-200/eval_samples/*.png`
- `checkpoint-final/lora_weights`

### 4. Paired Eval-Only Run

```bash
python -m vrl.scripts.cosmos.cosmos_predict2_2b_grpo \
  eval.eval_only=true \
  model.lora.path=outputs/cosmos_pred2_2b_mve_day2/checkpoint-200/lora_weights \
  eval.seeds=1 \
  model.reference_image=/absolute/path/to/reference.jpg \
  trainer.output_dir=outputs/cosmos_pred2_2b_mve_day2_eval
```

Output:

```text
outputs/cosmos_pred2_2b_mve_day2_eval/eval_only/eval_results.csv
```

Primary metric:

```text
delta = mean(lora_score) - mean(base_score)
```

Phase-1 pass gate:

- `delta > 0`
- at least `16 / 64` eval prompts have `delta > 0`
- no visible mode collapse in saved middle-frame PNGs
- no epoch-0 sanity warning from the smoke run

---

## Still To Solve

These are the only remaining items that block a defensible Phase-1 result.

### 1. Choose The Fixed Reference Image

The current config leaves `model.reference_image` empty. Pick one real reference
image and record its absolute path or commit a small non-secret fixture image.
The final result is not reproducible without this.

### 2. Run A Real Cosmos Smoke Test

The unit tests only validate local wiring. They do not prove that the real
Cosmos checkpoint loads, fits in memory, decodes 1280x704x93 video, or produces
usable reward gradients.

### 3. Decide Whether Sanity Gates Should Hard-Fail

The driver currently logs high `clip_fraction` / `approx_kl` as warnings. The
old design treated those as abort conditions. For a reliable experiment, either
manually stop on those warnings or change the code to raise when gates fail.

### 4. Complete The Paired LoRA-vs-Base Eval

No Phase-1 claim is valid until `eval_results.csv` exists and shows:

```text
mean(lora_score) > mean(base_score)
improved_prompts >= 16
```

### 5. Keep The World-Model Claim Narrow

Current code has `VideoGenerationRequest.action_sequence`, but the Cosmos path
does not consume action sequences. Phrase Phase 1 as:

```text
image-conditioned Video2World RL with visual reward
```

Do not claim action-conditioned world-model control unless a Phase-2 action path
and action-following reward are built.

### 6. Verify Real Eval Cost And Memory

The 1280x704, 93-frame eval path remains hardware-dependent. If the real eval
OOMs, document the exact sampling override used for the run and do not compare
it against full-resolution numbers without saying so.

---

## References

- `/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/cosmos/cosmos_predict2_2b_grpo.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/cosmos/train.py`
- `/home/mingfeiguo/Desktop/wm-infra/configs/experiment/cosmos_predict2_2b_grpo.yaml`
- `/home/mingfeiguo/Desktop/wm-infra/configs/model/cosmos/predict2_2b.yaml`
- `/home/mingfeiguo/Desktop/wm-infra/configs/sampling/cosmos_v2w_704p_93f.yaml`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/models/families/cosmos/predict2_policy.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/models/families/cosmos/executor.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/rollouts/collectors/cosmos_predict2.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/rollouts/evaluators/diffusion/flow_matching.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/algorithms/diffusion/sde.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/rewards/multi.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/rewards/aesthetic.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/models/diffusion.py`
- `/home/mingfeiguo/Desktop/wm-infra/tests/engine/generation/test_pipeline_cosmos.py`
- `/home/mingfeiguo/Desktop/wm-infra/tests/engine/generation/test_cosmos_executor_parity.py`
- `/home/mingfeiguo/Desktop/wm-infra/tests/rollouts/test_evaluators.py`
- `/home/mingfeiguo/Desktop/wm-infra/tests/config/test_load_all_experiments.py`
- `/home/mingfeiguo/Desktop/wm-infra/datasets/drawbench/test.txt`
- `/home/mingfeiguo/Desktop/wm-infra/datasets/drawbench/train_192.txt`
- `/home/mingfeiguo/Desktop/wm-infra/datasets/drawbench/eval_64.txt`
