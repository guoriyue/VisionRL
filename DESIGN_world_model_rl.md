# DESIGN: World Model RL — Phase 1 Experiment (Cosmos Predict2-2B Video2World)

Status: DRAFT 2026-04-27
Owner: visual-rl
Scope: ONE concrete Phase-1 experiment. ≤ 3 engineer-days for the MVE.

---

## 1. Strategic intent

Per `PLAN_diffusion_backend.txt` §4.2 ("赌注 B：World Model / Video2World RL", lines
164–185), visual-rl already has Cosmos code on disk (`vrl/models/families/cosmos/`,
`vrl/scripts/cosmos/cosmos_predict2_2b_grpo.py`, three test files). The plan's
delivery bar is:

> "保证至少一个可复现的 Cosmos GRPO recipe — 明确 reward / conditioning / eval 的实验协议"

This doc commits to that single recipe. It does not propose new model training, new
collectors, or new reward functions. It picks one row from the cross product of
(reward × conditioning × eval) that we can actually defend in the thesis.

---

## 2. Experiment scope

**One sentence**: Run GRPO on Cosmos Predict2-2B-Video2World with a single
reference image as conditioning, optimizing the aesthetic reward over a held-out
prompt set, and demonstrate a paired LoRA-vs-base reward delta on the same seeds.

What this is **not**:
- Not a multi-reward sweep (no aesthetic+CLIP+pickscore Pareto).
- Not a model comparison (no Predict1 vs Predict2 vs Predict2.5).
- Not a new infra build (no new collector, no new evaluator, no new reward).
- Not a long-horizon world-model evaluation (no rollout-conditioned future
  prediction; see §8).

The minimum we have to prove is: **the existing
`cosmos_predict2_2b_grpo.py` pipeline produces a positive `lora - base` reward
delta on a held-out prompt set, on a fixed reference image, with paired seeds.**

---

## 3. Input modality + conditioning

### 3.1 Modality decision: image-conditioned video generation

Grounded in `vrl/models/families/cosmos/predict2.py`:

- The Predict2 family wrapper supports two variants (lines 15–21):
  `predict2_video2world` (2B / 14B) and `predict2_text2image` (0.6B / 2B / 14B).
- We commit to `CosmosVariant.PREDICT2_VIDEO2WORLD` at `model_size="2B"`,
  i.e. HF id `nvidia/Cosmos-Predict2-2B-Video2World` (line 16). This is the same
  variant the existing GRPO script defaults to (`cosmos_predict2_2b_grpo.py:36`)
  and the same one the e2e test exercises (`tests/e2e/test_serving_cosmos.py:43`).
- The `text2image` 0.6B variant is cheaper but is not a world model — it would
  be off-thesis for §4.2. Rejected.

### 3.2 Conditioning shape (concrete)

From `predict2.py::encode_conditioning` (lines 201–232):

- `request.references` is a `list[str]` of file paths (`vrl/models/base.py:21`).
- For Video2World, **exactly one** reference image is required:
  - `predict2.py:218` raises `ValueError` if `request.references` is empty.
  - `predict2.py:219` reads only `request.references[0]`.
- The image is loaded as RGB via PIL and put into `state["reference_image"]`.
- In `denoise_init` (lines 405–416): if a reference image is present, it is
  preprocessed to `(B, 3, 1, H, W)` via `pipeline.video_processor.preprocess_video`
  and passed to `pipeline.prepare_latents(video=...)`. If absent, a zero tensor is
  used — the script logs a warning that this is "degenerate" (`cosmos_predict2_2b_grpo.py:312–317`).

**Decision**: every prompt in this experiment uses **the same single fixed
reference image**. We do not vary reference image across prompts in Phase 1.
Rationale:

1. The existing collector takes one `reference_image` at construction time
   (`cosmos_predict2.py:62`). Multi-image conditioning would require collector
   surgery — out of scope.
2. With a fixed image, the only thing the policy can change is *prompt-conditional
   appearance and motion* of the generated video. That is the cleanest single
   variable for measuring whether GRPO is doing anything.
3. We avoid the confound of "did rewards go up because of the prompt or because
   of the image".

### 3.3 Generation tensor shape

From `cosmos_predict2_2b_grpo.py:39–44` and `cosmos_predict2.py:23–31`:

- Width × Height = **1280 × 704**.
- `num_frames = 93` (the Predict2 default; "81 generated + 12 conditioning"
  per `cosmos_predict2.py:29`).
- `num_steps = 35`, `guidance_scale = 7.0`, `fps = 16`.

These defaults are baked into both the executor and the collector. We commit to
them — changing them is also out of scope for Phase 1.

---

## 4. Reward: aesthetic (single)

### 4.1 Decision

**Reward**: `AestheticReward` from `vrl/rewards/aesthetic.py`, registered
as `"aesthetic"` in `vrl/rewards/multi.py:38`. Single reward, weight = 1.0.

### 4.2 Why aesthetic and not the alternatives

Looking at the four registered rewards in `vrl/rewards/multi.py:31–41`:

| Reward      | Status for Cosmos Video2World           | Verdict |
|-------------|------------------------------------------|---------|
| `aesthetic` | Handles `[C, T, H, W]` video by sampling 3 frames at 25/50/75% (`aesthetic.py:96–97`). **Already video-aware.** | **Pick.** |
| `clipscore` | Handles 5-D video tensors but only by taking the **middle frame** (`clip.py:78–79`). Discards temporal information; reduces to per-frame text-image alignment. Acceptable but weaker signal for video. | Skip. |
| `ocr`       | Designed for text-in-image (drawbench OCR). Cosmos Video2World does not optimize toward textual content; the dataset of choice has no OCR ground truth. | Skip. |
| `pickscore` | Image-pair preference scorer ported from flow_grpo. Also middle-frame at best. Aesthetic is a strict subset of what pickscore proxies; aesthetic is simpler to debug. | Skip. |

So the **only video-frame-aware** reward in the existing registry is `aesthetic`.
Picking anything else means accepting a per-frame collapse of a 93-frame video
into a single image, which actively destroys the world-model signal we are
nominally optimizing.

### 4.3 What the reward signal actually means

`AestheticReward` is a CLIP-ViT-L/14 image embedder + a 5-layer MLP regressor
trained on SAC+LOGOs+AVA1 (`aesthetic.py:62–66`). It returns a scalar per image,
typically in `[~3.5, ~7.5]`. We average over 3 sampled frames per video
(`aesthetic.py:94–98`).

This is **visual fidelity / aesthetic quality of generated frames**, not:
- temporal consistency (frames are scored independently),
- trajectory match to the reference (no comparison to the reference image),
- physical plausibility (no dynamics-aware critic).

We are honest about this: §4.2 is "Video2World RL", but Phase 1's reward is a
per-frame aesthetic critic. The justification is the closure constraint —
this is the only reward in the registry that even reads more than one frame.
Anything stronger (temporal consistency, action-condition match) would require
a new reward function, which is explicitly out of scope (see §7).

---

## 5. Eval protocol

### 5.1 Held-out prompt set

- **Source**: `datasets/drawbench/test.txt` (999 prompts, exists in repo).
  Train file is empty (`drawbench/train.txt` is 0 bytes — verified via `wc -l`),
  so we partition `test.txt` ourselves: first 64 lines → eval; lines 65–256 →
  train. Both are deterministic by line index.
- **Why drawbench**: Already in the repo, prompt distribution is general-purpose
  (no OCR/text bias), and 999 prompts is more than enough for a 64-prompt eval
  hold-out + a small training pool. `pickscore_sfw/train.txt` (15486 prompts) is
  also acceptable but is much larger than needed and overweighted toward
  photoreal portraits.

### 5.2 Metrics (concrete, not vibes)

Driven by `_run_eval_only` in `cosmos_predict2_2b_grpo.py:466–556`, which
already implements paired base-vs-LoRA evaluation. We use it as-is.

Primary metric:

- **Paired aesthetic delta**: `Δ = mean(lora_score) − mean(base_score)`,
  computed over `eval_prompts × eval_seeds`, using **the same seed** for both
  LoRA and base on every (prompt, seed) pair (script lines 498–537).

Secondary metrics (already logged):

- `metrics.csv` per epoch: `loss`, `policy_loss`, `kl_penalty`, `reward_mean`,
  `reward_std`, `clip_fraction`, `approx_kl`, `advantage_mean`
  (`cosmos_predict2_2b_grpo.py:343–346`).
- Saved middle-frame PNGs per checkpoint (`_save_middle_frame`, lines 445–463)
  for visual inspection of mode collapse.

Sanity gates (already wired, lines 365–381):

- Epoch-0 `clip_fraction > 0.5` → log-prob mismatch between collect and
  forward_step. Pipeline is broken; abort.
- Epoch-0 `approx_kl > 0.1` → same diagnosis.

Pass/fail criterion for Phase 1:

- **Pass**: Δ > 0 with at least 16 of 64 eval prompts strictly improved (i.e.
  > 25% improvement rate on a paired sign test), and no epoch-0 sanity gate
  trips. **Visual** check: middle-frame PNGs at the final checkpoint do not
  exhibit mode collapse (single dominant image regardless of prompt).
- **Fail**: Δ ≤ 0, or Δ > 0 driven by < 8 prompts (reward hacking on a
  narrow slice), or visible mode collapse.

### 5.3 What we explicitly do NOT measure

- FVD, FID, IS — not implemented in `vrl/rewards/`. Adding them violates the
  no-new-evaluator constraint.
- Temporal consistency (e.g. optical-flow smoothness) — same.
- Action-following — Cosmos Predict2 doesn't take action sequences in this
  pipeline (`VideoGenerationRequest.action_sequence` exists at `base.py:40`
  but is unused by `predict2.py`).

---

## 6. Reproducibility

### 6.1 Pinned identifiers

| Item              | Value                                                         |
|-------------------|---------------------------------------------------------------|
| Model checkpoint  | `nvidia/Cosmos-Predict2-2B-Video2World` (HF)                  |
| Dtype             | `bfloat16` for transformer + text encoder, `float32` for VAE  |
| Reference image   | `${HF_HOME}/.../models--Wan-AI--Wan2.2-I2V-A14B-Diffusers/snapshots/*/examples/i2v_input.JPG` (per `tests/e2e/test_serving_cosmos.py:21`) — already used by the e2e test, so we know it loads. If unavailable on the training box, fall back to any 1280×704 JPG in `outputs/` and document the path in `metrics.csv` via the `ref_image` flag (already plumbed at `cosmos_predict2_2b_grpo.py:348`). |
| Train prompts     | `datasets/drawbench/test.txt`, lines 65–256 (192 prompts)     |
| Eval prompts      | `datasets/drawbench/test.txt`, lines 1–64 (64 prompts)        |
| Seed              | `--seed 0` (`cosmos_predict2_2b_grpo.py:92`)                  |

### 6.2 Hyperparameters (commit to defaults from the existing script)

All values from `CosmosPred2Config` defaults (`cosmos_predict2_2b_grpo.py:32–101`):

```
width=1280  height=704  num_frames=93  num_steps=35  guidance_scale=7.0  fps=16
lr=1e-5  beta=0.004  clip_range=1e-3  adv_clip_max=5.0
group_size=4  prompts_per_step=1  num_inner_epochs=1
mixed_precision=bf16  gradient_checkpointing=True
use_lora=True  lora_rank=64  lora_alpha=32
target_modules=["to_k","to_q","to_v","to_out.0","add_k_proj","add_q_proj","add_v_proj","to_add_out"]
ema=True  ema_decay=0.9  ema_update_interval=8
kl_reward=0.0  sde_window_size=0  same_latent=False  global_std=False
```

We do **not** tune any of these for Phase 1. If the MVE fails the Phase-1 gate
in §5.2, the next action is **not** to sweep — it is to read the failure mode
(see §7 step 3).

### 6.3 Training budget

- Single H100 / A100-80G.
- 200 epochs × `prompts_per_step=1` × `group_size=4` = 800 generated videos.
- Per-video wall-clock at `num_steps=35`, 93 frames, 1280×704: ~30s on H100
  by `predict2.py` profiling print (line 268, `elapsed_s`). 800 × 30s ≈ 6.7 h.
  Add the gradient pass (~2× for backward + optimizer) → ~13.5 h training.
- Checkpoint every 100 epochs (`save_interval=100`) → 2 checkpoints + final
  eval. Eval is 64 prompts × 1 seed × 2 (lora + base) × 30s ≈ 64 min per
  checkpoint.

**Total compute**: ≤ 16 GPU-hours per run on a single H100.

---

## 7. Phase 1 deliverable: 3-day MVE

**Goal**: validate the pipeline end-to-end before committing the full 16-h run.

### Day 1 — Smoke test (≤ 4 GPU-hours)

1. `pytest tests/models/test_cosmos_predict2_step.py
   tests/rollouts/test_cosmos_predict2_collector.py` — confirm step + collector
   tests still pass on current branch. (Already exist; no new test code.)
2. Run the existing script with a **dummy 8-epoch budget**:
   ```
   python -m vrl.scripts.cosmos.cosmos_predict2_2b_grpo \
     --reference-image <path> \
     --prompt-file datasets/drawbench/test.txt \
     --num-epochs 8 --save-interval 4 --log-interval 1 \
     --output-dir outputs/cosmos_pred2_2b_mve_day1 \
     --debug-first-step
   ```
   Note: this uses *all* 999 drawbench prompts because the script does not
   support line-range slicing. For day-1 smoke that is fine.
3. **Pass criteria**: epoch-0 `clip_fraction < 0.5`, `approx_kl < 0.1`, no
   OOM, two checkpoints saved, eval PNGs visually non-empty.
4. **Failure modes**:
   - If sanity gates trip → log-prob mismatch in `_predict_noise_impl`.
     Diff `cosmos_predict2.py::predict_noise` against
     diffusers `Cosmos2VideoToWorldPipeline.__call__`.
   - If OOM → drop `num_frames` to 33 (still > 1, so V2W path stays valid)
     and rerun. Document the change in §6.

### Day 2 — Real training run with proper splits

1. Add a 5-line slicing helper inline in the script (or pre-split into two
   files: `datasets/drawbench/train_192.txt`, `datasets/drawbench/eval_64.txt`).
   This is the ONLY change to non-config code in Phase 1.
2. Run 200 epochs:
   ```
   python -m vrl.scripts.cosmos.cosmos_predict2_2b_grpo \
     --reference-image <path> \
     --prompt-file datasets/drawbench/train_192.txt \
     --eval-prompts datasets/drawbench/eval_64.txt \
     --num-epochs 200 --save-interval 100 \
     --output-dir outputs/cosmos_pred2_2b_mve_day2
   ```
3. Tail `metrics.csv` and watch `reward_mean` trend. If after 50 epochs
   `reward_mean` is below the epoch-0 mean, abort — it's not learning.

### Day 3 — Eval-only paired comparison + writeup

1. Run paired eval at the final checkpoint (script already supports this):
   ```
   python -m vrl.scripts.cosmos.cosmos_predict2_2b_grpo \
     --eval-only \
     --lora-path outputs/cosmos_pred2_2b_mve_day2/checkpoint-200/lora_weights \
     --eval-prompts datasets/drawbench/eval_64.txt \
     --reference-image <path> --eval-seeds 1
   ```
   Output: `outputs/cosmos_pred2_2b_mve_day2/eval_only/eval_results.csv` with
   per-prompt `(lora_score, base_score, delta)`.
2. Compute Δ and the 25%-prompt sign-test gate (§5.2).
3. Write up the result as a single section in the visual-rl thesis chapter:
   "Cosmos Predict2-2B Video2World GRPO with aesthetic reward on drawbench
   eval-64; Δ = X.XX, prompts improved = Y/64."

**Rollback rule**: if Day 1 fails, do not start Day 2. Spend Day 2 fixing the
log-prob path and re-run Day 1. Phase 1 may slip to Day 4 — but we never
launch a 13-h run on a known-broken pipeline.

---

## 8. Open questions

These are facts I could not verify from code alone. None block Day 1.

1. **BLOCKER for "world model" framing**: the existing pipeline as wired in
   `cosmos_predict2_2b_grpo.py` does **not** consume action sequences.
   `VideoGenerationRequest.action_sequence` exists (`vrl/models/base.py:40`)
   but is never read by `predict2.py`. So "Video2World RL" here means
   "image-conditioned video generation RL with visual reward". If the thesis
   needs **action-conditioned** world-model rollouts, that is a Phase-2 build
   (new collector + action embedding + reward over action-following), not a
   Phase-1 tuning task. Need to confirm thesis framing with advisor before
   we ship results that overclaim.
2. **Scheduler return shape**: `cosmos_predict2.py:171–178` calls
   `sde_step_with_logprob` with `return_dt=cfg.kl_reward > 0`, then
   `sde_result.log_prob` is used unconditionally (line 184). If
   `sde_step_with_logprob` returns a different shape when `return_dt=True`,
   stacking at line 200 could silently broadcast. Day-1 smoke would catch this
   at epoch-0 sanity gate, but worth eyeballing
   `vrl/rollouts/evaluators/diffusion/flow_matching.py::sde_step_with_logprob`
   before launch.
3. **Reference-image availability**: `tests/e2e/test_serving_cosmos.py:21–24`
   pulls the reference from a Wan2.2 HF snapshot. If that snapshot isn't
   cached on the training box, Day 1 needs a fallback image. Mitigation
   already in §6.1.
4. **Eval cost at 1280×704**: 64 prompts × 30s × 2 (lora+base) ≈ 64 min, fine
   per checkpoint — but if `decode_vae` step OOMs at this resolution under
   `enable_sequential_cpu_offload=True` for the eval path
   (`predict2.py:133–134`), we may need `enable_vae_tiling` (already on per
   `predict2.py:135–136`). Verified-on-paper, not on hardware.

---

## 9. Sanity-check / 闭环交付

Files read end-to-end while writing this doc, with the load-bearing line ranges:

- `/home/mingfeiguo/Desktop/wm-infra/PLAN_diffusion_backend.txt` §4.2
  (lines 164–185) — strategic mandate.
- `/home/mingfeiguo/Desktop/wm-infra/vrl/models/families/cosmos/predict2.py`
  - lines 15–21: `_MODEL_ID_MAP` confirms Predict2 supports 2B Video2World
    and Text2Image only (no Predict2 Text2World variant).
  - lines 201–232: `encode_conditioning` confirms `request.references[0]` is
    the only reference consumed; raises if empty.
  - lines 372–479: `denoise_init` confirms reference image → `prepare_latents`
    flow, with zero-tensor fallback at lines 412–416.
  - lines 481–608: `predict_noise` / `_predict_noise_impl` — confirms the
    forward shape and CFG logic.
- `/home/mingfeiguo/Desktop/wm-infra/vrl/rollouts/collectors/cosmos_predict2.py`
  - lines 21–45: `CosmosPredict2CollectorConfig` defaults (1280×704, 93 frames,
    35 steps, 7.0 CFG).
  - lines 56–67: collector takes a single `reference_image`.
  - lines 78–267: `collect()` flow — encode_text → denoise_init → SDE loop with
    `predict_noise` → `decode_vae` → reward → `ExperienceBatch`.
- `/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/cosmos/cosmos_predict2_2b_grpo.py`
  - lines 32–101: `CosmosPred2Config` — all hyperparameters cited in §6.2.
  - lines 209–212: confirms reward is built via `get_reward(config.reward_type)`
    with `reward_type` defaulting to `"aesthetic"`.
  - lines 312–317: zero-conditioning warning — justifies §3.2's "fixed real
    reference image" decision.
  - lines 365–381: epoch-0 sanity gates — wired into §5.2.
  - lines 466–556: `_run_eval_only` — paired LoRA-vs-base comparison,
    same-seed protocol — wired into §5.2.
- `/home/mingfeiguo/Desktop/wm-infra/vrl/rewards/multi.py` lines 31–41 — only
  four registered rewards exist: aesthetic, clipscore, ocr, pickscore.
- `/home/mingfeiguo/Desktop/wm-infra/vrl/rewards/aesthetic.py` lines 81–110 —
  confirms 5-D video tensor support via 3-frame sampling at 25/50/75%.
- `/home/mingfeiguo/Desktop/wm-infra/vrl/rewards/clip.py` lines 76–82 — confirms
  CLIPScoreReward middle-frame collapse for 5-D video, justifying §4.2 rejection.
- `/home/mingfeiguo/Desktop/wm-infra/vrl/models/base.py` lines 16–43 —
  `VideoGenerationRequest.references: list[str]`, `action_sequence` exists
  but unused by Predict2 — basis for §8 question 1.
- `/home/mingfeiguo/Desktop/wm-infra/vrl/models/families/cosmos/variants.py`
  lines 12–19 — `CosmosVariant` enum, confirms only PREDICT2_VIDEO2WORLD and
  PREDICT2_TEXT2IMAGE exist for the 2B size.
- `/home/mingfeiguo/Desktop/wm-infra/tests/models/test_cosmos_predict2_step.py`
  lines 13–113 — confirms `denoise_init` returns `DenoiseLoopState` with the
  fields §3.2 relies on.
- `/home/mingfeiguo/Desktop/wm-infra/tests/rollouts/test_cosmos_predict2_collector.py`
  lines 12–58 — confirms collector defaults match §3.3.
- `/home/mingfeiguo/Desktop/wm-infra/tests/e2e/test_serving_cosmos.py`
  lines 17–26 and 71–82 — basis for the reference-image path in §6.1.
- `datasets/drawbench/test.txt` (999 prompts — `wc -l`), `datasets/drawbench/train.txt`
  (0 prompts) — basis for the §5.1 split decision.
- `configs/experiment/wan_2_1_1_3b_grpo.yaml` — read for config-file
  conventions; not used directly because Cosmos has no experiment YAML yet
  (`configs/model/cosmos/` is empty). Phase 1 stays CLI-driven via the script.

What I did NOT verify in code (and therefore flagged in §8):
- Concrete return-shape of `sde_step_with_logprob` (Q2).
- Whether `enable_vae_tiling=True` survives a 1280×704 eval pass on H100 (Q4).
- That the Wan2.2 reference image actually exists on the user's HF cache (Q3).
