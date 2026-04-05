# Wan2.2 Baseline on RTX 5090 32GB

This document records the first verified local baseline for running Wan2.2 on Nico's workstation.

## Goal

Establish a real, machine-local baseline for video generation before attempting any apples-to-apples comparison against vLLM-Omni or other serving stacks.

## Hardware / environment

- GPU: NVIDIA GeForce RTX 5090 32GB
- Host driver: 580.126.09
- Official Wan2.2 repo: `~/Desktop/Wan2.2`
- Model dir: `~/Desktop/Wan2.2/Wan2.2-T2V-A14B`
- Python env used: `kosen`

## Why this baseline matters

Before comparing serving systems, we need to answer a simpler question:

> Can Wan2.2 actually run on this machine at all?

Answer: **yes**, but only with aggressive memory-saving settings.

## Verified working command

```bash
source /home/mingfeiguo/miniconda3/etc/profile.d/conda.sh
conda activate kosen
cd /home/mingfeiguo/Desktop/Wan2.2
python generate.py \
  --task t2v-A14B \
  --size 832*480 \
  --frame_num 9 \
  --ckpt_dir /home/mingfeiguo/Desktop/Wan2.2/Wan2.2-T2V-A14B \
  --offload_model True \
  --convert_model_dtype \
  --t5_cpu \
  --sample_steps 4 \
  --sample_shift 12.0 \
  --sample_guide_scale 4.0 \
  --prompt 'a short test video of a dog walking' \
  --save_file /tmp/wan22_official_smoke.mp4
```

## Result

Status: **success**

Output file:
- `/tmp/wan22_official_smoke.mp4`

Observed behavior:
- T5 loaded successfully
- VAE loaded successfully
- Wan model loaded successfully
- Generation completed successfully
- Output video was saved

## Runtime observations

Generation progress log:
- step 1/4: ~29.56s
- step 2/4: progress reached 50% at ~31s total
- step 3/4: progress reached 75% at ~33s total
- full 4-step run: ~1m53s total

Approximate interpretation:
- 4 denoising steps at 832x480 and 9 frames already cost nearly 2 minutes
- this confirms the model is usable on 32GB VRAM, but not cheap
- larger frame counts / step counts will grow cost quickly

## Required memory-saving settings

The following settings were important for success:
- `--offload_model True`
- `--convert_model_dtype`
- `--t5_cpu`

Without these, a 32GB card is unlikely to be a reliable path for this model.

## Problems encountered before success

The official Wan2.2 repo imports more modules than needed for pure T2V, so additional dependencies had to be installed into the `kosen` environment before the smoke test could start:
- `easydict`
- `ftfy`
- `imageio[ffmpeg]`
- `dashscope`
- `decord`
- `librosa`

This is not a serving issue; it is a repo/runtime hygiene issue in the upstream inference entrypoint.

## Implication for wm-infra

This baseline gives us a real reference point for future serving work:

- Official baseline exists and is reproducible
- Wan2.2 is feasible on this machine in reduced settings
- Any serving-layer comparison must account for the fact that model execution itself is already expensive

## Recommended next profiling matrix

Run these systematically next:

### Frame count sweep
- 9 frames
- 17 frames
- 33 frames

### Step sweep
- 4 steps
- 8 steps
- 12 steps

### Resolution sweep
- 832x480
- smallest supported 480p-ish setting that preserves correct model assumptions
- optionally one higher-cost setting if memory allows

### Metrics to record
- end-to-end wall time
- per-step wall time
- peak VRAM (`nvidia-smi` sampled externally)
- whether run succeeded / failed
- output artifact path
- config flags used

## Comparison note: vLLM-Omni

At the time of this baseline:
- `vllm-omni` support for Wan2.2 expects a diffusers-style model layout
- the local official Wan2.2 checkpoint directory is **not** in the format that `vllm-omni` expects
- therefore the official repo was the fastest path to a real baseline

This means:
- official Wan2.2 baseline is now established
- `vllm-omni` comparison should happen later using a compatible Wan2.2 diffusers model layout

## Integration into wm-infra

This profiling data is now reflected in the implemented temporal control plane:

- **`WanTaskConfig`** (`wm_infra/controlplane/schemas.py`) carries first-class Wan execution fields such as
  `frame_count`, `width`, `height`, `num_steps`, `guidance_scale`, `shift`, `offload_model`,
  `convert_model_dtype`, `t5_cpu`, and `memory_profile`.
- **`RolloutTaskConfig`** also carries video-relevant execution fields for rollout-style jobs so scheduling
  and metadata do not depend on opaque ad hoc blobs.
- **Legacy metadata backfill** still exists for compatibility, but new callers should use first-class config objects.
- **Resource estimation** (`wm_infra/controlplane/resource_estimator.py`) provides a lightweight scheduler-facing
  estimate keyed to frame count, steps, and resolution.
- **Scheduler policy** includes a coarse `memory_aware` mode so smaller temporal jobs are less likely to be trapped
  behind obviously heavier ones.

## Serving design implications now captured in wm-infra

Based on this baseline, `wm-infra` treats the following as first-class execution inputs instead of ad hoc metadata:
- `wan_config.frame_count`
- `wan_config.width` / `wan_config.height`
- `wan_config.num_steps`
- `wan_config.offload_model`
- `wan_config.convert_model_dtype`
- `wan_config.t5_cpu`
- `wan_config.memory_profile`

Scheduler impact:
- a lightweight frame-aware resource estimate exists so video jobs can be ordered with some awareness of frame-driven memory pressure
- this is intentionally a coarse policy scaffold, not a claim of exact VRAM prediction
