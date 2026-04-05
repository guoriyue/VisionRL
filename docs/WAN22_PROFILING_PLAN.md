# Wan2.2 Profiling Plan

This plan exists to turn a one-off smoke test into a reproducible benchmark set.

## Objective

Build a small but useful profile matrix for Wan2.2 on RTX 5090 32GB, then use it as a baseline for later serving comparisons.

## Baseline already verified

Working configuration:
- official repo
- `t2v-A14B`
- `832*480`
- `9` frames
- `4` steps
- `offload_model=True`
- `convert_model_dtype=True`
- `t5_cpu=True`

Observed wall time:
- ~113 seconds end-to-end

## Profiling matrix

### Axis 1: steps
- 4
- 8
- 12

### Axis 2: frame counts
- 9
- 17
- 33

### Axis 3: memory mode
- offload + t5_cpu + convert_model_dtype (required baseline)
- optional: test without one memory-saving option only if failure is acceptable

## Minimum matrix to run first

| frames | steps | expected purpose |
|---|---:|---|
| 9 | 4 | confirmed smoke test |
| 9 | 8 | isolate step scaling |
| 17 | 4 | isolate frame scaling |
| 17 | 8 | mid-cost practical test |
| 33 | 4 | more realistic short clip |

## Metrics

For each run, record:
- command
- resolution
- frames
- steps
- whether success/failure
- total wall time
- rough per-step time
- peak VRAM if available
- output path
- notes on failure mode

## What not to do yet

- do not jump to high-quality settings first
- do not compare against vLLM-Omni before Wan2.2 diffusers-format model path is available
- do not optimize serving before understanding raw model cost

## Deliverables for wm-infra

After this profiling round, add:
- a benchmark result table
- one reusable runner script
- one comparison note explaining what part is model cost vs serving cost
