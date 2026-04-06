# Genie ECS Baseline Snapshot

This document records the Step 1 baseline for the `genie-rollout` ECS execution plan.

Scope constraints:

- only `genie-rollout`
- in-process benchmark harness
- `execution_mode=chunked`
- current GPU: NVIDIA GeForce RTX 5090

## Benchmark Commands

```bash
python benchmarks/bench_samples_api.py \
  --in-process \
  --workload genie \
  --device cuda \
  --execution-mode chunked \
  --iterations 8 \
  --concurrency 4 \
  --timeout-s 30 \
  --genie-max-concurrent-jobs 1 \
  --genie-cross-request-batching off \
  --genie-transition-batch-wait-ms 0 \
  --genie-transition-max-batch-size 1 \
  --persist-root benchmarks/results/genie_default_baseline_workspace \
  --output benchmarks/results/genie_default_baseline.json

python benchmarks/bench_samples_api.py \
  --in-process \
  --workload genie \
  --device cuda \
  --execution-mode chunked \
  --iterations 8 \
  --concurrency 4 \
  --timeout-s 30 \
  --genie-max-concurrent-jobs 1 \
  --genie-cross-request-batching on \
  --genie-transition-batch-wait-ms 2 \
  --genie-transition-max-batch-size 4 \
  --persist-root benchmarks/results/genie_default_batched_workspace \
  --baseline-file benchmarks/results/genie_default_baseline.json \
  --output benchmarks/results/genie_default_batched.json

python benchmarks/bench_samples_api.py \
  --in-process \
  --workload genie \
  --payload-file benchmarks/payloads/genie_profile.json \
  --device cuda \
  --execution-mode chunked \
  --iterations 8 \
  --concurrency 8 \
  --timeout-s 30 \
  --genie-max-concurrent-jobs 1 \
  --genie-cross-request-batching off \
  --genie-transition-batch-wait-ms 0 \
  --genie-transition-max-batch-size 1 \
  --persist-root benchmarks/results/genie_profile_baseline_workspace \
  --output benchmarks/results/genie_profile_baseline.json

python benchmarks/bench_samples_api.py \
  --in-process \
  --workload genie \
  --payload-file benchmarks/payloads/genie_profile.json \
  --device cuda \
  --execution-mode chunked \
  --iterations 8 \
  --concurrency 8 \
  --timeout-s 30 \
  --genie-max-concurrent-jobs 1 \
  --genie-cross-request-batching on \
  --genie-transition-batch-wait-ms 2 \
  --genie-transition-max-batch-size 8 \
  --persist-root benchmarks/results/genie_profile_batched_workspace \
  --baseline-file benchmarks/results/genie_profile_baseline.json \
  --output benchmarks/results/genie_profile_batched.json

python benchmarks/bench_samples_api.py \
  --in-process \
  --workload genie \
  --payload-file benchmarks/payloads/genie_heavy.json \
  --device cuda \
  --execution-mode chunked \
  --iterations 8 \
  --concurrency 4 \
  --timeout-s 180 \
  --genie-max-concurrent-jobs 4 \
  --genie-cross-request-batching off \
  --genie-transition-batch-wait-ms 0 \
  --genie-transition-max-batch-size 1 \
  --persist-root benchmarks/results/genie_heavy_off_run \
  --output benchmarks/results/genie_heavy_off.json

python benchmarks/bench_samples_api.py \
  --in-process \
  --workload genie \
  --payload-file benchmarks/payloads/genie_heavy.json \
  --device cuda \
  --execution-mode chunked \
  --iterations 8 \
  --concurrency 4 \
  --timeout-s 180 \
  --genie-max-concurrent-jobs 4 \
  --genie-cross-request-batching on \
  --genie-transition-batch-wait-ms 2 \
  --genie-transition-max-batch-size 8 \
  --persist-root benchmarks/results/genie_heavy_on_run \
  --baseline-file benchmarks/results/genie_heavy_off.json \
  --output benchmarks/results/genie_heavy_on.json
```

## Baseline Metrics

| Artifact | Submit Mean (ms) | Submit P95 (ms) | Terminal Mean (ms) | Terminal P95 (ms) | Success |
| --- | ---: | ---: | ---: | ---: | ---: |
| `genie_default_baseline.json` | 1105.245 | 1837.430 | 2382.765 | 3830.279 | 1.0 |
| `genie_default_batched.json` | 707.580 | 1408.499 | 2427.505 | 3444.319 | 1.0 |
| `genie_profile_baseline.json` | 2035.567 | 2035.729 | 6694.124 | 9207.270 | 1.0 |
| `genie_profile_batched.json` | 11.392 | 11.530 | 10066.318 | 10066.389 | 1.0 |
| `genie_heavy_off.json` | 841.121 | 1667.731 | 4114.002 | 5043.300 | 1.0 |
| `genie_heavy_on.json` | 5.733 | 5.846 | 4446.339 | 5709.773 | 1.0 |

## Gate Assessment

Default workload gate:

- terminal mean ratio: `1.0188x`
- terminal p95 ratio: `0.8992x`
- verdict: pass

Heavy workload gate:

- terminal mean ratio: `1.0808x`
- terminal p95 ratio: `1.1322x`
- verdict: fail

Profile workload snapshot:

- terminal mean ratio: `1.5038x`
- terminal p95 ratio: `1.0933x`
- verdict: regression

## Step 1 Conclusion

Step 1 confirms that the current `genie-rollout` chunked runtime has a mixed batching story:

- default workload batching is acceptable and keeps the default benchmark gate green
- heavy workload batching regresses end-to-end latency and currently fails the cleanup gate threshold
- profile workload batching shows a larger tail-latency regression than heavy and needs runtime/policy investigation before cleanup is safe

The next step is to tighten cross-request batching policy and remove the heavy workload regression without regressing the default workload.
