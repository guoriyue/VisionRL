# Genie ECS Gates

This document records the Step 4 executable cleanup gates for `genie-rollout`.

## What is gated

Legacy Genie cleanup is only allowed when all four conditions stay green:

1. semantic equivalence between single-request and batched execution
2. default Genie workload benchmark gate
3. heavy Genie workload benchmark gate
4. sample manifest, temporal lineage, and artifact visibility retention

## Executable checks

### Benchmark gate

The benchmark gate now runs from committed benchmark artifacts through repository tests.

Primary check:

```bash
pytest tests/test_benchmarking.py
```

Important assertions:

- `tests/test_benchmarking.py::test_committed_genie_cleanup_gate_artifacts_pass`
- `tests/test_benchmarking.py::test_benchmark_gate_report_enforces_thresholds`

The gate logic lives in:

- `wm_infra/benchmarking.py`

Thresholds:

- success rate must stay `1.0`
- default workload terminal mean must stay `<= 1.05x`
- default workload terminal p95 must stay `<= 1.10x`
- heavy workload terminal mean must stay `<= 1.05x`
- heavy workload terminal p95 must stay `<= 1.10x`

### Semantic-equivalence gate

Primary check:

```bash
pytest tests/test_genie.py -k semantic
```

Important assertion:

- `tests/test_genie.py::test_single_and_batched_execution_preserve_semantics`

This test keeps the following surfaces aligned across single and batched execution:

- stage graph
- benchmark profile keys
- checkpoint delta semantics
- temporal refs
- output artifact surface

### Manifest / lineage / artifact retention gate

Primary checks:

```bash
pytest tests/test_server.py
pytest tests/test_genie.py
```

Coverage mapping:

- sample manifest retention:
  - `tests/test_server.py::test_create_and_get_sample_manifest`
- temporal lineage retention:
  - `tests/test_server.py::test_create_temporal_entities_and_genie_rollout`
  - `tests/test_genie.py::test_single_and_batched_execution_preserve_semantics`
- artifact visibility retention:
  - `tests/test_server.py::test_list_artifacts`
  - `tests/test_server.py::test_get_artifact_metadata`
  - `tests/test_server.py::test_get_artifact_content`
  - `tests/test_server.py::test_genie_rollout_artifact_content_downloadable`

## Full Step 4 Command

```bash
pytest tests/test_benchmarking.py \
  tests/test_bench_samples_api.py \
  tests/test_genie.py \
  tests/test_genie_batcher.py \
  tests/test_genie_runtime.py \
  tests/test_server.py
```

## Current Gate Snapshot

Current committed benchmark verdict:

- default terminal mean ratio: `0.9685x`
- default terminal p95 ratio: `0.9848x`
- heavy terminal mean ratio: `0.9492x`
- heavy terminal p95 ratio: `0.9309x`
- default gate: pass
- heavy gate: pass

## How to interpret a failure

- If the benchmark gate fails, do not start cleanup. Fix batching/runtime behavior or refresh the benchmark baseline only when the workload definition itself changed.
- If semantic equivalence fails, treat that as a behavior regression even if benchmark latency improves.
- If manifest, lineage, or artifact tests fail, cleanup is blocked because the control-plane contract regressed.

## Step 4 Conclusion

The cleanup gate is now executable inside the repository:

- benchmark thresholds are machine-checked
- semantic equivalence is machine-checked
- manifest, lineage, and artifact retention remain under test

This is the minimum guardrail set required before deleting legacy Genie code.
