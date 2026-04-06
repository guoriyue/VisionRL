# Genie ECS Runtime Profile

This document records the Step 3 runtime/profile convergence for `genie-rollout`.

## Why Step 3 existed

Before this step, `execute_job(...)` and `execute_job_batch(...)` both emitted runtime metadata, but they did not use the same profiling contract.

The drift was visible in three places:

- scheduler metadata had different keys and different semantics across the single-request and batched paths
- stage timing existed, but there was no stable per-stage summary payload
- benchmark-relevant scheduling signals were spread across ad-hoc runtime fields

That made it harder to reason about scheduling behavior directly from persisted runtime artifacts.

## Runtime/Profile Contract

Both the single-request path and the batched path now emit the same profile surfaces in:

- sample runtime payload
- rollout metadata

### Stage fields

The runtime now includes:

- `stage_graph`
- `stage_timings_ms`
- `stage_history`
- `stage_profile`

`stage_profile` is the stable summary surface. For each stage in the Genie graph it records:

- `count`
- `elapsed_ms`
- `queue_lanes`
- `runner_modes`
- `chunk_count`
- `max_chunk_size`
- `frame_ranges`

It also records:

- `completed_stages`
- `stage_count`
- `total_elapsed_ms`

### Scheduler fields

The runtime scheduler payload is now structurally aligned across both execution paths.

Common fields:

- `execution_path`
- `transition_entities`
- `chunk_count`
- `chunks`
- `queue_lanes`
- `scheduler_inputs`
- `observed_batch_sizes`
- `batched_across_requests`
- `max_chunk_size`
- `max_observed_batch_size`
- `avg_expected_occupancy`
- `cross_request_batcher`

Compatibility rule:

- single-request async execution uses `execution_path="transition_batcher"`
- queue-batched execution uses `execution_path="runner_window_batch"`
- the schema is the same; only the execution path and `cross_request_batcher` source differ

### Runtime-state fields

`runtime_state` now includes the hot execution signals that matter for scheduler interpretation:

- `resident_tier`
- `materialized_bytes`
- `reuse_hits`
- `reuse_misses`
- `last_completed_frame`
- `checkpoint_delta_ref`
- `dirty_since_checkpoint`
- `source_cache_key`

### Benchmark-facing fields

The runtime now emits `benchmark_profile` so benchmark and regression tooling can read the key stage/runtime signals without reparsing stage history.

Fields:

- `state_token_prep_ms`
- `transition_ms`
- `checkpoint_ms`
- `artifact_persist_ms`
- `controlplane_commit_ms`
- `total_elapsed_ms`
- `chunk_count`
- `max_chunk_size`
- `max_observed_batch_size`
- `avg_expected_occupancy`
- `batched_across_requests`

## Step 3 Benchmark Snapshot

| Artifact | Submit Mean (ms) | Submit P95 (ms) | Terminal Mean (ms) | Terminal P95 (ms) | Success |
| --- | ---: | ---: | ---: | ---: | ---: |
| `genie_default_baseline.json` | 2560.384 | 3749.167 | 2564.645 | 3753.584 | 1.0 |
| `genie_default_batched.json` | 6.002 | 6.621 | 2244.601 | 3160.030 | 1.0 |
| `genie_profile_baseline.json` | 8601.106 | 8601.230 | 8610.152 | 8610.302 | 1.0 |
| `genie_profile_batched.json` | 1668.998 | 1669.166 | 5817.897 | 8138.055 | 1.0 |
| `genie_heavy_off.json` | 4436.842 | 5610.114 | 4441.595 | 5614.879 | 1.0 |
| `genie_heavy_on.json` | 1191.071 | 2365.676 | 4492.411 | 5749.016 | 1.0 |

Ratios after Step 3:

- default terminal mean ratio: `0.8752x`
- default terminal p95 ratio: `0.8419x`
- profile terminal mean ratio: `0.6757x`
- profile terminal p95 ratio: `0.9452x`
- heavy terminal mean ratio: `1.0114x`
- heavy terminal p95 ratio: `1.0239x`

## Step 3 Conclusion

The stage runtime now exposes a stable profile contract:

- stage-local summaries are explicit instead of inferred from raw history
- scheduler metadata is structurally aligned across single and batched execution
- benchmark-relevant runtime fields are first-class and persisted

This is the runtime/profile baseline needed before Step 4 turns the cleanup gates into stronger automated regression protection.
