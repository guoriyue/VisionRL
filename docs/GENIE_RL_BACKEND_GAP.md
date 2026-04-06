# Genie RL Backend Contract

`genie-rollout` now has an experimental trainer-facing RL contract, but it is
not a full-fidelity task backend yet.

The current code path is a persisted sample-production backend centered on prompt
tokens, frame windows, checkpoints, and artifact lineage. That is useful, but it
is not the same thing as an action-conditioned `WorldModel`.

## What Genie Already Has

These capabilities are already present and should be reused instead of rebuilt:

- prepared prompt/token state in `GenieRunner.prepare_inputs()`
- bounded transition windows in `GenieRunner.run_window()` and `run_window_batch()`
- persisted token/state outputs in `GenieRunner.persist_outputs()`
- chunk scheduling and stage profiling in `wm_infra/backends/genie.py`
- temporal lineage objects such as episodes, branches, state handles, checkpoints

Relevant files:

- `wm_infra/backends/genie.py`
- `wm_infra/backends/genie_runner.py`
- `wm_infra/backends/genie_runtime.py`
- `wm_infra/backends/genie_scheduler.py`

## What RL Needed

The RL env/session contract currently expects this shape:

```python
def predict_next(latent_state, action): ...
def rollout(input): ...
def get_initial_state(observation): ...
```

For a real RL backend, that also implies:

- `reset() -> observation, info`
- stable `state_handle -> observation` materialization
- reward hook
- `terminated` / `truncated` hook

The raw sample-production backend did not expose those semantics directly.

## What Is Implemented Now

The current repo now includes an explicit `GenieWorldModelAdapter` in
`wm_infra/rl/genie_adapter.py`.

It does three concrete things:

1. defines a stable RL state tensor backed by Genie token history
2. defines action semantics as explicit control over the latest prompt frame
3. uses `GenieRunner.prepare_inputs()` plus `run_window_batch()` to materialize
   `predict_next(latent_state, action)`

For correctness and reproducibility, the default adapter behavior is:

- CPU-only runtime: stub Genie contract
- CUDA runtime: attempt real Genie execution

The current experimental env exposed through `/v1/envs` is:

- `genie-token-grid-v0`

Its contract is:

- state: flattened token history with shape `[history_frames * H * W, 1]`
- observation: current token history concatenated with goal token history
- action space:
  - `stay`
  - `shift_left`
  - `shift_right`
  - `token_plus`
  - `token_minus`
- reward: dense token L1 distance against the goal frame
- termination: token distance below threshold
- truncation: horizon reached

## Remaining Limits

### 1. The action contract is explicit but still synthetic

The current control space is honest and stable, but it is still a synthetic
token-control task. It is not yet a semantic environment action space grounded
in an external task or embodied simulator.

### 2. Reward is attached cleanly, but task semantics are narrow

Reward, termination, and truncation now live in explicit RL code paths instead
of metadata blobs. The limitation is that the current reward only measures token
distance to a goal frame.

### 3. Runtime integration is still manager-led

`step_many` now uses `ExecutionChunk`, `BatchSignature`, and runtime profiling,
but RL batching still originates inside `RLEnvironmentManager`. It has not yet
been pushed deeper into a more general runtime entrypoint.

## Non-Goals

These approaches should be rejected:

- treating prompt text or `sample_spec.controls` as the RL action contract
- hiding reward/done in ad hoc metadata
- claiming Genie task semantics are complete just because token control now exists
- coupling trainer logic directly into `GenieRolloutBackend`

## Current Verdict

The old blocker is resolved at the contract level:

- `predict_next(latent_state, action)` now exists
- reset/step/step_many/fork/checkpoint work for a Genie-backed RL env session
- reward / terminated / truncated are explicit

What remains is not missing glue.
What remains is broadening this from an experimental token-control RL backend
into a richer, more task-grounded Genie workload.
