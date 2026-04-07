# RL Environment Usage

`wm-infra` is still a temporal sample-production stack first. The new `wm_infra.rl`
module does not turn the repository into a full RL platform yet. What it does give
you today is both:

- a northbound environment-session API at `/v1/envs`
- a local trainer-facing experiment surface on top of the same contract

This is the right current split:

- `POST /v1/samples` remains the product-facing API for persisted sample production.
- `POST /v1/envs`, `/reset`, `/step`, `/step_many`, `/fork`, and `/checkpoint`
  are the trainer-facing control-plane surface for RL sessions.
- `wm_infra.rl` provides local experiment primitives that exercise the same env,
  trajectory, replay, and evaluation semantics without requiring an external trainer.

## What You Need To Learn For Real RL Integration

If you want to integrate a real RL stack, learn these topics in this order.

1. Environment contract
   - `reset() -> observation, info`
   - `step(action) -> observation, reward, terminated, truncated, info`
   - batched stepping for many environments at once

2. Reward and termination semantics
   - dense vs sparse rewards
   - success criteria
   - truncation because of horizon vs termination because the task is solved

3. Rollout collection
   - how to gather trajectories from many environments in parallel
   - what data a trainer actually needs: observation, action, reward, done, value

4. Policy/trainer separation
   - the world model or simulator should not own the policy update loop
   - the policy/trainer should treat the environment as a stable contract

5. Replay and dataset surfaces
   - trajectory storage
   - evaluation splits
   - reproducible task definitions
   - versioned environment catalogs

6. Runtime efficiency
   - vectorized stepping
   - state reuse
   - stage-local batching
   - asynchronous rollout workers when the trainer and world model are decoupled

These priorities line up with the public direction shown by platforms such as
OpenReward, and RL runtimes such as Slime and Miles. The important lesson is not
the branding. The important lesson is the contract shape: a trainer needs a clean
environment/session abstraction, batched stepping, clear reward semantics, and a
decoupled rollout path.

## Current RL Surface In This Repository

The current RL module contains:

- `GoalReward`: dense goal-reaching reward on latent states
- `GenieTokenReward`: dense goal-reaching reward on Genie token grids
- `WorldModelEnv`: single-environment adapter
- `WorldModelVectorEnv`: vectorized batched adapter
- `ToyLineWorldModel`: a deterministic 1D toy world model for stable examples
- `GenieWorldModelAdapter`: action-conditioned Genie token-history adapter
- `RLEnvironmentManager`: northbound environment/session runtime for `/v1/envs`
- `ExperimentSpec`: declarative local experiment configuration
- `SynchronousCollector`: batched rollout collector backed by env sessions
- `LocalActorCriticLearner`: minimal learner adapter
- `FixedTaskEvaluator`: fixed-task evaluator that writes `EvaluationRunRecord`
- `run_local_experiment()`: a runnable end-to-end local experiment entrypoint

Relevant files:

- `wm_infra/rl/env.py`
- `wm_infra/rl/toy.py`
- `wm_infra/rl/runtime.py`
- `wm_infra/rl/training.py`
- `wm_infra/rl/demo.py`
- `examples/rl_train_toy_world.py`
- `benchmarks/bench_rl_env.py`
- `docs/GENIE_RL_BACKEND_GAP.md`

## Northbound RL API

The trainer-facing HTTP surface now exists.

- `GET /v1/env-specs`
- `GET /v1/task-specs`
- `POST /v1/envs`
- `GET /v1/envs`
- `GET /v1/envs/{env_id}`
- `POST /v1/envs/{env_id}/reset`
- `POST /v1/envs/{env_id}/step`
- `POST /v1/envs/{env_id}/step_many`
- `POST /v1/envs/{env_id}/fork`
- `POST /v1/envs/{env_id}/checkpoint`
- `DELETE /v1/envs/{env_id}`
- `GET /v1/transitions`
- `GET /v1/trajectories`
- `GET /v1/evaluations`

The current toy env registry ships with:

- `toy-line-v0`
- `toy-line-train`
- `toy-line-eval`

The current experimental Genie RL registry ships with:

- `genie-token-grid-v0`
- `genie-token-train`
- `genie-token-eval`

This Genie env is intentionally narrow. It is an honest RL contract over Genie
token history, not a claim that Genie task semantics are complete. The action
space is explicit token control over the latest prompt frame, and reward is a
dense token-distance objective.

By default, the experimental Genie RL path uses stub mode on CPU-only machines.
That keeps the contract runnable and benchmarkable without pretending CPU has a
stable real Genie execution path. On CUDA-capable machines, the adapter can
attempt real Genie execution.

Every step response carries the stable RL fields the trainer actually needs:

- `observation`
- `reward`
- `terminated`
- `truncated`
- `info`
- `env_id`
- `episode_id`
- `task_id`
- `state_handle_id`
- `checkpoint_id`
- `policy_version`

The `step_many` route is not just a list of independent HTTP calls. It executes a
single batched `predict_next()` over all selected sessions and persists the resulting
`TransitionRecord` objects into the temporal control plane.

The runtime profile now also makes step semantics explicit:

- `batch_policy`
- `step_semantics`
- `northbound_reset_policy`
- `chunk_fill_ratios`

This matters because northbound env sessions keep explicit reset semantics even
when local collectors choose to auto-reset completed envs for throughput.

## Single-Environment Usage

```python
import numpy as np
import torch

from wm_infra.rl.env import GoalReward, WorldModelEnv
from wm_infra.rl.toy import ToyLineWorldModel


def initial_sampler(batch_size, device, dtype, generator):
    return torch.zeros(batch_size, 1, 1, device=device, dtype=dtype)


def goal_sampler(batch_size, device, dtype, generator):
    return torch.full((batch_size, 1, 1), 0.4, device=device, dtype=dtype)


env = WorldModelEnv(
    ToyLineWorldModel(),
    initial_state_sampler=initial_sampler,
    goal_state_sampler=goal_sampler,
    reward_fn=GoalReward(success_threshold=0.01, reward_scale=4.0),
    action_dim=3,
    max_episode_steps=5,
)

obs, info = env.reset(seed=3)
obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
```

The observation is the concatenation of current latent state and goal latent. For
real backends you would usually replace that with a backend-specific observation
structure, but the contract stays the same.

## Vectorized Usage

`WorldModelVectorEnv` is the important piece for throughput. It batches one-step
world-model transitions across many environments at once.

```python
import numpy as np
import torch

from wm_infra.rl.env import GoalReward, WorldModelVectorEnv
from wm_infra.rl.toy import ToyLineWorldModel


def initial_sampler(batch_size, device, dtype, generator):
    return torch.zeros(batch_size, 1, 1, device=device, dtype=dtype)


def goal_sampler(batch_size, device, dtype, generator):
    return torch.full((batch_size, 1, 1), 0.4, device=device, dtype=dtype)


env = WorldModelVectorEnv(
    ToyLineWorldModel(),
    num_envs=64,
    initial_state_sampler=initial_sampler,
    goal_state_sampler=goal_sampler,
    reward_fn=GoalReward(success_threshold=0.01, reward_scale=4.0),
    action_dim=3,
    max_episode_steps=12,
    auto_reset=True,
)

obs, info = env.reset(seed=7)
actions = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (64, 1))
next_obs, reward, terminated, truncated, info = env.step(actions)
```

This is the current bridge from `wm-infra`'s ECS-style runtime ideas to trainer
workloads: the trainer sees a vectorized environment contract, while the runtime
sees homogeneous state updates that can be batched.

## What We Borrow From EnvPool

`EnvPool` is a useful reference for the RL environment layer, but not for the
entire `wm-infra` product boundary.

The parts we now adopt directly are:

- explicit batched stepping as a first-class path rather than a helper wrapper
- explicit sync step semantics for `step_many`
- explicit batch policy metadata in runtime profiles
- explicit separation between northbound reset semantics and collector-local
  auto-reset behavior

The concrete repo behavior is now:

- northbound `/v1/envs/.../step` and `step_many` keep explicit reset semantics
- collector-side rollout gathering may auto-reset finished env sessions locally
- execution chunk formation is described by a shared `ExecutionBatchPolicy`
  instead of being only an implicit manager detail

This is intentionally different from EnvPool in one important way: `wm-infra`
still owns temporal control-plane objects such as trajectories, checkpoints,
artifacts, replay shards, and evaluation runs.

## Local Experiment Entry Point

Run the built-in experiment:

```bash
python examples/rl_train_toy_world.py \
  --updates 40 \
  --num-envs 64 \
  --horizon 12 \
  --eval-episodes 16 \
  --eval-interval 10 \
  --replay-dir benchmarks/results/rl_toy_replay \
  --temporal-root /tmp/wm_infra_rl_demo \
  --output benchmarks/results/rl_toy_demo.json
```

This entrypoint now does three things in one run:

- creates a declarative `ExperimentSpec`
- drives a `SynchronousCollector`, `LocalActorCriticLearner`, and `FixedTaskEvaluator`
- exports a replay shard artifact for the latest collected batch

Example summary output:

```json
{
  "best_mean_return": -2.475139617919922,
  "final_mean_return": -2.7087159156799316,
  "final_success_rate": 1.0,
  "num_envs": 64,
  "updates": 80
}
```

The full JSON artifact also stores per-update metrics. A healthy run should show:

- `best_mean_return` better than the first update's return
- `last_evaluation.success_rate` near `1.0`
- `replay_shard.uri` pointing at a persisted replay export

## RL Benchmark Entry Point

Use the dedicated benchmark harness when you want stable RL runtime numbers instead
of a raw training artifact dump.

```bash
python benchmarks/bench_rl_env.py \
  --updates 40 \
  --num-envs 32 \
  --horizon 8 \
  --eval-num-envs 8 \
  --eval-episodes 8 \
  --eval-interval 10 \
  --replay-dir benchmarks/results/rl_toy_replay_bench \
  --temporal-root /tmp/wm_infra_rl_benchmark \
  --output benchmarks/results/rl_toy_env_benchmark.json
```

Run the experimental Genie benchmark path:

```bash
python benchmarks/bench_rl_env.py \
  --experiment-name genie-token-benchmark \
  --env-name genie-token-grid-v0 \
  --train-task-id genie-token-train \
  --eval-task-id genie-token-eval \
  --updates 8 \
  --num-envs 8 \
  --horizon 4 \
  --eval-num-envs 4 \
  --eval-episodes 4 \
  --eval-interval 4 \
  --replay-dir benchmarks/results/rl_genie_replay \
  --temporal-root /tmp/wm_infra_rl_genie_benchmark \
  --output benchmarks/results/rl_genie_env_benchmark.json
```

The benchmark artifact now reports:

- `env_steps_per_sec`
- `step_latency_ms`
- `reward_stage_latency_ms`
- `trajectory_persist_latency_ms`
- `chunk_count`
- `avg_chunk_size`
- `max_chunk_size`
- `state_locality_hit_rate`
- `final_success_rate`

## Replay And Evaluation Data Plane

RL stepping now persists first-class control-plane objects:

- `EnvironmentSpec`
- `TaskSpec`
- `EnvironmentSessionRecord`
- `TransitionRecord`
- `TrajectoryRecord`
- `EvaluationRunRecord`
- `ReplayShardManifest`

That means a local experiment can export:

- per-step training transitions
- per-trajectory return summaries
- fixed-split evaluation runs
- replay shards for offline inspection

## How This Connects To Real World Models

The current example uses a toy world model because the northbound RL surface is the
thing being validated here, not a specific backend checkpoint.

To connect a real backend later, the backend needs to provide the same `WorldModel`
contract:

- `predict_next(latent_state, action)`
- `rollout(input)`
- `get_initial_state(observation)`

For a real world-model backend, you will usually add:

- task-conditioned resets
- richer observations than latent+goal concatenation
- explicit reward evaluators
- richer state handles than a toy latent snapshot

## Honest Current Limits

The repository still does **not** have these pieces yet:

- ECS-native env stepping in the runtime core. The current `step_many` path is
  batched in the RL manager, not yet chunked through the general ECS execution path.
- trainer adapters for external libraries such as Gymnasium, SB3, TorchRL, Slime,
  or Miles
- real backend integration for `genie-rollout` or other world-model backends.
  `genie-rollout` currently exposes prompt/token window generation, not a stable
  action-conditioned `WorldModel` contract.
- distributed asynchronous rollout workers for multi-node RL training
- industrial observability for RL metrics such as chunk size, reward-stage latency,
  state locality, and replay persistence latency

That means the current deliverable is a **trainer-facing RL substrate with a real
northbound env API**, not a full industrial RL platform yet. The contract shape,
trajectory data plane, replay export, and local experiment loop are in place. The
remaining work is to push those same semantics into ECS-native runtime batching,
real backends, and external trainer integrations.

For the current Genie-specific blocker details, read:

- `docs/GENIE_RL_BACKEND_GAP.md`
