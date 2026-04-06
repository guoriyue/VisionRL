# RL 训练集成计划

## 目标

把 `wm-infra` 从“能做时序 sample/rollout”的仓库，推进到“能承载真实 RL 训练接入”的状态。

这里的目标不是一晚上变成完整 RL 平台，而是把最关键的基础打牢：

1. 训练器能以标准 env contract 跑起来
2. world model 的状态转移能批量化给训练器使用
3. reward / done / truncation 语义明确
4. 明确下一阶段 northbound API 和 replay/export 应该怎么长

## 已完成

### Step 1. 识别外部参照系

已经对齐了需要学习的外部方向：

- OpenReward / OpenReward Standard
- Slime
- Miles

核心不是照搬它们的产品，而是学习它们的共性：

- 统一 env/session 抽象
- batched environment stepping
- 明确的 reward / done 语义
- rollout 和 trainer 解耦
- 任务目录、版本、split、dataset/replay 的概念

验收标准：

- 这些概念已经收敛进当前计划和使用文档

### Step 2. 给仓库补 trainer-facing RL 适配层

已经新增：

- `wm_infra/rl/env.py`
- `wm_infra/rl/toy.py`
- `wm_infra/rl/demo.py`
- `examples/rl_train_toy_world.py`

当前已具备：

- 单环境 `reset/step`
- 向量化 batched `reset/step`
- goal-conditioned reward
- 可运行的 actor-critic 示例

验收标准：

- 示例脚本可直接运行
- 训练结果能明显优于初始 return

### Step 3. 建立最小可运行 RL 示例

当前可直接运行：

```bash
python examples/rl_train_toy_world.py --updates 80 --num-envs 64 --horizon 12
```

当前验证口径：

- `best_mean_return` 好于第一轮
- `final_success_rate` 接近 `1.0`

验收标准：

- 本地 smoke run 成功
- 结果落盘为 JSON artifact

### Step 4. 补齐测试，防止这条线回归

已经新增并跑通：

- `tests/test_rl_env.py`
- `tests/test_rl_api.py`
- `tests/test_rl_training.py`

当前已覆盖：

1. `WorldModelEnv.reset/step`
2. `WorldModelVectorEnv.step` 和 `auto_reset`
3. `/v1/envs`、`reset`、`step`、`step_many`、`fork`
4. `Collector` 的 batched transition 持久化
5. `run_local_experiment()` 的训练、评测、replay 导出

验收标准：

- 相关 pytest 用例通过
- 不依赖外部模型和外部服务

### Step 5. 写清楚“怎么用”和“现在还缺什么”

已经补齐：

- `docs/RL_ENV_USAGE.md`

文档当前已明确：

1. 当前 RL 入口在哪里
2. `/v1/envs` 怎么用
3. 本地 experiment entrypoint 怎么跑
4. replay / evaluation 数据面现在是什么
5. 当前离工业级 RL 还差什么

验收标准：

- 明早起来可以直接照文档运行
- 文档对能力边界诚实

### Step 6. 设计 northbound RL environment API

已经新增 northbound 接口：

- `POST /v1/envs`
- `GET /v1/env-specs`
- `GET /v1/task-specs`
- `POST /v1/envs/{env_id}/reset`
- `POST /v1/envs/{env_id}/step`
- `POST /v1/envs/{env_id}/step_many`
- `POST /v1/envs/{env_id}/fork`
- `POST /v1/envs/{env_id}/checkpoint`
- `DELETE /v1/envs/{env_id}`

当前已满足：

- schema 明确
- 单环境和 batched 返回结构一致
- 能表达 reward、terminated、truncated、info

### Step 7. 让 control plane 真正理解 RL 轨迹

已经新增并落盘：

- `EnvironmentSpec`
- `TaskSpec`
- `EnvironmentSessionRecord`
- `TransitionRecord`
- `TrajectoryRecord`
- `EvaluationRunRecord`
- `ReplayShardManifest`

当前已满足：

- 可导出训练轨迹
- 可导出 replay shard
- 可按 policy version / task split 回看 evaluation

## 正在推进

## 接下来按顺序推进

### Step 6. 设计 northbound RL environment API

目标不是替代 `POST /v1/samples`，而是补一层专门给训练器的接口。

建议新增语义对象：

- `EnvironmentSession`
- `EnvironmentSpec`
- `StepRequest`
- `StepResponse`
- `ResetRequest`
- `BatchedStepRequest`

建议 northbound 接口：

- `POST /v1/envs`
- `POST /v1/envs/{env_id}/reset`
- `POST /v1/envs/{env_id}/step`
- `POST /v1/envs/{env_id}/step_many`
- `POST /v1/envs/{env_id}/fork`
- `POST /v1/envs/{env_id}/checkpoint`
- `DELETE /v1/envs/{env_id}`

验收标准：

- schema 明确
- 单环境和 batched 语义一致
- 能表达 reward、terminated、truncated、info

### Step 7. 让 control plane 真正理解 RL 轨迹

当前 control plane 擅长：

- sample
- artifact
- episode / branch / checkpoint / state_handle

还缺：

- trajectory
- transition
- reward record
- episode return summary
- trainer/eval split

需要新增的持久化对象：

- `TrajectoryRecord`
- `TransitionRecord`
- `RewardRecord`
- `EpisodeSummary`

验收标准：

- 可导出训练轨迹
- 可做离线分析和评测

### Step 8. 把 ECS execution runtime 真正接进 RL workload

当前已经有 ECS-like runtime 思路，但主要还服务 rollout/sample。

当前状态：

- `step_many` 已经按 `ExecutionChunk` 切块执行
- 已经暴露 `chunk_count`、`max_chunk_size`、`reward_stage_ms`、`trajectory_persist_ms`、`state_locality_hit_rate`
- 但 locality-aware scheduler 和 state reuse 还只是最小形态，不是完整工业级 runtime

下一阶段要做的是：

1. 用 `ExecutionChunk` 驱动 batched env stepping
2. 用 `BatchSignature` 按 state shape / task shape 分组
3. 用 locality-aware scheduler 降低 state 迁移
4. 用 state reuse / branch reuse 降低 reset/fork 成本

验收标准：

- batched RL stepping 比逐 env stepping 明显更快
- profile 中能看到 chunk size、cache hit、state locality 指标

### Step 9. 接入真实 world-model backend

优先顺序：

1. `genie-rollout`
2. `cosmos`
3. 其他未来 backend

接入要求：

- backend 能实现 `WorldModel` contract
- reset / step 的输入输出形状稳定
- reward evaluator 可外挂
- 能在 batched env path 上跑

当前阻塞：

- `genie-rollout` 现有实现是 prompt/token window generation
- 还没有 action-conditioned `predict_next`
- 还没有面向 RL session 的稳定 `reset / state / observation` contract

验收标准：

- 不只是 toy world，至少有一条真实 backend 能用 RL env 跑通

### Step 10. 接入真实 trainer

下一层要做的不是再造轮子，而是补 adapter。

优先方向：

- Gymnasium-style adapter
- TorchRL collector adapter
- Slime/Miles 风格 rollout adapter

验收标准：

- 训练器能直接消费 `wm-infra` 的 env/session 接口
- rollout 和 trainer 进程可以解耦

### Step 11. 建立 benchmark 和 profile 基线

真实要看的指标：

- env steps/sec
- batch size / chunk size
- queue wait
- step latency
- reward latency
- state materialize latency
- state transfer latency
- cache hit rate
- GPU utilization

验收标准：

- 每次优化前后都有可对照基线
- 不再凭感觉讨论“适不适合 RL”

当前状态：

- 已新增 `benchmarks/bench_rl_env.py`
- 已新增 `tests/test_bench_rl_env.py`
- 已有可落盘 artifact：`benchmarks/results/rl_toy_env_benchmark.json`
- 当前 benchmark 已直接输出：
  - `env_steps_per_sec`
  - `step_latency_ms`
  - `reward_stage_latency_ms`
  - `trajectory_persist_latency_ms`
  - `chunk_count`
  - `avg_chunk_size`
  - `max_chunk_size`
  - `state_locality_hit_rate`

## 明早可直接看的交付物

明早应该能直接看的东西包括：

1. 可运行示例脚本
2. RL 使用文档
3. 测试覆盖
4. 这份分步骤计划

## 明确的学习优先级

如果你是亲自补这条线，学习顺序应该是：

1. Gym 风格 env contract
2. batched env stepping
3. reward / terminated / truncated 语义
4. trajectory / replay 数据面
5. trainer 和 rollout 解耦
6. state-aware runtime 与 ECS batching
7. 再去学 OpenReward、Slime、Miles 的系统边界和调度方式

不要反过来。

先把 env contract 和 batched stepping 做对，再谈大平台。
