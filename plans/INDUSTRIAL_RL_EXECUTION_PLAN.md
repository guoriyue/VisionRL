# 工业级 RL 训练集成执行文档

## 核心结论

`wm-infra` 现在已经有了一个能跑的本地 RL 起点，但还远远不是工业级 RL 训练系统。

接下来不能继续沿着 “toy env + toy trainer + 本地脚本” 这条线横向堆功能。正确方向是：

1. 保留 `wm-infra` 现有的 temporal control plane 身份
2. 引入真正的 trainer-facing env/session contract
3. 复制 `Slime` 和 `Miles` 里对我们最有价值的结构性元素
4. 让 rollout、trainer、replay、evaluation、stateful execution 彼此解耦
5. 用真实 benchmark 和 profile 驱动每一轮实现

这份文档不是学习笔记，而是后续开发时必须遵守的执行计划。

## 当前执行状态

截至当前仓库状态，前 3 个固定步骤已经有最小可运行实现：

1. `env contract`
   - 已有 `POST /v1/envs`
   - 已有 `reset / step / step_many / fork / checkpoint`
   - 已有 env/task catalog 查询

2. `trajectory / replay`
   - 已有 `EnvironmentSpec`
   - 已有 `TaskSpec`
   - 已有 `EnvironmentSessionRecord`
   - 已有 `TransitionRecord`
   - 已有 `TrajectoryRecord`
   - 已有 `EvaluationRunRecord`
   - 已有 `ReplayShardManifest`

3. `trainer adapter`
   - 已有 `ExperimentSpec`
   - 已有 `Collector`
   - 已有 `LearnerAdapter`
   - 已有 `Evaluator`
   - 已有本地 experiment entrypoint

当前还没有完成的重点是：

4. ECS runtime 原生 batched stepping
5. 真实 backend 接入
6. trainer/runtime 工业级解耦与 observability

当前新增进展：

- `genie-rollout` 已经补出一层显式 `GenieWorldModelAdapter`
- 已有实验性真实 backend env：`genie-token-grid-v0`
- 它已经通过 `reset / step / step_many / fork / checkpoint` 接入现有 RL env/session contract
- reward / terminated / truncated 已经是显式 RL 语义，不再藏在 metadata 里

当前剩余缺口：

- Genie RL action 现在是 token-control 级别，不是更高层任务语义
- RL chunking 仍然主要由 `RLEnvironmentManager` 发起，而不是更深的 runtime 入口
- benchmark 现在已能对比 toy 和 experimental Genie smoke，但还没有更长跑的 Genie task baseline

## 下一阶段执行顺序

从当前仓库状态继续开发时，不要再回头重复做 toy demo 层的工作。直接按下面顺序推进。

### Phase A. 掰正真实 backend contract

目标：

- 把 `genie-rollout` 和 RL env contract 之间的硬缺口写清楚
- 判断哪些能力可以复用，哪些必须新建

要做的事：

1. 盘点 `wm_infra/backends/genie.py`、`wm_infra/backends/genie_runner.py`、`wm_infra/models/base.py`
2. 明确 Genie 当前已经具备的状态对象：
   - prompt tokens
   - token window state
   - checkpoint delta
   - temporal lineage
3. 明确 Genie 当前缺失的 RL 必要能力：
   - action-conditioned `predict_next(latent_state, action)`
   - stable `reset() -> observation`
   - stable `state_handle -> observation` materialization
   - reward / terminated / truncated 挂接点
4. 产出一个最短闭环设计：
   - 是扩展现有 `GenieRunner`
   - 还是新增 `GenieWorldModelAdapter`
   - action 怎么编码到 Genie transition
   - observation 用 token / latent / decoded frame 的哪一种

完成状态：

- 已完成

完成结果：

- 已新增 `GenieWorldModelAdapter`
- 已定义 state / observation / action / reward / termination contract
- 已把 contract 文档从 blocker 说明推进到当前实现说明

### Phase B. 把 RL stepping 真接进 runtime

目标：

- RL workload 真正吃到 runtime batching 和 locality
- 不再只是 manager 层切块

要做的事：

1. 盘点 `wm_infra/core/execution.py`、`wm_infra/core/engine.py`、`wm_infra/core/scheduler.py`
2. 明确 RL env step 对应的 runtime 实体：
   - `ExecutionEntity`
   - `BatchSignature`
   - `ExecutionChunk`
3. 把当前 RL chunking 从 manager 层下推到更通用的 runtime 入口
4. 增加最小 runtime 指标：
   - `env_step_chunk_total`
   - `env_step_chunk_size`
   - `reward_stage_ms`
   - `trajectory_persist_ms`
   - `state_locality_hit_rate`
5. 为 reset / fork / checkpoint 明确 state reuse 路径

当前状态：

- 已部分完成

当前结果：

- `step_many` 已显式使用 `ExecutionEntity` / `BatchSignature` / `ExecutionChunk`
- runtime profile 已暴露 chunk 和 locality 指标
- 还没有把 RL chunking 完整下推到更通用 runtime entrypoint

### Phase C. 做扎实 benchmark

目标：

- 不再只看“能跑”
- 要有 RL workload 的基线和回归门

要做的事：

1. 固定 toy RL benchmark 配置
2. 新增 benchmark 输出以下指标：
   - `env_steps_per_sec`
   - `step_latency_ms`
   - `reward_stage_ms`
   - `trajectory_persist_ms`
   - `chunk_count`
   - `max_chunk_size`
   - `state_locality_hit_rate`
3. 保留至少一份 artifact 到 `benchmarks/results/`
4. 给 RL path 增加 benchmark smoke test 或 artifact contract test

完成标准：

- 每轮 runtime 改动前后都能对比 RL 指标
- benchmark 结果可以直接和当前 toy baseline 对照

### Phase D. 只有在 contract 清楚后，才继续真实 backend

目标：

- 用真实 backend 跑通一条 RL env path

执行规则：

1. 先做 `genie-rollout`
2. 已完成 contract 接线后，再继续扩展任务语义和 runtime 下推
3. 不要把当前 experimental token-control env 夸大成完整 Genie RL 平台
4. 所有后续 Genie 工作都必须继续保留 benchmark 和 artifact

完成状态：

- 已达到最小完成标准

完成结果：

- `genie-token-grid-v0` 已能通过 RL env/session contract 跑通
- trainer-facing API 不再只绑定 toy world

## 下轮执行注意事项

1. 每一轮都先定义目标指标，再改代码
2. 每一轮只打一个主瓶颈，不要一次堆很多猜测性改动
3. 先跑 focused tests，再跑更宽的 server tests
4. benchmark artifact 要落盘，不要只在终端看
5. 如果碰到真实 backend contract 缺失，要把缺口写成明确 blocker，而不是模糊结论

## 目标

把仓库推进到下面这个层级：

- 不只是 sample/rollout 基础设施
- 不只是 world-model runtime substrate
- 而是能承载真实 RL 训练循环的工业级 world-model environment infra

这里的“工业级”至少意味着：

1. trainer 可以把 `wm-infra` 当环境系统来接
2. batched stepping 是一等公民，不是附带功能
3. reward / done / truncation / info 语义稳定
4. rollout collection、trajectory、replay、evaluation 有正式对象模型
5. trainer 和 rollout worker 解耦
6. ECS-style execution runtime 为 RL workload 真正贡献吞吐、cache hit、state locality

## 不要做错的事

后续开发中，下面这些方向都算错误方向：

1. 把现有 `POST /v1/samples` 强行当 RL env API 用
2. 把 trainer 逻辑塞进 backend adapter
3. 把 reward/done 藏进 metadata blob
4. 没有 trajectory/replay schema 就先写大而全 trainer
5. 没有 benchmark/profile 基线就大改 runtime
6. 继续把 RL 只做成 in-process demo，不补 northbound session API

## 必须从 Slime 复制的必要元素

`Slime` 最值得复制的不是某个算法实现，而是 **训练和 rollout 的解耦结构**。

### 要复制的要点

1. 训练器和 rollout worker 分离
   - trainer 只负责参数更新
   - rollout worker 只负责收集 trajectory
   - 两边之间传的是 trajectory / batch / metrics，不是 ad hoc Python 对象

2. batched rollout collection
   - rollout 的基础单位不是单个 env
   - 应该是可批处理的 env session 集合

3. 明确的数据面
   - observation
   - action
   - reward
   - terminated
   - truncated
   - next_observation
   - episode_id / env_id / task_id / policy_version

4. 训练与推理资源隔离
   - trainer 占用更新资源
   - world-model stepping 占用 rollout 资源
   - 不能把两者塞进一个同步 Python 循环里长期运行

5. 可插拔 evaluator
   - evaluation 不应作为训练 loop 中的临时代码
   - 需要单独 evaluator runner 和固定评测任务集

### 在 `wm-infra` 中的对应落点

- `wm_infra/api/`: 新增 env/session northbound API
- `wm_infra/controlplane/`: 新增 trajectory/replay/eval schema
- `wm_infra/core/`: rollout worker runtime、batched stepping、state-aware scheduler
- `wm_infra/ops/` 或 `wm_infra/api/metrics.py`: trainer / rollout / replay / eval 指标

## 必须从 Miles 复制的必要元素

`Miles` 最值得复制的是 **工业化实验管理和环境目录抽象**。

### 要复制的要点

1. environment registry / task catalog
   - 环境不是隐式 Python 类
   - 任务、split、版本、默认 reward 规则要可枚举

2. 明确 experiment spec
   - 一个 RL 实验必须能被 declarative 地描述
   - 包括 env、task set、policy config、collector config、trainer config、eval config

3. 训练运行单元标准化
   - rollout worker
   - learner
   - evaluator
   - replay / storage
   - metrics sink

4. 结果可追踪
   - 哪个 policy version 在哪个 task 上跑出什么结果
   - 哪条 trajectory 来自哪个 env session、哪个 world-model checkpoint

5. 任务与环境解耦
   - 同一类 env 可以有很多 task
   - 同一个 trainer 可以跑不同 task split

### 在 `wm-infra` 中的对应落点

- `wm_infra/controlplane/`: `EnvironmentSpec`、`TaskSpec`、`TaskSplit`
- `wm_infra/api/protocol.py`: experiment / collector / evaluator request schema
- `wm_infra/api/server.py`: registry 查询与 env session 创建接口
- `docs/` 与 `plans/`: 标准实验配置与使用文档

## `wm-infra` 的最终定位

`wm-infra` 不应该变成一个通用 RL 算法仓库。

它应该成为：

- world-model / video-model 环境系统
- temporal rollout 与 state 管理系统
- trajectory / replay / evaluation 控制面
- 与 trainer 解耦的 RL execution substrate

也就是说：

- trainer 算法本身可以是外部的
- 但 environment API、session、trajectory、state locality、artifact lineage 应该由 `wm-infra` 拥有

## 北向 API 目标形状

当前主产品面仍然是 `POST /v1/samples`。这不能删，但 RL 需要新接口。

### 必须新增的 env/session API

1. `POST /v1/envs`
   - 创建 environment session
   - 返回 `env_id`、`task_id`、`episode_id`、`state_handle_id`

2. `POST /v1/envs/{env_id}/reset`
   - 重置环境
   - 返回 observation、info、checkpoint/state refs

3. `POST /v1/envs/{env_id}/step`
   - 单 session 一步推进
   - 返回 observation、reward、terminated、truncated、info

4. `POST /v1/envs/{env_id}/step_many`
   - batched stepping
   - 这是工业级吞吐的关键接口

5. `POST /v1/envs/{env_id}/fork`
   - 从已有状态分叉
   - 对 imagined branching 和 curriculum 很重要

6. `POST /v1/envs/{env_id}/checkpoint`
   - 显式存 checkpoint

7. `DELETE /v1/envs/{env_id}`
   - 回收 session 和 runtime state

### 返回对象必须固定包含

- `observation`
- `reward`
- `terminated`
- `truncated`
- `info`
- `env_id`
- `episode_id`
- `task_id`
- `state_handle_id`
- `checkpoint_id`（如果发生切点）
- `policy_version`（如果请求中带入）

## Control Plane 必须新增的对象

### 1. EnvironmentSpec

描述一个 environment family：

- `env_name`
- `backend`
- `observation_mode`
- `action_space`
- `reward_schema`
- `default_horizon`
- `supports_batch_step`
- `supports_fork`

### 2. TaskSpec

描述一个任务实例或模板：

- `task_id`
- `env_name`
- `task_family`
- `goal_spec`
- `seed_policy`
- `difficulty`
- `split`
- `reward_overrides`

### 3. EnvironmentSessionRecord

描述一个活跃或已结束的 env session：

- `env_id`
- `episode_id`
- `task_id`
- `backend`
- `status`
- `current_step`
- `state_handle_id`
- `checkpoint_id`
- `policy_version`

### 4. TransitionRecord

每一步的标准训练数据对象：

- `transition_id`
- `env_id`
- `episode_id`
- `step_idx`
- `observation_ref`
- `action`
- `reward`
- `terminated`
- `truncated`
- `next_observation_ref`
- `info`

### 5. TrajectoryRecord

轨迹级聚合对象：

- `trajectory_id`
- `episode_id`
- `task_id`
- `policy_version`
- `num_steps`
- `return`
- `success`
- `transition_refs`

### 6. EvaluationRunRecord

评测运行对象：

- `eval_run_id`
- `policy_version`
- `task_split`
- `env_name`
- `aggregate_metrics`

## Runtime 必须演化到的形状

工业级 RL 训练不接受“每个 env 独立 Python step 一次”。

### 当前必须成立的 runtime 原则

1. batch unit 不是 request
2. batch unit 也不是 token
3. batch unit 是同阶段、同状态形状、同任务签名、同 observation/reward contract 的 env sessions

### ECS-style runtime 在 RL 下的具体对应

- `ExecutionEntity`
  - 一个可执行 env step 单元
- `BatchSignature`
  - `backend + state_shape + task_family + observation_mode + reward_schema`
- `ExecutionChunk`
  - 一批同构 env step
- `StageSystem`
  - `state_materialize -> transition -> reward -> checkpoint -> trajectory_persist -> evaluation_hook`

### scheduler 必须新增的 RL 维度

1. state locality
2. task homogeneity
3. observation contract homogeneity
4. policy version affinity
5. env session stickiness

## 阶段化开发顺序

下面的顺序就是接下来实际执行顺序，不能乱。

### Step 1. 固化 env contract

目标：

- 在代码里正式引入 trainer-facing env protocol
- 不再停留在 demo 级 API

要做的事：

1. 在 `wm_infra/api/protocol.py` 增加：
   - `CreateEnvironmentRequest`
   - `CreateEnvironmentResponse`
   - `ResetEnvironmentRequest`
   - `StepEnvironmentRequest`
   - `StepEnvironmentResponse`
   - `BatchedStepRequest`
   - `BatchedStepResponse`

2. 在 `wm_infra/controlplane/` 增加：
   - `EnvironmentSpec`
   - `TaskSpec`
   - `EnvironmentSessionRecord`

3. 在 `wm_infra/api/server.py` 增加最小 northbound env routes

完成标准：

- 本地可以通过 HTTP 创建 env、reset、step、step_many
- 单环境和 batched 返回结构一致

### Step 2. 让 trajectory/replay 成为正式对象

目标：

- 训练数据不再只是运行时临时张量

要做的事：

1. 增加：
   - `TransitionRecord`
   - `TrajectoryRecord`
   - `ReplayShardManifest`

2. 每次 `step/step_many` 后落标准 transition 数据
3. 每个 episode 结束后形成 trajectory summary

完成标准：

- 任意训练 run 可以导出 trajectory
- 可以按 policy version / task split 回看结果

### Step 3. 把现有 RL demo 升级成 trainer adapter

目标：

- 不再只有脚本
- 要有标准 trainer-facing 接口

要做的事：

1. 引入：
   - `Collector`
   - `LearnerAdapter`
   - `Evaluator`

2. 先做最小本地实现：
   - synchronous collector
   - local learner loop
   - fixed task evaluation

3. 训练脚本改成真正的 experiment entrypoint

完成标准：

- 一个 experiment spec 可以完整驱动 collector、learner、evaluator

### Step 4. 让 ECS runtime 真正服务 RL batched stepping

目标：

- batched step 不是 HTTP 层拼数组
- 而是 runtime 原生支持

要做的事：

1. 把 env step 请求映射成 `ExecutionEntity`
2. 按 `BatchSignature` 分组形成 chunk
3. 在 runtime 里新增 reward stage 与 trajectory persist stage
4. 暴露指标：
   - `env_step_chunk_size`
   - `env_step_chunk_total`
   - `reward_stage_ms`
   - `trajectory_persist_ms`
   - `state_locality_hit_rate`

完成标准：

- `step_many` 的吞吐显著好于逐个 `step`
- benchmark 能直接看到 chunk-level 指标

### Step 5. 引入任务目录与 split

目标：

- 环境系统不能只有“裸 env”
- 必须有任务目录

要做的事：

1. 增加 task registry
2. 支持：
   - train split
   - val split
   - test split
3. 支持 seed policy、difficulty、goal template

完成标准：

- trainer 能指定 task split
- evaluator 能跑固定评测集

### Step 6. 接入真实 world-model backend

目标：

- 不再只跑 toy world

优先顺序：

1. `genie-rollout`
2. `cosmos`

要做的事：

1. 抽象 backend-specific observation builder
2. 抽象 reward evaluator hook
3. 抽象 task-conditioned reset
4. 在 batched env path 上跑通至少一个真实 backend

完成标准：

- 至少有一个真实 backend 能用 env/session API 驱动训练或 collector

### Step 7. 做训练器解耦

目标：

- trainer 进程和 rollout/runtime 分开

要做的事：

1. rollout worker 只负责 env session 和 trajectory 收集
2. learner 只消费 replay/trajectory
3. evaluator 独立运行

完成标准：

- rollout worker 挂掉不会直接破坏 learner
- learner 可以在不直接持有 env state 的情况下工作

### Step 8. 做工业级 observability

目标：

- 不靠猜测看系统状态

必须有的指标：

- env steps/sec
- batched step latency
- reward stage latency
- trajectory persist latency
- active env sessions
- session reset rate
- state materialize latency
- state transfer latency
- cache hit rate
- chunk occupancy
- evaluator throughput

完成标准：

- 有明确 benchmark artifact 和 profile 结果

## 明天开始逐步开发时的执行规则

接下来每一轮开发必须遵守下面这个 loop：

1. 先选定具体 workload
   - `toy vector env`
   - `genie env stepping`
   - `cosmos env stepping`

2. 明确目标指标
   - 正确性
   - env steps/sec
   - state locality
   - cache hit
   - end-to-end latency

3. 先测 baseline
4. 只改一个主要瓶颈
5. 重新 benchmark
6. 记录 before/after
7. 再进入下一轮

## 文档驱动的开发顺序

当后续按这份文档继续开发时，顺序固定如下：

1. 先做 `Step 1`
2. 通过测试和最小示例
3. 再做 `Step 2`
4. 通过 replay/trajectory 验证
5. 再做 `Step 3`
6. 再做 `Step 4`
7. 再接真实 backend

不能跳过 `env contract` 和 `trajectory schema` 就直接冲 trainer 或多机调度。

## 明确的学习清单

如果目标是真正把这条线做成工业级，你需要补的不是更多算法名词，而是这些系统能力：

1. `reset/step` env contract 设计
2. batched environment stepping
3. reward / `terminated` / `truncated` 稳定语义
4. trajectory / replay 数据面
5. trainer 与 rollout/runtime 解耦
6. state-aware batching / ECS-style execution
7. task catalog / split / evaluation sets
8. industrial experiment spec 与 observability

## 当前已有基础

当前仓库已经有的有利条件：

1. temporal control plane 已经存在
2. `episode / branch / checkpoint / state_handle` 已经是一等公民
3. ECS-style execution runtime 已经有雏形
4. 本地 trainer-facing toy env 已经能跑通

所以这条路不是从零开始，而是从“有 world-model infra 基础，但没有工业级 RL 北向接口”开始。

## 交付标准

后续我按这份文档继续推进时，每个阶段都必须同时满足：

1. 代码存在
2. 测试存在
3. 文档存在
4. benchmark 或 profile 结果存在
5. 能清楚说明这一步复制了 Slime/Miles 的哪一个必要元素

如果做不到这五点，这一步就不算完成。
