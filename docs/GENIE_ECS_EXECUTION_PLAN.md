# Genie ECS 执行计划

本文件是 Genie ECS 长任务的唯一执行计划。
执行时只允许推进 `genie-rollout`，不得把 Cosmos、Wan 或其他 backend 混入执行范围。

## 全局原则

- 唯一状态游标：`plans/genie_ecs_status.json`
- 每一步完成前必须同时满足该步的 `Done When`
- 每一步失败后必须严格执行对应的 `Next On Fail`
- 每一步完成后必须完成以下动作，且缺一不可：
  1. 更新 `plans/genie_ecs_status.json`
  2. 保存该步 benchmark 结果到 `benchmarks/results/`
  3. 更新相关文档
  4. 运行该步要求的测试
  5. 提交一个 git commit
- 不得删除 legacy Genie 路径，除非 cleanup gate 全部满足

## Cleanup Gate

只有同时满足以下四项，才允许进入删除 legacy Genie 代码的步骤：

1. 语义等价测试通过
2. 默认 Genie workload 不回归
3. heavy Genie workload 不显著回归
4. sample manifest、temporal lineage、artifact visibility 全部保留

本计划中的“不回归”按 benchmark gate 定义：

- success rate 必须保持 `1.0`
- 默认 workload:
  - `genie_default_batched.json` 的 `terminal mean` 不得高于 `genie_default_baseline.json` 的 `1.05x`
  - `genie_default_batched.json` 的 `terminal p95` 不得高于 `genie_default_baseline.json` 的 `1.10x`
- heavy workload:
  - `genie_heavy_on.json` 的 `terminal mean` 不得高于 `genie_heavy_off.json` 的 `1.05x`
  - `genie_heavy_on.json` 的 `terminal p95` 不得高于 `genie_heavy_off.json` 的 `1.10x`

## 必须保留的 Benchmark Artifact

- `benchmarks/results/genie_default_baseline.json`
- `benchmarks/results/genie_default_batched.json`
- `benchmarks/results/genie_profile_baseline.json`
- `benchmarks/results/genie_profile_batched.json`

## 执行步骤

### Step 1: 固化 baseline

目标：

- 用当前 `genie-rollout` 路径重新固化 baseline 与 batched 对照
- 产出默认 workload、profile workload、heavy workload 的基线数据
- 把当前结论写进文档与状态游标，作为后续所有 gate 的比较基准

必须产出：

- `benchmarks/results/genie_default_baseline.json`
- `benchmarks/results/genie_default_batched.json`
- `benchmarks/results/genie_profile_baseline.json`
- `benchmarks/results/genie_profile_batched.json`
- `benchmarks/results/genie_heavy_off.json`
- `benchmarks/results/genie_heavy_on.json`

测试要求：

- `pytest tests/test_benchmarking.py tests/test_bench_samples_api.py tests/test_genie.py tests/test_genie_batcher.py tests/test_genie_runtime.py`

Done When：

- 六个 benchmark artifact 均由当前工作树重跑生成
- `plans/genie_ecs_status.json` 记录了六个 artifact 的路径、关键延迟指标与 gate 基线
- 相关文档记录了 benchmark 命令、运行环境与当前基线结论
- 测试全部通过
- 已提交一个仅包含 Genie ECS 本步相关变更的 commit

Next On Fail：

- 如果 benchmark 无法运行或测试失败，停止推进
- 在状态游标写入失败原因、失败命令、最后成功生成的 artifact、建议下一步

### Step 2: 收敛 cross-request batching policy，解决 heavy workload regression

目标：

- 只调整 `genie-rollout` 的 batching policy
- 收敛 queue batching 与 transition batching 的兼容性规则
- 解决 heavy workload 的 batching 回归；若无回归，则把当前策略固化成显式 gate

优先维护路径：

- `wm_infra/backends/genie.py`
- `wm_infra/backends/genie_batcher.py`
- `wm_infra/backends/job_queue.py`

必须产出：

- 更新后的 `benchmarks/results/genie_default_batched.json`
- 更新后的 `benchmarks/results/genie_profile_batched.json`
- 更新后的 `benchmarks/results/genie_heavy_on.json`
- 必要时附带调参试验 artifact，但不能替代上述固定文件

测试要求：

- `pytest tests/test_genie.py tests/test_genie_batcher.py tests/test_server.py tests/test_bench_samples_api.py`

Done When：

- batching policy 的兼容键、批次等待策略、批次规模策略已明确落到代码与文档
- heavy workload 满足 cleanup gate 中的 heavy benchmark gate
- 默认 workload 仍满足 cleanup gate 中的 default benchmark gate
- 文档写明策略、权衡与已验证结果
- 测试全部通过
- 已提交一个仅包含 Genie ECS 本步相关变更的 commit

Next On Fail：

- 只允许继续做 batching policy 收敛与回归定位
- 不得跳到 Step 3 或后续步骤
- 在状态游标中记录当前最佳策略、失败场景、阻塞指标与建议下一步

### Step 3: 扩展 stage runtime 的调度和 profile 能力

目标：

- 扩展 `genie-rollout` stage runtime 的调度可观测性与 profile 表达
- 统一单请求与批请求执行时的 stage/profile 语义
- 为后续删除 legacy 路径创造共享执行骨架

优先维护路径：

- `wm_infra/backends/genie.py`
- `wm_infra/backends/genie_runner.py`
- `wm_infra/backends/genie_runtime.py`
- `wm_infra/backends/genie_scheduler.py`

测试要求：

- `pytest tests/test_genie.py tests/test_genie_runtime.py tests/test_server.py`

Done When：

- stage runtime 能稳定产出调度与 profile 元数据
- 单请求路径与批请求路径的 runtime/profile 结构一致或明确兼容
- 文档写清 stage profile 字段、调度字段、可用于 benchmark gate 的字段
- 测试全部通过
- 已提交一个仅包含 Genie ECS 本步相关变更的 commit

Next On Fail：

- 只允许继续修复 runtime/profile 语义
- 不得提前删除 legacy 路径
- 在状态游标中记录缺失字段、失败测试与建议下一步

### Step 4: 补齐 benchmark gate 和回归防护

目标：

- 把 benchmark 对比、回归阈值、语义等价保护变成仓库内可执行的回归防护

测试要求：

- `pytest tests/test_benchmarking.py tests/test_bench_samples_api.py tests/test_genie.py tests/test_genie_batcher.py tests/test_genie_runtime.py tests/test_server.py`

Done When：

- benchmark artifact 内嵌对比信息或仓库测试能直接验证 gate
- 默认 workload 与 heavy workload 的 gate 可自动判断
- 语义等价、manifest 保留、temporal lineage 保留、artifact visibility 保留都有测试覆盖
- 文档写明如何重跑 gate 与如何解释失败
- 测试全部通过
- 已提交一个仅包含 Genie ECS 本步相关变更的 commit

Next On Fail：

- 只允许继续补齐 gate 和测试
- 不得进入 legacy cleanup
- 在状态游标中记录缺少的 gate、失败测试与建议下一步

### Step 5: 仅在所有 gate 满足后删除 legacy Genie 代码

目标：

- 删除 legacy Genie 路径
- 保留 `genie-rollout` 的 stage runtime、manifest、lineage、artifact 语义

优先维护路径：

- `wm_infra/backends/genie.py`
- `wm_infra/backends/genie_runner.py`
- `wm_infra/backends/genie_runtime.py`

测试要求：

- `pytest tests/test_genie.py tests/test_genie_batcher.py tests/test_genie_runtime.py tests/test_server.py`

Done When：

- cleanup gate 四项全部满足
- legacy Genie 代码已删除，且无行为回退
- 文档明确说明 legacy 已移除以及保留的 Genie ECS 路径
- 测试全部通过
- 已提交一个仅包含 Genie ECS 本步相关变更的 commit

Next On Fail：

- 立即停止 cleanup
- 回到满足 gate 的最近状态继续修复，不得继续删代码
- 在状态游标中记录失败 gate、失败测试与建议下一步

### Step 6: cleanup 后重跑基准和测试并更新文档

目标：

- 在 legacy 删除后重新验证完整 Genie ECS 结果

必须产出：

- 更新后的四个必须保留 benchmark artifact
- 如有 heavy 验证，保留更新后的 `genie_heavy_off.json` 与 `genie_heavy_on.json`

测试要求：

- `pytest`

Done When：

- cleanup 后 benchmark 与测试全部重跑完成
- cleanup gate 继续满足
- 文档、计划结论与状态游标全部同步到最终状态
- 已提交一个仅包含 Genie ECS 本步相关变更的 commit

Next On Fail：

- 停止推进
- 在状态游标中记录失败 benchmark、失败测试、最后完成步骤与建议下一步
