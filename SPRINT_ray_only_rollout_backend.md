# Sprint: Ray-only Rollout Backend

## 核心结论

把 rollout backend 收敛为 Ray-only 是合理的，前提是一次性删干净 driver 进程内的 local fallback。

这次不是删除 `GenerationRuntime`、`GenerationWorker` 或 executor 协议。Ray worker 内部仍然需要这些 engine 组件来执行 generation。要删除的是：

```text
RolloutCollector 在 driver 进程里持有 live model，并在没有显式 runtime 时自动构建 in-process runtime
```

目标架构变成：

```text
RolloutCollector
  -> injected RolloutBackend
    -> RayDistributedRuntime
      -> RayRolloutWorker
        -> GenerationRuntime
```

单 GPU debug 也走 Ray：

```yaml
distributed:
  backend: ray
  rollout:
    num_workers: 1
    gpus_per_worker: 1.0
```

## 为什么要做

现在有两个 backend 名字：

```text
local
ray
```

虽然 generation contract 已经共享，但 `local` 仍然带来一个容易混淆的心理模型：单 GPU 训练时可以绕过 Ray，等多 GPU 再切 Ray。这样会让 Ray serialization、actor init、CUDA ownership、runtime rebuild、weight sync 等问题暴露得太晚。

如果项目目标是优先服务 multi-GPU / multi-node rollout，那么默认和日常 debug 都应该走 Ray：

```text
ray num_workers=1
  debug 单卡 Ray plumbing 和 rollout 语义

ray num_workers>1
  debug 多 worker split / gather / placement / sync
```

## 非目标

不要删除这些 engine 组件：

```text
vrl/engine/generation/runtime.py
vrl/engine/generation/worker.py
vrl/engine/generation/registry.py
vrl/engine/generation/runtime_spec.py
vrl/engine/generation/request.py
vrl/engine/generation/output.py
```

原因：Ray worker 仍然要在 actor 内构建 `GenerationRuntime`。

也不要把 model family 的构建逻辑写回 Ray launcher。Ray 层应该只消费 `GenerationRuntimeSpec` 和 `ChunkGatherer`。

## 目标边界

删除后只保留一个 backend：

```text
distributed.backend = ray
```

`RolloutCollector` 只负责：

```text
request -> runtime.generate(...) -> reward -> pack
```

`RolloutCollector` 不再负责：

```text
根据 live model 构建 runtime
在缺少 runtime 时 lazy fallback 到 local
解释 backend config
```

runtime 构建统一在 rollout wiring 层完成：

```text
build_rollout_runtime_inputs(...)
build_rollout_backend_from_cfg(...)
collector.set_runtime(...)
```

## 计划

### 1. 收紧 backend config

修改：

```text
vrl/rollouts/backend_config.py
```

目标：

```text
backend 默认改为 ray
只允许 backend == "ray"
删除 local/ray 双 backend 校验
缺少 distributed 配置时也返回 Ray 默认配置
```

建议保留 `backend` 字段一段时间，但只接受 `"ray"`。这样 YAML 仍然清楚表达资源后端，同时避免未来又静默加回 local。

### 2. 删除 local backend selector

修改：

```text
vrl/rollouts/backend.py
```

删除：

```text
local_runtime_builder 参数
backend == "local" 分支
local num_workers=1 校验
local 相关错误信息
```

保留：

```text
runtime 注入路径
Ray driver CUDA ownership 校验
runtime_spec + gatherer -> RayRolloutLauncher 路径
_require_rollout_backend(...)
```

目标逻辑：

```text
1. validate Ray config
2. 如果 runtime 已注入，直接使用
3. 否则必须有 runtime_spec + gatherer
4. 调 RayRolloutLauncher.launch(...)
5. 缺少 Ray launch inputs 时 fail fast
```

### 3. 删除 collector local runtime fallback

修改：

```text
vrl/rollouts/collect.py
```

删除：

```text
build_local_generation_runtime import
RolloutCollector.build_runtime()
runtime property 里的 lazy build fallback
```

保留：

```text
set_runtime(runtime)
runtime property
collect(...)
shutdown(...)
```

新的 `runtime` property 应该在没有 runtime 时明确报错：

```text
RolloutCollector runtime is not initialized; call set_runtime(...) before collect(...)
```

这可以防止训练脚本忘记设置 Ray runtime 时静默走旧路径。

### 4. 删除 local factory public export

修改：

```text
vrl/engine/generation/factory.py
vrl/engine/generation/__init__.py
```

如果 `build_local_generation_runtime(...)` 删除后没有调用者，就删除整个 `factory.py` 或至少停止导出。

注意：这不影响 Ray worker，因为 Ray worker 通过 `GenerationRuntimeSpec.runtime_builder` 加载 family runtime builder，不依赖这个 local factory。

### 5. 修改训练脚本 wiring

修改：

```text
vrl/scripts/sd3_5/train.py
vrl/scripts/wan_2_1/train.py
vrl/scripts/cosmos/train.py
vrl/scripts/janus_pro/train.py
vrl/scripts/nextstep_1/train.py
```

删除调用里的：

```text
local_runtime_builder=collector.build_runtime
```

确保每个脚本都先构建：

```text
rollout_runtime_inputs = build_rollout_runtime_inputs(...)
```

然后调用：

```text
collector.set_runtime(
    build_rollout_backend_from_cfg(
        cfg,
        runtime=rollout_runtime,
        driver_bundle=...,
        driver_policy=...,
        runtime_spec=rollout_runtime_inputs.runtime_spec,
        gatherer=rollout_runtime_inputs.gatherer,
    )
)
```

Ray-only 后，`rollout_runtime_inputs` 不应该是 `None`。如果是 `None`，应该 fail fast。

### 6. 修改 runtime input builder

修改：

```text
vrl/rollouts/runtime_inputs.py
```

删除：

```text
backend != "ray" return None
_RolloutBackendConfig.backend
```

目标：

```text
build_rollout_runtime_inputs(...) 总是为 Ray 构建 RolloutRuntimeInputs
```

`gpus_per_worker` 仍然从 `RolloutBackendConfig` 读取，用来决定 worker device：

```text
gpus_per_worker > 0 -> cuda
gpus_per_worker == 0 -> cpu
```

### 7. 配置默认值收敛

修改或新增：

```text
configs/base/distributed/ray_rollout.yaml
configs/experiment/*.yaml
```

当前很多 experiment 没有 include Ray distributed config。Ray-only 后必须处理这个问题：

```text
方案 A：每个 experiment defaults 显式加 /base/distributed/ray_rollout
方案 B：Python config 默认就是 Ray，experiment 不强制 include
```

建议使用方案 A。原因是资源配置应该显式出现在实验配置里，尤其是 `num_workers`、`gpus_per_worker`、`sync_trainable_state`。

单 GPU debug 可以新增一个 base config：

```text
configs/base/distributed/ray_rollout_single_gpu.yaml
```

内容：

```yaml
distributed:
  backend: ray
  rollout:
    num_workers: 1
    gpus_per_worker: 1.0
    cpus_per_worker: 4.0
    placement_strategy: STRICT_PACK
    allow_driver_gpu_overlap: false
    max_inflight_chunks_per_worker: 1
    sync_trainable_state: lora_only
```

### 8. 更新测试

删除或重写：

```text
tests/engine/generation/test_runtime_factory.py
tests/rollouts/test_runtime_inputs.py::test_rollout_runtime_inputs_return_none_for_local_backend
```

新增或修改测试覆盖：

```text
RolloutBackendConfig 只接受 ray
build_rollout_backend_from_cfg 缺少 runtime_spec/gatherer 时 fail fast
build_rollout_backend_from_cfg 有 runtime 注入时不启动 Ray
build_rollout_runtime_inputs 总是返回 RolloutRuntimeInputs
RolloutCollector collect 前未 set_runtime 时 fail fast
Ray driver CUDA ownership 校验仍然有效
```

保留 Ray 现有测试：

```text
tests/distributed/ray
tests/rollouts
tests/engine/generation
```

## 验证标准

基础静态检查：

```bash
.venv/bin/ruff check vrl tests
.venv/bin/python -m compileall -q vrl/scripts vrl/rollouts vrl/distributed/ray vrl/engine/generation
git diff --check
```

目标测试：

```bash
.venv/bin/python -m pytest -q tests/rollouts tests/distributed/ray tests/engine/generation
```

结构性检查：

```bash
rg -n "local_runtime_builder|build_local_generation_runtime|backend == \"local\"|backend: local|\"local\".*backend" vrl tests configs -g '*.py' -g '*.yaml'
```

允许出现的 `"local"` 只应该是无关含义，例如 node fallback metadata：

```text
node_ip = "local"
```

## 风险和处理

### 风险 1：单卡实验启动成本变高

这是预期代价。解决方式不是保留 local backend，而是提供明确的单卡 Ray config：

```text
ray num_workers=1
```

### 风险 2：driver policy 在 CUDA 上导致 Ray worker 抢同一张 GPU

继续保留现有保护：

```text
Ray backend 默认不允许 driver GPU overlap
```

Ray rollout 下 driver 侧应尽量不持有 rollout policy 的 CUDA 副本，或者只在明确 colocate 实验时开启：

```yaml
distributed:
  rollout:
    allow_driver_gpu_overlap: true
```

### 风险 3：测试依赖 Ray 后变慢

单元测试应该优先测试 builder、spec、collector contract，不一定每个测试都启动真实 Ray。真实 Ray smoke test 保留在 `tests/distributed/ray`。

### 风险 4：误删 engine runtime

不要把 `GenerationRuntime` 当成 local backend。它是 worker 内部 engine，Ray path 仍然依赖它。

## 完成标准

这次 sprint 完成时应满足：

```text
1. 配置层只接受 Ray rollout backend
2. RolloutCollector 不再能 lazy build local runtime
3. 训练脚本不再传 local_runtime_builder
4. build_rollout_runtime_inputs 不再为 local 返回 None
5. repo 中没有 local backend 分支或 local backend 测试
6. Ray num_workers=1 成为单 GPU debug path
7. Ray num_workers>1 继续走同一个 runtime/spec/gatherer contract
```

最终心智模型：

```text
Ray owns rollout process placement and resource isolation.
Engine owns generation scheduling and execution inside each worker.
RolloutCollector owns request/reward/packing only.
```
