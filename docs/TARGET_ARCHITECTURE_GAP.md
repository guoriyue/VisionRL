# 目标架构与现状差距重构图

这份文档把 `wm-infra` 的目标架构、当前代码落点、以及最具体的重构差距放在一张图里。

当前版本刻意收敛到一个更窄的主题：

- 先聚焦纯低层 serving
- 先聚焦 extreme serving performance
- 暂时不把 higher-level scheduling、review/export、广义 control-plane 扩展放在主叙事中心

也就是说，这份文档现在优先回答的是：

> 如果目标是把 `wan-video` 和 `genie-rollout` 做成更强的底层 serving runtime，最该重构什么？

产品前提保持明确：

- 主产品路径是 `wan-video` 和 `genie-rollout`
- `rollout-engine` 仍然保留，但它是运行时基座与 bring-up 基础设施
- 这不是一个泛化的“serve any model”仓库

## 1. 一张图看目标 serving 架构

```text
                                   northbound sample request
                         POST /v1/samples  GET /v1/samples/{id}  artifacts
                                                    |
                                                    v
  +--------------------------------------------------------------------------------------+
  | backend-owned serving entry                                                          |
  | wan-video request normalization | genie-rollout request normalization                |
  | explicit backend knobs | execution-family key | runtime-fit checks                   |
  +--------------------------------------------------------------------------------------+
                                                    |
                                                    v
  +--------------------------------------------------------------------------------------+
  | low-level serving runtime                                                            |
  | admission for runtime fit | shape-family batching | graph/engine cache               |
  | compiled profile reuse | stage executor | memory movement overlap | hot-state reuse  |
  +--------------------------------------------------------------------------------------+
           |                              |                                |
           v                              v                                v
  +----------------------+   +---------------------------+    +---------------------------+
  | model stage workers  |   | memory/runtime substrate  |    | observability            |
  | text/prompt encode   |   | cuda graphs               |    | stage latency            |
  | transition/diffusion |   | TensorRT/torch compile    |    | transfer latency         |
  | decode/postprocess   |   | stream management         |    | cache hit rate           |
  | persist handoff      |   | h2d/d2h overlap           |    | artifact IO latency      |
  +----------------------+   | activation/state reuse    |    | gpu occupancy            |
                             +---------------------------+    +---------------------------+

note

- control plane still exists and remains durable truth
- but in this document it is treated as a constraint around serving, not the main optimization target
```

## 2. 当前代码与目标分层映射

```text
today

wm_infra/api/
  FastAPI surface, protocol, metrics, app wiring

wm_infra/controlplane/
  sample manifests, temporal entities, storage, resource estimation

wm_infra/backends/
  wan.py / wan_runtime.py / wan_engine.py
  genie.py / genie_runtime.py / genie_scheduler.py / genie_batcher.py
  rollout.py / registry.py / job_queue.py

wm_infra/core/
  engine.py / scheduler.py / state.py / execution.py

gap summary

1. low-level execution families are not yet the primary organizing boundary everywhere
2. Genie has a stage-runtime skeleton, but the serving fast path is still transition-heavy rather than stage-complete
3. Wan has a real backend path, but engine cache / graph reuse / data plane are still partial
4. rollout-engine is still more of a substrate skeleton than a true high-performance serving core
5. memory movement, graph cache, and compiled-profile reuse are not yet first-class enough across the repo
6. observability exists, but not yet at the level required for strong low-level serving claims
```

## 3. 低层 serving 目标架构和现状差距

### 3.1 先明确不聚焦什么

当前不优先的方向：

- 更高层的队列策略
- 更宽的多租户 policy
- review / eval / export
- 泛化的 orchestration 讨论
- 过早把 repo 抽象成统一的 multi-backend meta-platform

这些内容不是不重要，而是现在不该吃掉主要注意力。

当前阶段更值得聚焦的是：

- stage executor 是否真正高效
- 图编译/engine cache 是否真的复用
- batch family 是否正确
- memory transfer 是否压到最低
- kernel/runtime boundary 是否稳定
- 端到端 latency 到底被哪个 stage 吃掉

### 3.2 低层 serving 的核心对象

目标：

- backend request 先归一化成 execution family
- runtime 主要围绕 execution family、stage worker、compiled profile、memory residency 工作
- 极端性能优化首先发生在这些对象上，而不是高层 queue policy 上

建议的低层 serving 对象：

- `ExecutionFamily`
- `CompiledProfile`
- `StageWorker`
- `TransferPlan`
- `ResidencyRecord`
- `StageTiming`

推荐形状：

```text
ExecutionFamily
  backend
  model
  stage
  dtype
  device
  width
  height
  frame_count
  num_steps
  prompt_frames
  tokenizer/layout key
  memory mode

CompiledProfile
  family key
  graph key
  batch sizes seen
  warm/cold
  compile latency
  last used

TransferPlan
  input bytes
  state bytes
  checkpoint bytes
  artifact bytes
  overlapable or not
```

现状：

- `WanBatchSignature` 和 warmed profile pool 已经接近这个方向
- `GenieBatchSignature`、`GenieExecutionChunk`、`GenieRuntimeState` 也已经接近这个方向
- 但仓库还没有在所有关键 fast path 上都用这些对象做第一组织原则

差距：

- Wan 和 Genie 还没有形成共享的“低层 serving primitive vocabulary”
- compiled profile / graph reuse / transfer plan 还没有被统一而明确地表达
- 低层 serving 的关键 cache hit / miss / compile state 还没有在所有路径稳定暴露

### 3.3 `wan-video` 低层 serving 目标面与差距

目标：

- `wan-video` 先成为强低层 video serving runtime
- 重点不在高层 queue policy，而在 shape-family execution、graph reuse、memory/runtime efficiency
- 先把视频 serving 的实际执行成本拆干净：encoder、diffusion、decode、postprocess、artifact IO

现状：

- 已有真实的 `wan-video` backend
- 已有 `WanTaskConfig`、resource estimator、queue compatibility、warm profile pool、quality-cost hints
- 已经有显式 stage 序列：`text_encode -> diffusion -> vae_decode -> safety -> postprocess -> persist`

差距：

- 还缺少更明确的 `ExecutionFamily -> CompiledProfile -> StageWorker` 路径
- text encoder / diffusion / VAE decode 之间还缺少更低开销的 handoff 表达
- graph cache、compiled batch-size family、CUDA graph capture 仍然更像未来空间，不是已经站稳的 serving core
- 对 artifact IO、host-device transfer、decode/postprocess 的代价剖分还不够细

重构图：

```text
today
sample request
  -> wan request normalization
  -> warm-profile helper
  -> in-process official adapter
  -> text_encode / diffusion / vae_decode / safety / postprocess / persist
  -> sample manifest

target
sample request
  -> wan execution-family key
  -> compiled profile lookup
  -> graph/engine cache selection
  -> stage workers with low-overhead handoff
  -> optional data plane: ingest / decode / preprocess
  -> text_encode / diffusion / vae_decode / safety / postprocess
  -> artifact persist
  -> control-plane commit
```

具体差距：

- 缺少独立而明确的 execution-family abstraction
- 缺少 compiled profile / engine cache 生命周期管理
- 缺少更硬的 stage microbenchmark 与 transfer benchmark
- 缺少更明确的 video data-plane fast path

优先重构顺序：

1. 先补强 Wan stage profiling 和 encoder / diffusion / decode / artifact IO 观测
2. 再固化 execution family 和 compiled profile cache
3. 再拆清楚 data plane 与 model-execution plane
4. 最后才考虑更大的外部 engine 接入

### 3.4 `genie-rollout` 低层 serving 目标面与差距

目标：

- `genie-rollout` 先成为强低层 temporal serving runtime
- 重点放在 state-hot execution、window transition fast path、checkpoint cost control、transfer minimization
- 优先把 transition runtime、prompt/state reuse、checkpoint delta 这些低层 serving 成本打透

现状：

- `genie-rollout` 已经有 stage runtime skeleton
- 已有 execution entities、chunk scheduler、transition batcher、checkpoint helper、runtime profile
- control-plane temporal nouns 已经能支撑 Genie 路径

差距：

- transition fast path 之外的 stage 还不够“serving-engine-like”
- state materialize / prompt prepare / checkpoint / persist 之间的 low-level cost model 还不够硬
- residency、reuse、transfer 还没有成为最核心的 runtime primitive
- 当前仍偏 stage runtime skeleton，不是 fully hardened serving core

重构图：

```text
today
northbound sample
  -> validate temporal refs
  -> prepare inputs
  -> build transition entities
  -> schedule chunks
  -> transition batching
  -> persist outputs

target
northbound sample
  -> execution-family key
  -> state materialize
  -> prompt prepare
  -> transition window execution
  -> checkpoint delta build
  -> artifact persist handoff
  -> control-plane commit
```

具体差距：

- 还没有真正围绕 hot-state reuse 组织整个 runtime
- 还没有把 checkpoint delta / recovery cost 压成低层 serving primitive
- transition 之外的 stage 还没有形成同样强的 batchable execution boundary
- transfer latency、state materialize latency、artifact IO latency 还不够像第一诊断指标

优先重构顺序：

1. 先把 transition / materialize / checkpoint / persist 的 stage cost 观测打实
2. 再把 runtime primitive 收敛到 hot-state reuse、transfer plan、checkpoint delta
3. 再把 batching 从 transition 扩到其他真正占主成本的 stage
4. 最后才讨论更宽的 many-world scheduler 抽象

### 3.5 `rollout-engine` 低层 runtime 基座目标面与差距

目标：

- `rollout-engine` 是 runtime substrate
- 它为 temporal execution 提供低层 runtime primitive、bring-up、benchmark、实验底盘
- 它不抢占 `wan-video` 和 `genie-rollout` 的产品定位

现状：

- `wm_infra/core/` 里已经有 engine / scheduler / state / execution
- 它仍然是很多通用 runtime 概念的落点

差距：

- 当前 `core` 还不是一个极致低层 serving substrate
- execution grain 仍然偏逻辑 batch，而不是更硬的 stage-local work unit
- 对 graph reuse、transfer overlap、residency primitive 的沉淀还不够

重构动作：

1. 把 `core` 继续定位为 substrate，不把它包装成新的产品 backend
2. 先让 `Wan` 和 `Genie` 各自验证低层 serving primitive
3. 只有在两个主路径都证明有效时，再把 primitive 下沉到 `core`

### 3.6 外部 inference engine 的位置

目标：

- 外部 engine 是可插拔下层，不是仓库身份本身
- `vLLM` / `SGLang` / TensorRT 类系统只在某个 backend 的低层 serving bottleneck 已被证明时接入
- `wm-infra` 自己继续拥有 temporal schema、control plane、stateful lineage、sample semantics

现状：

- 这一层在战略和文档上已经明确
- 代码里还没有形成很强的 adapter boundary

差距：

- backend adapter 和 external engine adapter 的边界还不够显式
- 还没有形成“哪些 stage 需要外部 engine，哪些 stage 必须保留在 backend runtime 内”的稳定接口

重构动作：

1. 为未来 external engine 接入保留 adapter boundary，但不提前泛化
2. 只在 `wan-video` 或 `genie-rollout` 的实测瓶颈已证明需要时再接入
3. 不把 external engine 的接口形状反向污染产品 API

### 3.7 低层 serving observability

目标：

- stage latency、graph compile cost、cache hit rate、transfer latency、artifact IO latency、GPU 占用都可观测
- 所有低层 serving 判断都能回到证据，而不是 runtime 叙事

现状：

- 已有 benchmark harness
- 已有 Prometheus metrics
- 已有 Genie stage runtime 相关观测

差距：

- `TTFT`、`TPOT`、state transfer latency、artifact IO latency、compile latency、cache hit rate 还没有全部形成稳定基线
- Wan 路径的细粒度阶段观测仍弱于 Genie
- “指标改善对应哪个低层 serving 改动”这件事仍然容易混淆

重构动作：

1. 把 workload 基线固定在 `wan-video` 与 `genie-rollout`
2. 指标按 stage、execution family、compile/cache state 分解，而不是只看 end-to-end
3. 每次优化只打一个低层 bottleneck，避免 attribution 混乱

## 4. 最具体的低层重构优先级

### P0: serving fast path 叙事统一

- 统一对外叙事：主产品路径是 `wan-video` 和 `genie-rollout`
- `rollout-engine` 降为 substrate 叙事
- 所有新文档优先描述 low-level serving path，而不是更高层 orchestration

### P1: `wan-video` 低层 serving 强化

- 目标：把 Wan 从“真实 backend + helpers”推进到“强低层 video serving runtime”
- 交付：
  - stage-complete latency breakdown
  - execution-family key
  - compiled profile / graph reuse state
  - artifact IO 与 transfer cost 可观测
  - data-plane fast path 原型

### P1: `genie-rollout` 低层 serving 强化

- 目标：把 Genie 从“stage runtime skeleton”推进到“更硬的 temporal serving runtime”
- 交付：
  - state materialize / transition / checkpoint / persist 的成本闭环
  - hot-state reuse primitive
  - transfer plan primitive
  - checkpoint delta fast path

### P2: substrate primitive 下沉

- 目标：把已经证明有效的低层 primitive 下沉到 `core`
- 交付：
  - shared execution-family vocabulary
  - shared transfer/residency primitive
  - shared microbenchmark hooks

### P2: external engine adapter boundary

- 目标：给未来 TensorRT / vLLM / SGLang 等接入留明确边界
- 交付：
  - backend-owned low-level adapter contract
  - 不污染 northbound sample API 的 external engine binding

## 5. 定义“这次重构做对了”

下面这些变化同时成立，才算接近目标架构：

- `wan-video` 和 `genie-rollout` 的 backend 身份更清楚，而不是更模糊
- `rollout-engine` 更像 substrate，而不是更像产品入口
- 低层 serving primitive 更清楚，而不是 helper 越堆越多
- 运行时执行更多基于 stage-local、state-local、shape-local work
- 指标能证明低层瓶颈在哪里，优化改善了什么
- 外部 engine 若接入，只是下层能力，不重写仓库身份

## 6. 推荐的下一步文档动作

如果继续推进，可以按这个顺序补文档：

1. 在 `Wan` 路径上补一份 `WAN_EXECUTION_RUNTIME.md`
2. 在 `Genie` 路径上补一份“low-level serving gap closure”清单
3. 单独补一份 `SERVING_PRIMITIVES.md`，只记录 graph cache、compiled profile、transfer overlap、state reuse
4. 在 repo 顶层 roadmap 中把本文件作为架构总图入口
