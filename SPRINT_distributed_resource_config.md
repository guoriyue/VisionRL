# SPRINT: Ray Distributed Resource Configuration

## 0. Core Decision

本 sprint 只支持一套 canonical schema：显式 role allocation。

不采用 `mode + num_gpus + trainer_gpus` 作为主配置，因为它会变成第二套 shorthand，后面和 `trainer/rollout/reward/ref` 这类 role 配置冲突。`split` / `colocate` 不作为用户必须手写的 primary knob，而是从 `trainer.devices` 和 `rollout.devices` 是否重叠推导出来。

目标配置：

```yaml
distributed:
  backend: ray

  resources:
    visible_devices: auto

    trainer:
      num_gpus: 1
      devices: auto

    rollout:
      num_gpus: auto
      gpus_per_worker: 1
      num_workers: auto
      devices: auto

    allow_overlap: false
```

高级手动 pinning：

```yaml
distributed:
  backend: ray

  resources:
    visible_devices: [0, 1, 2, 3]

    trainer:
      devices: [0]

    rollout:
      devices: [1, 2, 3]
      gpus_per_worker: 1
      num_workers: auto

    allow_overlap: false
```

单 GPU colocate smoke：

```yaml
distributed:
  backend: ray

  resources:
    visible_devices: auto

    trainer:
      devices: [0]

    rollout:
      devices: [0]
      gpus_per_worker: 1
      num_workers: 1

    allow_overlap: true

  rollout:
    release_after_collect: true
```

## 1. Why This Shape

这个设计跟现代 RL serving/training 框架的资源抽象一致：

- 用户默认声明 role 需要多少资源，不默认写物理 GPU 编号。
- Ray placement group 负责 actor placement。
- 物理 GPU pinning 是高级 override，用于单机调试、混部、或明确避开某些卡。
- 模型配置不包含 GPU placement。`model` 只描述 checkpoint、dtype、LoRA、backend；GPU 分配属于 `distributed.resources`。

`slime` 的主语义是 `actor_num_gpus_per_node`、`rollout_num_gpus`、`rollout_num_gpus_per_engine`、`colocate`。本 repo 不直接复制它的 CLI 形状，但保留同一个资源边界：trainer 和 rollout 是两个 role，Ray placement 负责切资源。

## 2. Required Semantics

### 2.1 `visible_devices`

`visible_devices` 定义本次训练允许使用的 GPU budget。

规则：

- `auto`：读取当前进程 / Ray cluster 可见 CUDA devices。
- `[]`：CPU-only，Ray rollout worker 必须 `gpus_per_worker=0`。
- 显式 list：所有 role 的 `devices` 必须是它的子集。
- 不在这里做 free-memory 预测。显存动态变化，只能做 static resource ownership 和 OOM retry。

### 2.2 `trainer`

`trainer` 定义 driver/training side 的 GPU ownership。

规则：

- `trainer.devices: auto` 时，默认取 `visible_devices` 的前 `trainer.num_gpus` 张。
- `trainer.num_gpus` 只在 `devices: auto` 时生效。
- 如果 `trainer.devices` 显式设置，`trainer.num_gpus` 必须为空或等于 `len(devices)`。
- 当前阶段只支持 single-process trainer，因此 `len(trainer.devices)` 必须是 `0` 或 `1`。
- 后续 FSDP / RayTrainGroup 阶段再放开 `trainer.devices` 多卡。

### 2.3 `rollout`

`rollout` 定义 Ray rollout workers 的 GPU budget。

规则：

- `rollout.devices: auto` 且 `allow_overlap=false`：默认使用 `visible_devices - trainer.devices`。
- `rollout.devices: auto` 且 `allow_overlap=true`：如果没有剩余 GPU，可以复用 `trainer.devices`。
- `rollout.num_gpus: auto`：等于 `len(rollout.devices)`。
- `rollout.num_workers: auto`：等于 `rollout.num_gpus / gpus_per_worker`，必须整除。
- `gpus_per_worker` 支持 `0`、`1`，后续多 GPU per worker 再支持 `>1`。
- `rollout.devices` 为空且 `gpus_per_worker > 0` 必须 fail-fast。

### 2.4 overlap policy

默认不允许 trainer 和 rollout overlap。

规则：

- 如果 `trainer.devices ∩ rollout.devices != ∅` 且 `allow_overlap=false`，启动时报错。
- 如果 overlap 被允许，必须同时满足 `distributed.rollout.release_after_collect=true`，否则单 GPU/混部路径会保留两份模型，OOM 风险不可控。
- overlap 只用于 smoke/debug，不作为 throughput path。

## 3. Target Internal Types

新增文件：

```text
vrl/distributed/resources.py
```

目标 dataclass：

```python
@dataclass(frozen=True, slots=True)
class RoleResourceConfig:
    num_gpus: int | str | None = "auto"
    devices: list[int] | str = "auto"


@dataclass(frozen=True, slots=True)
class RolloutResourceConfig(RoleResourceConfig):
    gpus_per_worker: float = 1.0
    num_workers: int | str = "auto"


@dataclass(frozen=True, slots=True)
class DistributedResourceConfig:
    visible_devices: list[int] | str = "auto"
    trainer: RoleResourceConfig = field(default_factory=RoleResourceConfig)
    rollout: RolloutResourceConfig = field(default_factory=RolloutResourceConfig)
    allow_overlap: bool = False


@dataclass(frozen=True, slots=True)
class ResolvedDistributedResources:
    visible_devices: tuple[int, ...]
    trainer_devices: tuple[int, ...]
    rollout_devices: tuple[int, ...]
    rollout_num_workers: int
    rollout_gpus_per_worker: float
    colocated: bool
```

核心函数：

```python
def resolve_distributed_resources(cfg: Any) -> ResolvedDistributedResources:
    ...
```

`resolve_distributed_resources()` 是唯一入口。不要让 train script、Ray launcher、runtime backend 各自解析 GPU config。

## 4. Code Changes

### Phase 1: Config Schema

修改：

```text
configs/base/distributed/ray_rollout.yaml
configs/base/distributed/ray_rollout_single_gpu.yaml
vrl/rollouts/runtime/config.py
```

目标：

- `ray_rollout.yaml` 使用 `distributed.resources`，默认多 GPU split。
- `ray_rollout_single_gpu.yaml` 使用 explicit overlap，并设置 `release_after_collect=true`。
- `RolloutBackendConfig` 继续保留 rollout execution 参数，但不再负责 trainer/rollout device allocation。

`RolloutBackendConfig` 应保留：

```yaml
distributed:
  rollout:
    cpus_per_worker: 4.0
    placement_strategy: SPREAD
    max_inflight_chunks_per_worker: 1
    sync_trainable_state: lora_only
    release_after_collect: false
```

`RolloutBackendConfig` 不应继续拥有：

```yaml
num_workers
gpus_per_worker
allow_driver_gpu_overlap
```

这些字段由 `ResolvedDistributedResources` 统一提供。

### Phase 2: Resource Resolver

新增：

```text
vrl/distributed/resources.py
tests/distributed/test_resources.py
```

测试覆盖：

- 4 visible GPUs, trainer auto 1, rollout auto -> trainer `(0,)`, rollout `(1,2,3)`, workers `3`。
- explicit trainer `[0]`, rollout `[1,2,3]` -> no overlap。
- explicit overlap `[0]` / `[0]` with `allow_overlap=false` -> fail。
- explicit overlap `[0]` / `[0]` with `allow_overlap=true` -> colocated true。
- explicit devices not subset of visible -> fail。
- `num_workers: auto` with non-divisible `num_gpus / gpus_per_worker` -> fail。
- single GPU auto split with `allow_overlap=false` -> fail with clear message。
- single GPU overlap with `release_after_collect=false` -> fail at backend validation。

### Phase 3: Ray Placement Integration

修改：

```text
vrl/distributed/ray/placement/group.py
vrl/distributed/ray/rollout/launcher.py
vrl/distributed/ray/rollout/types.py
```

目标：

- `RayRolloutLauncher.launch()` 接收 `ResolvedDistributedResources` 或从 cfg 内部 resolve。
- placement group bundle 数量来自 `resolved.rollout_num_workers`。
- actor `num_gpus` 来自 `resolved.rollout_gpus_per_worker`。
- worker metadata 记录 assigned Ray GPU IDs，并和 resolved rollout devices 做一致性检查。

注意：

- Ray 不能直接保证“物理 GPU1-3”除非它的 cluster 可见资源和 placement group 对齐。
- 第一版必须在日志里打印 resolved plan 和 Ray 实际 assigned GPU IDs。
- 如果 Ray 返回的 GPU IDs 不在 `rollout_devices` 内，启动失败，不 silent fallback。

### Phase 4: Driver CUDA Ownership Validation

修改：

```text
vrl/rollouts/runtime/backend.py
vrl/scripts/sd3_5/train.py
vrl/scripts/wan_2_1/train.py
vrl/scripts/cosmos/train.py
vrl/scripts/janus_pro/train.py
vrl/scripts/nextstep_1/train.py
```

目标：

- train script 不再直接写 `torch.device("cuda" if torch.cuda.is_available() else "cpu")`。
- train script 调用 `resolve_distributed_resources(cfg)`，并把 `trainer_devices[0]` 转成 driver device。
- backend validation 从“driver 是否在 CUDA”升级成“driver CUDA device 是否和 rollout devices overlap”。
- overlap 时必须检查 `allow_overlap=true` 和 `release_after_collect=true`。

错误信息必须具体：

```text
Trainer device cuda:0 overlaps rollout devices [0], but resources.allow_overlap=false.
Use resources.rollout.devices=[1,2,3] for split mode, or set allow_overlap=true with rollout.release_after_collect=true for single-GPU smoke.
```

### Phase 5: README and Examples

修改：

```text
README.md
configs/base/distributed/ray_rollout.yaml
configs/base/distributed/ray_rollout_single_gpu.yaml
```

README 需要新增两个例子：

自动 split：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vrl.scripts.train \
  --config experiment/sd3_5_ocr_grpo \
  distributed.resources.trainer.num_gpus=1 \
  distributed.resources.rollout.num_gpus=auto
```

显式 pinning：

```bash
python -m vrl.scripts.train \
  --config experiment/sd3_5_ocr_grpo \
  distributed.resources.visible_devices='[0,1,2,3]' \
  distributed.resources.trainer.devices='[0]' \
  distributed.resources.rollout.devices='[1,2,3]'
```

文档必须明确：

- 默认推荐 auto split。
- manual pinning 是高级选项。
- 单 GPU colocate 只用于 smoke/debug。
- throughput path 是 trainer GPU(s) 和 rollout GPU(s) 分离。

## 5. Non-Goals

本 sprint 不做：

- FSDP trainer 多卡。
- RayTrainGroup 接入主训练入口。
- reward model serving。
- multi-node hostname / rack-aware placement。
- 基于实时 free memory 的动态 GPU 选择。
- SGLang server 复用。
- per-token AR scheduling。

这些都依赖更高层的 distributed training / serving 设计，不应该塞进本 sprint。

## 6. Acceptance Criteria

代码层：

- 所有训练脚本都通过同一个 `resolve_distributed_resources()` 选择 trainer device。
- `RolloutBackendConfig` 不再解析 `num_workers`、`gpus_per_worker`、`allow_driver_gpu_overlap`。
- Ray launcher 不再从 rollout config 猜 worker 数，而是使用 resolved resources。
- driver / rollout device overlap 有明确 fail-fast。
- 单 GPU smoke 必须显式 `allow_overlap=true` 和 `release_after_collect=true`。

测试层：

```bash
python -m pytest -q tests/distributed/test_resources.py
python -m pytest -q tests/distributed/ray
python -m pytest -q tests/config/test_load_all_experiments.py
python -m pytest -q tests/scripts
```

文档层：

- README 有 auto split 和 manual pinning 示例。
- `configs/base/distributed/ray_rollout.yaml` 表达多 GPU split。
- `configs/base/distributed/ray_rollout_single_gpu.yaml` 表达 colocate smoke。

真实 checkpoint DoD：

- 在至少一个 diffusion family 上用真实 checkpoint 跑一次 single-GPU colocate smoke。
- 在至少一个 family 上用 4 visible GPUs 跑一次 resolved plan smoke，确认日志显示 trainer `cuda:0`，rollout workers 分配到 `1,2,3`。
- 如果没有 4 GPU 环境，必须保留 skip 标记和清晰的手动命令，不把 fake Ray test 当成真实 DoD。

## 7. Expected Final Behavior

4 GPU 默认行为：

```text
visible_devices = [0, 1, 2, 3]
trainer.num_gpus = 1
rollout.num_gpus = auto

resolved:
  trainer_devices = [0]
  rollout_devices = [1, 2, 3]
  rollout_num_workers = 3
  colocated = false
```

单 GPU smoke：

```text
visible_devices = [0]
trainer.devices = [0]
rollout.devices = [0]
allow_overlap = true
release_after_collect = true

resolved:
  trainer_devices = [0]
  rollout_devices = [0]
  rollout_num_workers = 1
  colocated = true
```

错误配置：

```yaml
distributed:
  resources:
    visible_devices: [0]
    trainer:
      devices: [0]
    rollout:
      devices: [0]
    allow_overlap: false
```

必须失败：

```text
Trainer devices [0] overlap rollout devices [0], but resources.allow_overlap=false.
```

## 8. References

Local references:

- `/home/mingfeiguo/Desktop/wm-infra/vrl/rollouts/runtime/config.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/rollouts/runtime/backend.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/distributed/ray/placement/group.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/distributed/ray/rollout/launcher.py`
- `/home/mingfeiguo/Desktop/wm-infra/configs/base/distributed/ray_rollout.yaml`
- `/home/mingfeiguo/Desktop/wm-infra/configs/base/distributed/ray_rollout_single_gpu.yaml`

Architecture references:

- `/home/mingfeiguo/Desktop/slime/slime/utils/arguments.py`
- `/home/mingfeiguo/Desktop/slime/slime/ray/placement_group.py`
- `/home/mingfeiguo/Desktop/slime/slime/ray/rollout.py`
- `/home/mingfeiguo/Desktop/slime/slime/backends/sglang_utils/sglang_engine.py`
- `/home/mingfeiguo/Desktop/sglang/python/sglang/srt/server_args.py`
