"""FSDP utilities for distributed training.

Ported from flow_grpo/fsdp_utils.py.  Provides a config + wrapper for
PyTorch FSDP, activation checkpointing, and optimizer CPU offloading.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


@dataclass(slots=True)
class FSDPConfig:
    """Configuration for FSDP wrapping."""

    sharding_strategy: str = "FULL_SHARD"
    backward_prefetch: str = "BACKWARD_PRE"
    cpu_offload: bool = False
    num_replicate: int = 1
    num_shard: int = 8
    mixed_precision_dtype: Any = torch.bfloat16
    use_activation_checkpointing: bool = True
    use_device_mesh: bool = False


def fsdp_wrapper(
    model: torch.nn.Module,
    fsdp_config: FSDPConfig,
    get_transformer_layer_cls: Callable[[], list[type]],
    ignored_modules: list[torch.nn.Module] | None = None,
) -> FSDP:
    """Wrap a model with FSDP, mixed precision, and optional activation checkpointing."""
    if ignored_modules is None:
        ignored_modules = []

    device_mesh = None
    if fsdp_config.sharding_strategy == "HYBRID_SHARD" and fsdp_config.use_device_mesh:
        from torch.distributed.device_mesh import init_device_mesh

        device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
            mesh_dim_names=("replicate", "shard"),
        )

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=get_transformer_layer_cls(),
        ),
        ignored_modules=ignored_modules,
        mixed_precision=MixedPrecision(
            param_dtype=fsdp_config.mixed_precision_dtype,
            reduce_dtype=fsdp_config.mixed_precision_dtype,
            buffer_dtype=fsdp_config.mixed_precision_dtype,
        ),
        device_id=dist.get_rank() % torch.cuda.device_count(),
        sharding_strategy=ShardingStrategy[fsdp_config.sharding_strategy],
        backward_prefetch=BackwardPrefetch[fsdp_config.backward_prefetch],
        cpu_offload=CPUOffload(offload_params=fsdp_config.cpu_offload),
        device_mesh=device_mesh,
        use_orig_params=True,
    )

    if fsdp_config.use_activation_checkpointing:
        layer_cls = tuple(get_transformer_layer_cls())

        def _check_fn(module: torch.nn.Module) -> bool:
            return isinstance(module, layer_cls)

        apply_activation_checkpointing(
            fsdp_model,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            ),
            check_fn=_check_fn,
        )

    return fsdp_model


def save_fsdp_checkpoint(
    save_dir: str,
    model: FSDP,
    global_step: int,
    rank: int,
) -> None:
    """Save a full FSDP state dict as safetensors (rank 0 only)."""
    from safetensors.torch import save_file

    save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
    os.makedirs(save_path, exist_ok=True)

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
    ):
        state_dict = model.state_dict()
        if rank == 0:
            save_file(state_dict, os.path.join(save_path, "model.safetensors"))
        del state_dict

    dist.barrier()


class OptimizerOffloadHook:
    """CPU-offloads optimizer states between steps to save GPU memory.

    Saves ~50% GPU memory for Adam-family optimizers.  Register via
    ``register_optimizer_offload_hooks(optimizer)``.
    """

    def __init__(self) -> None:
        self.cpu_states: dict[torch.nn.Parameter, dict[str, torch.Tensor]] = {}

    def pre_step_hook(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param in optimizer.state and param in self.cpu_states:
                    state = optimizer.state[param]
                    for key, cpu_tensor in self.cpu_states[param].items():
                        state[key] = cpu_tensor.to(param.device, non_blocking=True)
                    del self.cpu_states[param]

    def post_step_hook(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        for group in optimizer.param_groups:
            for param in group["params"]:
                if optimizer.state.get(param):
                    state = optimizer.state[param]
                    self.cpu_states[param] = {}
                    for key, val in state.items():
                        if isinstance(val, torch.Tensor):
                            self.cpu_states[param][key] = val.to("cpu", non_blocking=True)
                            state[key] = torch.empty(0, device=param.device)


def register_optimizer_offload_hooks(
    optimizer: torch.optim.Optimizer,
) -> tuple[list[Any], OptimizerOffloadHook]:
    """Register pre/post step hooks for CPU offloading of optimizer states."""
    hook = OptimizerOffloadHook()
    pre_handle = optimizer.register_step_pre_hook(hook.pre_step_hook)
    post_handle = optimizer.register_step_post_hook(hook.post_step_hook)
    return [pre_handle, post_handle], hook


def init_distributed() -> tuple[bool, int, int, int]:
    """Initialize torch.distributed from environment variables."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        return False, 0, 1, 0

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return True, rank, world_size, local_rank
