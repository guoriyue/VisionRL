"""Unified runtime engine for world-model inference.

Public API re-exports for the engine module.
"""

from wm_infra.engine.types import (
    EngineRunConfig,
    EntityRequest,
    Phase,
    SchedulerOutput,
    StepResult,
    SwapHandle,
)
from wm_infra.engine.managers.engine_loop import EngineLoop
from wm_infra.engine.engine import create_async_engine
from wm_infra.engine.model_executor.worker import (
    DynamicsStage,
    EncodeStage,
    StageRunner,
    StageSpec,
)
from wm_infra.engine.model_executor.task_graph import TaskEdge, TaskGraph, TaskNode
from wm_infra.engine.managers.scheduler import ContinuousBatchingScheduler, EntityState
from wm_infra.engine.mem_cache.paged_pool import PagedLatentPool, PageTable
from wm_infra.engine.mem_cache.radix_cache import RadixNode, RadixStateCache
from wm_infra.engine.model_executor.worker import AsyncQueue, RequestQueue, ResultQueue
from wm_infra.engine.model_executor.worker import Worker

# Rollout engine (high-level world model engine API)
from wm_infra.engine.engine import (
    AsyncWorldModelEngine,
    WorldModelEngine,
)
from wm_infra.engine.types import (
    DEFAULT_FRAME_COUNT,
    DEFAULT_HEIGHT,
    DEFAULT_RESOURCE_UNITS_PER_GB,
    DEFAULT_WIDTH,
    HIGH_QUALITY_MEMORY_MULTIPLIER,
    LOW_VRAM_MEMORY_MULTIPLIER,
    RolloutJob,
    RolloutRequest,
    RolloutResult,
    RolloutState,
    ScheduledBatch,
)
from wm_infra.engine.managers.scheduler import RolloutScheduler
from wm_infra.engine.managers.state_manager import LatentStateManager

__all__ = [
    # Types
    "EngineRunConfig",
    "EntityRequest",
    "Phase",
    "SchedulerOutput",
    "StepResult",
    "SwapHandle",
    # Loop
    "EngineLoop",
    "create_async_engine",
    # Scheduler
    "ContinuousBatchingScheduler",
    "EntityState",
    # State
    "PagedLatentPool",
    "PageTable",
    "RadixNode",
    "RadixStateCache",
    # Pipeline
    "DynamicsStage",
    "EncodeStage",
    "StageRunner",
    "StageSpec",
    "TaskEdge",
    "TaskGraph",
    "TaskNode",
    # Workers
    "AsyncQueue",
    "RequestQueue",
    "ResultQueue",
    "Worker",
    # Rollout engine
    "AsyncWorldModelEngine",
    "DEFAULT_FRAME_COUNT",
    "DEFAULT_HEIGHT",
    "DEFAULT_RESOURCE_UNITS_PER_GB",
    "DEFAULT_WIDTH",
    "HIGH_QUALITY_MEMORY_MULTIPLIER",
    "LOW_VRAM_MEMORY_MULTIPLIER",
    "LatentStateManager",
    "RolloutJob",
    "RolloutRequest",
    "RolloutResult",
    "RolloutScheduler",
    "RolloutState",
    "ScheduledBatch",
    "WorldModelEngine",
]
