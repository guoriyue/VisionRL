"""Unified runtime engine for world-model inference.

Public API re-exports for the engine module.
"""

from wm_infra.engine._types import (
    EngineRunConfig,
    EntityRequest,
    Phase,
    SchedulerOutput,
    StepResult,
    SwapHandle,
)
from wm_infra.engine.loop import EngineLoop
from wm_infra.engine.pipeline.stage import (
    DynamicsStage,
    EncodeStage,
    StageRunner,
    StageSpec,
)
from wm_infra.engine.pipeline.task_graph import TaskEdge, TaskGraph, TaskNode
from wm_infra.engine.scheduler import ContinuousBatchingScheduler, EntityState
from wm_infra.engine.state.paged_pool import PagedLatentPool, PageTable
from wm_infra.engine.state.radix_cache import RadixNode, RadixStateCache
from wm_infra.engine.workers.queues import AsyncQueue, RequestQueue, ResultQueue
from wm_infra.engine.workers.worker import Worker

# Rollout engine (high-level world model engine API)
from wm_infra.engine.rollout import (
    AsyncWorldModelEngine,
    DEFAULT_FRAME_COUNT,
    DEFAULT_HEIGHT,
    DEFAULT_RESOURCE_UNITS_PER_GB,
    DEFAULT_WIDTH,
    HIGH_QUALITY_MEMORY_MULTIPLIER,
    LOW_VRAM_MEMORY_MULTIPLIER,
    LatentStateManager,
    RolloutJob,
    RolloutRequest,
    RolloutResult,
    RolloutScheduler,
    RolloutState,
    ScheduledBatch,
    WorldModelEngine,
)

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
