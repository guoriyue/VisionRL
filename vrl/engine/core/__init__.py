"""Core engine contracts and runtime objects."""

from vrl.engine.core.protocols import (
    BatchedFamilyPipelineExecutor,
    ChunkedFamilyPipelineExecutor,
    FamilyPipelineExecutor,
    PipelineChunkResult,
)
from vrl.engine.core.registry import ExecutorKey, FamilyPipelineRegistry
from vrl.engine.core.runtime import GenerationRuntime, RolloutBackend
from vrl.engine.core.runtime_spec import GenerationRuntimeSpec
from vrl.engine.core.types import (
    GenerationMetrics,
    GenerationRequest,
    GenerationSampleSpec,
    OutputBatch,
    RolloutDebugTensors,
    RolloutDenoisingEnv,
    RolloutDitTrajectory,
    RolloutTrajectoryData,
    WorkloadSignature,
)
from vrl.engine.core.worker import GenerationIdFactory, GenerationWorker

__all__ = [
    "BatchedFamilyPipelineExecutor",
    "ChunkedFamilyPipelineExecutor",
    "ExecutorKey",
    "FamilyPipelineExecutor",
    "FamilyPipelineRegistry",
    "GenerationIdFactory",
    "GenerationMetrics",
    "GenerationRequest",
    "GenerationRuntime",
    "GenerationRuntimeSpec",
    "GenerationSampleSpec",
    "GenerationWorker",
    "OutputBatch",
    "PipelineChunkResult",
    "RolloutBackend",
    "RolloutDebugTensors",
    "RolloutDenoisingEnv",
    "RolloutDitTrajectory",
    "RolloutTrajectoryData",
    "WorkloadSignature",
]
