"""Visual generation engine primitives."""

from vrl.engine.batching import forward_batch_by_merging_prompts
from vrl.engine.core.protocols import (
    BatchedFamilyPipelineExecutor,
    ChunkedFamilyPipelineExecutor,
    FamilyPipelineExecutor,
    PipelineChunkResult,
)
from vrl.engine.core.registry import ExecutorKey, FamilyPipelineRegistry
from vrl.engine.core.runtime import (
    GenerationRuntime,
    RolloutBackend,
)
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
from vrl.engine.gather import (
    gather_pipeline_chunks,
    require_chunked_executor,
)

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
    "forward_batch_by_merging_prompts",
    "gather_pipeline_chunks",
    "require_chunked_executor",
]
