"""SGLang-style generation runtime."""

from vrl.engine.generation.batching import forward_batch_by_merging_prompts
from vrl.engine.generation.factory import build_local_generation_runtime
from vrl.engine.generation.gather import (
    gather_pipeline_chunks,
    require_chunked_executor,
)
from vrl.engine.generation.local_worker_pool import (
    LocalRolloutWorker,
    LocalRolloutWorkerPool,
    LocalWorkerSpec,
)
from vrl.engine.generation.protocols import (
    BatchedFamilyPipelineExecutor,
    ChunkedFamilyPipelineExecutor,
    FamilyPipelineExecutor,
    PipelineChunkResult,
)
from vrl.engine.generation.registry import ExecutorKey, FamilyPipelineRegistry
from vrl.engine.generation.runtime import (
    GenerationRuntime,
    RolloutBackend,
)
from vrl.engine.generation.types import (
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
from vrl.engine.generation.worker import GenerationIdFactory, GenerationWorker

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
    "GenerationSampleSpec",
    "GenerationWorker",
    "LocalRolloutWorker",
    "LocalRolloutWorkerPool",
    "LocalWorkerSpec",
    "OutputBatch",
    "PipelineChunkResult",
    "RolloutBackend",
    "RolloutDebugTensors",
    "RolloutDenoisingEnv",
    "RolloutDitTrajectory",
    "RolloutTrajectoryData",
    "WorkloadSignature",
    "build_local_generation_runtime",
    "forward_batch_by_merging_prompts",
    "gather_pipeline_chunks",
    "require_chunked_executor",
]
