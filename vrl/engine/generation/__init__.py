"""SGLang-style generation runtime."""

from vrl.engine.generation.gather import (
    gather_pipeline_chunks,
    require_chunked_executor,
)
from vrl.engine.generation.local_worker_pool import (
    LocalRolloutWorker,
    LocalRolloutWorkerPool,
    LocalWorkerSpec,
)
from vrl.engine.generation.registry import ExecutorKey, FamilyPipelineRegistry
from vrl.engine.generation.runtime import (
    GenerationBatchPlanner,
    GenerationModelRunner,
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
    "ExecutorKey",
    "FamilyPipelineRegistry",
    "GenerationBatchPlanner",
    "GenerationIdFactory",
    "GenerationMetrics",
    "GenerationModelRunner",
    "GenerationRequest",
    "GenerationRuntime",
    "GenerationSampleSpec",
    "GenerationWorker",
    "LocalRolloutWorker",
    "LocalRolloutWorkerPool",
    "LocalWorkerSpec",
    "OutputBatch",
    "RolloutBackend",
    "RolloutDebugTensors",
    "RolloutDenoisingEnv",
    "RolloutDitTrajectory",
    "RolloutTrajectoryData",
    "WorkloadSignature",
    "gather_pipeline_chunks",
    "require_chunked_executor",
]
