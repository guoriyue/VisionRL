"""Control-plane schemas and interfaces for sample production.

This package deliberately sits above the runtime layer.
The runtime executes model workloads.
The control plane tracks what was requested, what was produced,
and how outputs can flow into evaluation and training.
"""

from .schemas import (
    ArtifactKind,
    ArtifactRecord,
    EvaluationRecord,
    EvaluationStatus,
    ExperimentRef,
    FailureTag,
    ProduceSampleRequest,
    SampleRecord,
    SampleSpec,
    SampleStatus,
    TaskType,
    TrainingExportRecord,
)

__all__ = [
    "ArtifactKind",
    "ArtifactRecord",
    "EvaluationRecord",
    "EvaluationStatus",
    "ExperimentRef",
    "FailureTag",
    "ProduceSampleRequest",
    "SampleRecord",
    "SampleSpec",
    "SampleStatus",
    "TaskType",
    "TrainingExportRecord",
]
