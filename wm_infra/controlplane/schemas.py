"""Schema-first control-plane models for video data production.

These models are intentionally higher level than the low-level rollout API.
They define the entities needed to turn inference into reproducible sample
production, evaluation, and export.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    WORLD_MODEL_ROLLOUT = "world_model_rollout"
    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO = "image_to_video"
    VIDEO_TO_VIDEO = "video_to_video"
    POSTPROCESS = "postprocess"


class SampleStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    ACCEPTED = "accepted"


class EvaluationStatus(str, Enum):
    PENDING = "pending"
    AUTO_PASSED = "auto_passed"
    AUTO_FAILED = "auto_failed"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    HUMAN_ACCEPTED = "human_accepted"
    HUMAN_REJECTED = "human_rejected"


class ArtifactKind(str, Enum):
    VIDEO = "video"
    IMAGE = "image"
    LATENT = "latent"
    THUMBNAIL = "thumbnail"
    METADATA = "metadata"
    LOG = "log"
    EMBEDDING = "embedding"


class FailureTag(str, Enum):
    IDENTITY_DRIFT = "identity_drift"
    TEMPORAL_FLICKER = "temporal_flicker"
    PROMPT_MISMATCH = "prompt_mismatch"
    CAMERA_INSTABILITY = "camera_instability"
    UNSAFE_CONTENT = "unsafe_content"
    DECODE_FAILURE = "decode_failure"
    LOW_MOTION_QUALITY = "low_motion_quality"
    BROKEN_PHYSICS = "broken_physics"
    UNKNOWN = "unknown"


class ExperimentRef(BaseModel):
    experiment_id: str = Field(..., description="Stable experiment identifier")
    run_id: Optional[str] = Field(default=None, description="Specific run within experiment")
    tags: list[str] = Field(default_factory=list)


class SampleSpec(BaseModel):
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    duration_seconds: Optional[float] = Field(default=None, ge=0)
    fps: Optional[int] = Field(default=None, ge=1)
    width: Optional[int] = Field(default=None, ge=1)
    height: Optional[int] = Field(default=None, ge=1)
    seed: Optional[int] = None
    references: list[str] = Field(default_factory=list, description="URIs or asset IDs")
    controls: dict[str, Any] = Field(default_factory=dict, description="Camera/motion/control inputs")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactRecord(BaseModel):
    artifact_id: str
    kind: ArtifactKind
    uri: str
    mime_type: Optional[str] = None
    bytes: Optional[int] = None
    sha256: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationRecord(BaseModel):
    evaluator: str = Field(..., description="Name of auto scorer or human review queue")
    status: EvaluationStatus = EvaluationStatus.PENDING
    score: Optional[float] = None
    failure_tags: list[FailureTag] = Field(default_factory=list)
    notes: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainingExportRecord(BaseModel):
    export_id: str
    export_format: str = Field(..., description="pairwise_ranking | scorer_training | lora_finetune_manifest")
    dataset_name: Optional[str] = None
    split: Optional[str] = None
    uri: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProduceSampleRequest(BaseModel):
    task_type: TaskType
    backend: str = Field(..., description="Backend/runtime identifier")
    model: str = Field(..., description="Logical model identifier")
    model_revision: Optional[str] = None
    experiment: Optional[ExperimentRef] = None
    sample_spec: SampleSpec
    return_artifacts: list[ArtifactKind] = Field(default_factory=lambda: [ArtifactKind.VIDEO])
    evaluation_policy: Optional[str] = Field(default=None, description="Policy name for auto-QC / review")
    priority: float = 0.0
    labels: dict[str, str] = Field(default_factory=dict)


class SampleRecord(BaseModel):
    sample_id: str
    task_type: TaskType
    backend: str
    model: str
    model_revision: Optional[str] = None
    status: SampleStatus = SampleStatus.QUEUED
    experiment: Optional[ExperimentRef] = None
    sample_spec: SampleSpec
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    evaluations: list[EvaluationRecord] = Field(default_factory=list)
    exports: list[TrainingExportRecord] = Field(default_factory=list)
    lineage_parent_ids: list[str] = Field(default_factory=list)
    runtime: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
