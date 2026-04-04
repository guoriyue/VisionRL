"""Tests for control-plane schemas."""

from wm_infra.controlplane import (
    ArtifactKind,
    EvaluationRecord,
    EvaluationStatus,
    ExperimentRef,
    FailureTag,
    ProduceSampleRequest,
    SampleRecord,
    SampleSpec,
    SampleStatus,
    TaskType,
)


def test_produce_sample_request_defaults():
    req = ProduceSampleRequest(
        task_type=TaskType.TEXT_TO_VIDEO,
        backend="wan-runtime",
        model="wan2.2-i2v",
        sample_spec=SampleSpec(prompt="a corgi walking in neon tokyo"),
    )

    assert req.task_type == TaskType.TEXT_TO_VIDEO
    assert req.return_artifacts == [ArtifactKind.VIDEO]
    assert req.sample_spec.prompt == "a corgi walking in neon tokyo"



def test_sample_record_can_capture_lineage_and_evaluation():
    record = SampleRecord(
        sample_id="sample_001",
        task_type=TaskType.IMAGE_TO_VIDEO,
        backend="diffusers-runtime",
        model="wan2.2-live2d",
        status=SampleStatus.ACCEPTED,
        experiment=ExperimentRef(experiment_id="exp_live2d", run_id="run_01"),
        sample_spec=SampleSpec(
            prompt="subtle live2d breathing motion",
            references=["asset://character/front.png"],
            width=512,
            height=512,
            fps=12,
        ),
        lineage_parent_ids=["asset_prepare_001"],
        evaluations=[
            EvaluationRecord(
                evaluator="auto_qc_v1",
                status=EvaluationStatus.HUMAN_REVIEW_REQUIRED,
                score=0.81,
                failure_tags=[FailureTag.LOW_MOTION_QUALITY],
            )
        ],
    )

    assert record.status == SampleStatus.ACCEPTED
    assert record.experiment is not None
    assert record.experiment.experiment_id == "exp_live2d"
    assert record.evaluations[0].failure_tags == [FailureTag.LOW_MOTION_QUALITY]
