from __future__ import annotations

import torch
from omegaconf import OmegaConf

from vrl.trainers.checkpointing import (
    TrainingCheckpoint,
    capture_rng_state,
    prepare_metrics_csv,
    prepare_model_config_for_training_resume,
    restore_rng_state,
    sample_prompt_indices,
)


def test_resume_prompt_rng_restore_matches_continuous_sampling() -> None:
    continuous_rng = torch.Generator().manual_seed(123)
    _epoch0 = sample_prompt_indices(
        continuous_rng,
        num_examples=20,
        rollout_batch_size=5,
    )
    saved = capture_rng_state(prompt_generator=continuous_rng)
    epoch1 = sample_prompt_indices(
        continuous_rng,
        num_examples=20,
        rollout_batch_size=5,
    )

    resumed_rng = torch.Generator().manual_seed(999)
    restore_rng_state(saved, prompt_generator=resumed_rng)
    resumed_epoch1 = sample_prompt_indices(
        resumed_rng,
        num_examples=20,
        rollout_batch_size=5,
    )

    assert resumed_epoch1 == epoch1


def test_metrics_csv_append_does_not_rewrite_history(tmp_path) -> None:
    csv_path = tmp_path / "metrics.csv"
    header = "epoch,loss\n"
    csv_path.write_text(header + "0,1.0\n")

    prepare_metrics_csv(csv_path, header, resume=True)

    assert csv_path.read_text() == header + "0,1.0\n"


def test_metrics_csv_new_run_writes_header(tmp_path) -> None:
    csv_path = tmp_path / "metrics.csv"

    prepare_metrics_csv(csv_path, "epoch,loss\n", resume=False)

    assert csv_path.read_text() == "epoch,loss\n"


def test_resume_clears_empty_lora_path_before_model_build(tmp_path) -> None:
    cfg = OmegaConf.create({"model": {"use_lora": True, "lora": {"path": ""}}})
    checkpoint = _training_checkpoint(tmp_path / "checkpoint-2")

    prepare_model_config_for_training_resume(cfg, checkpoint, strict=True)

    assert cfg.model.lora.path == ""


def test_resume_rejects_warm_start_lora_path_in_strict_mode(tmp_path) -> None:
    cfg = OmegaConf.create(
        {"model": {"use_lora": True, "lora": {"path": str(tmp_path / "lora")}}},
    )
    checkpoint = _training_checkpoint(tmp_path / "checkpoint-2")

    import pytest

    with pytest.raises(ValueError, match=r"model\.lora\.path"):
        prepare_model_config_for_training_resume(cfg, checkpoint, strict=True)


def _training_checkpoint(checkpoint_dir) -> TrainingCheckpoint:
    return TrainingCheckpoint(
        checkpoint_dir=checkpoint_dir,
        checkpoint_path=checkpoint_dir / "checkpoint.pt",
        payload={
            "schema_version": 1,
            "family": "sd3_5",
            "trainer": {"step": 2, "global_step": 2},
            "model": {"trainable_modules": {}},
            "progress": {"next_epoch": 2},
            "rng": {},
        },
        meta={"family": "sd3_5", "next_epoch": 2},
    )
